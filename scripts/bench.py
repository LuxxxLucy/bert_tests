"""CPU latency of BERT variants across input lengths.

batch=1, fp32. Backends: torch (eager) and onnx (optimum ORT export, fp32).
Per (model, length): warmup + measured forwards, record p50/p95/p99/mean ms.
Lengths default 64..8192; a model is skipped at lengths above its max_pos.

--dry-run runs a few iters per cell, confirms the pipeline works, and prints an
estimated wall-clock for the full run at --warmup/--measure iters.
"""
from __future__ import annotations
import argparse, csv, gc, os, platform, sys, time
from pathlib import Path
from statistics import mean as smean

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, pick  # noqa: E402

LENGTHS = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
FILL = "The quick brown fox jumps over the lazy dog. Then it ran away quickly. "


def pctl(xs, p):
    s = sorted(xs); k = (len(s) - 1) * p / 100; f = int(k); c = min(f + 1, len(s) - 1)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


def rss_mb():
    return psutil.Process().memory_info().rss / 1048576


def make_input(tok, L):
    text = FILL
    while len(tok(text, add_special_tokens=True)["input_ids"]) < L:
        text += FILL
    return tok(text, return_tensors="pt", add_special_tokens=True,
               truncation=True, max_length=L, padding=False)


def max_pos(cfg):
    return (getattr(cfg, "max_position_embeddings", None)
            or getattr(cfg, "n_positions", None) or 512)


def load_tokenizer(hf_id):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(hf_id, use_fast=True)
    except Exception:  # noqa: BLE001
        return AutoTokenizer.from_pretrained(hf_id, use_fast=False)


class TorchRunner:
    backend = "torch"

    def __init__(self, hf_id):
        from transformers import AutoModelForSequenceClassification
        self.tok = load_tokenizer(hf_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            hf_id, num_labels=3, ignore_mismatched_sizes=True).to("cpu").eval()
        self.cfg = self.model.config
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def run(self, enc):
        with torch.inference_mode():
            self.model(**enc)


class OnnxRunner:
    backend = "onnx"

    def __init__(self, hf_id):
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoConfig
        self.tok = load_tokenizer(hf_id)
        self.cfg = AutoConfig.from_pretrained(hf_id)
        self.model = ORTModelForSequenceClassification.from_pretrained(
            hf_id, export=True, cache_dir=str(ROOT / "onnx_export"))
        self.n_params = 0  # ONNX graph; param count read from torch elsewhere

    def run(self, enc):
        self.model(**{k: v for k, v in enc.items()})


def make_runner(backend, hf_id):
    return (OnnxRunner if backend == "onnx" else TorchRunner)(hf_id)


COLUMNS = ["backend", "model", "hf_id", "params_M", "max_pos", "threads",
           "target_length", "actual_length", "p50_ms", "p95_ms", "p99_ms",
           "mean_ms", "throughput_rps", "load_s", "warmup", "measure", "peak_rss_mb"]


def bench_model(spec, backend, lengths, warmup, measure):
    print(f"\n=== {spec.name}  [{backend}/{spec.group}]  {spec.hf_id}")
    t0 = time.perf_counter()
    r = make_runner(backend, spec.hf_id)
    load_s = time.perf_counter() - t0
    mp = max_pos(r.cfg)
    nprm = round((r.n_params or 0) / 1e6, 2)
    print(f"  loaded {load_s:.1f}s — {nprm}M params, max_pos={mp}")
    # ModernBERT-family use RoPE (no learned position limit) and run the full
    # sweep; DeBERTa/BERT use learned/relative positions capped at max_pos.
    if spec.group == "modernbert":
        valid = list(lengths)
    else:
        valid = [L for L in lengths if L <= mp]
        if len(valid) != len(lengths):
            print(f"  skip lengths > {mp}: {[L for L in lengths if L > mp]}")
    rows = []
    for L in valid:
        enc = make_input(r.tok, L)
        actual = int(enc["input_ids"].shape[1])
        for _ in range(warmup):
            r.run(enc)
        gc.collect(); gc.disable()
        lat = []
        try:
            for _ in range(measure):
                t = time.perf_counter_ns(); r.run(enc); lat.append(time.perf_counter_ns() - t)
        finally:
            gc.enable()
        ms = [x / 1e6 for x in lat]
        rows.append({
            "backend": backend, "model": spec.name, "hf_id": spec.hf_id,
            "params_M": nprm, "max_pos": mp, "threads": torch.get_num_threads(),
            "target_length": L, "actual_length": actual,
            "p50_ms": round(pctl(ms, 50), 3), "p95_ms": round(pctl(ms, 95), 3),
            "p99_ms": round(pctl(ms, 99), 3), "mean_ms": round(smean(ms), 3),
            "throughput_rps": round(1000 / smean(ms), 2),
            "load_s": round(load_s, 2), "warmup": warmup, "measure": measure,
            "peak_rss_mb": round(rss_mb()),
        })
        print(f"  L={L:>5} (act {actual:>5}): p50={rows[-1]['p50_ms']:8.2f}  "
              f"p99={rows[-1]['p99_ms']:8.2f}  mean={rows[-1]['mean_ms']:8.2f}  "
              f"thr={rows[-1]['throughput_rps']:8.1f}/s")
    del r; gc.collect()
    return rows, load_s


def write_csv(rows, path, append):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (not append) or (not path.exists())
    with path.open("a" if append else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if header:
            w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None, help="comma-separated names; overrides --group")
    ap.add_argument("--group", default="all", help="all | modernbert | deberta | baseline")
    ap.add_argument("--backend", default="torch", choices=["torch", "onnx"])
    ap.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--measure", type=int, default=50)
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="few iters per cell, confirm it runs, estimate full-run time")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    picks = pick(args.models, args.group)
    out = Path(args.out) if args.out else ROOT / "results" / f"perf_{args.backend}.csv"

    print(f"host: {platform.platform()} | {platform.processor() or platform.machine()}")
    print(f"torch threads: {torch.get_num_threads()} | backend: {args.backend}")
    print(f"models ({len(picks)}): {[m.name for m in picks]}")
    print(f"lengths: {args.lengths}")

    if args.dry_run:
        dry_w, dry_m = 2, 3
        print(f"\n[dry-run] {dry_w} warmup / {dry_m} measure per cell; "
              f"estimating full run at {args.warmup}/{args.measure}")
        t_start = time.perf_counter()
        total_load = 0.0
        est_full = 0.0
        for spec in picks:
            rows, load_s = bench_model(spec, args.backend, args.lengths, dry_w, dry_m)
            total_load += load_s
            for row in rows:
                per_iter = row["mean_ms"] / 1000
                est_full += (args.warmup + args.measure) * per_iter
        dry_s = time.perf_counter() - t_start
        est_full += total_load  # models reloaded once in the full run
        print(f"\n[dry-run] pipeline OK in {dry_s:.1f}s.")
        print(f"[dry-run] estimated FULL run ({args.warmup}/{args.measure}, "
              f"{len(picks)} models): ~{est_full/60:.1f} min "
              f"({est_full:.0f}s compute + {total_load:.0f}s load).")
        return 0

    all_rows = []
    for i, spec in enumerate(picks):
        try:
            rows, _ = bench_model(spec, args.backend, args.lengths, args.warmup, args.measure)
            all_rows += rows
            write_csv(rows, out, append=(i > 0))
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {spec.name}: {e!r}")
            import traceback; traceback.print_exc()
    print(f"\ndone: {len(all_rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
