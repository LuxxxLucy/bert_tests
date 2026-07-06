"""results/perf_*.csv -> results/TABLE.md : model x length p50 latency (ms)."""
from __future__ import annotations
import csv, glob, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT  # noqa: E402

LENGTHS = [64, 128, 256, 512, 1024, 2048, 4096, 8192]


def main():
    files = sorted(glob.glob(str(ROOT / "results" / "perf_*.csv")))
    if not files:
        print("no results/perf_*.csv yet; run bench.py first")
        return 1
    rows = []
    for f in files:
        rows += list(csv.DictReader(open(f)))
    host = ""  # not stored per-row; leave for the runner to fill in prose
    out = ["# CPU latency of BERT variants\n",
           "P50 latency in milliseconds, batch=1, fp32, by input length (tokens).",
           "Empty cells = length exceeds the model's max position (512-context models).\n"]
    backends = sorted({r["backend"] for r in rows})
    for backend in backends:
        br = [r for r in rows if r["backend"] == backend]
        threads = sorted({r["threads"] for r in br})
        out.append(f"## backend: {backend}  (threads={','.join(threads)})\n")
        out.append("| model | params (M) | max_pos | " +
                   " | ".join(f"{L}" for L in LENGTHS) + " |")
        out.append("|---|--:|--:|" + "|".join("--:" for _ in LENGTHS) + "|")
        seen = {}
        order = []
        for r in br:
            key = r["model"]
            if key not in seen:
                seen[key] = {"params": r["params_M"], "max_pos": r["max_pos"], "cells": {}}
                order.append(key)
            seen[key]["cells"][int(r["target_length"])] = r["p50_ms"]
        order.sort(key=lambda k: float(seen[k]["params"]) if seen[k]["params"] else 0.0)
        for k in order:
            d = seen[k]
            cells = " | ".join(str(d["cells"].get(L, "")) for L in LENGTHS)
            out.append(f"| {k} | {d['params']} | {d['max_pos']} | {cells} |")
        out.append("")
    (ROOT / "results" / "TABLE.md").write_text("\n".join(out) + "\n")
    print(f"wrote {ROOT / 'results' / 'TABLE.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
