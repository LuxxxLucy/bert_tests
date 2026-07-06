# bert_tests

We test the CPU latency of different BERT variants across input lengths.

```bash
uv run python scripts/download.py          # fetch the models
uv run python scripts/bench.py --dry-run   # check it runs + estimate full-run time
uv run python scripts/bench.py             # full run -> results/perf.csv
uv run python scripts/make_table.py        # results/perf.csv -> results/TABLE.md
```

Models are listed in `manifests/models.yaml`. Backends: `--backend torch` (default) and `--backend onnx`.
Lengths default to 64..8192; models capped at 512 positions skip the longer lengths.
