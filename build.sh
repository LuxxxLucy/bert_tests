#!/usr/bin/env bash
# Wrapper for the bench pipeline. Latency uses synthetic inputs, so only the
# models are fetched (no datasets needed).
#
#   ./build.sh fetch              fetch the models listed in manifests/models.yaml
#   ./build.sh dry-run [args]     check it runs + estimate full-run time
#   ./build.sh run [args]         full run (torch fp32) -> results/perf_torch.csv + TABLE.md
#   ./build.sh onnx [args]        full run (onnx fp32)  -> results/perf_onnx.csv  + TABLE.md
#   ./build.sh all                fetch, then dry-run, then run
#
# Extra args pass through to bench.py, e.g. ./build.sh run --threads 16
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

cmd="${1:-help}"; shift || true
case "$cmd" in
  fetch)   uv run python scripts/download.py ;;
  dry-run) uv run python scripts/bench.py --dry-run "$@" ;;
  run)     uv run python scripts/bench.py "$@"; uv run python scripts/make_table.py ;;
  onnx)    uv run python scripts/bench.py --backend onnx "$@"; uv run python scripts/make_table.py ;;
  all)     uv run python scripts/download.py
           uv run python scripts/bench.py --dry-run
           uv run python scripts/bench.py
           uv run python scripts/make_table.py ;;
  *) echo "usage: ./build.sh {fetch|dry-run|run|onnx|all} [extra bench.py args]"; exit 1 ;;
esac
