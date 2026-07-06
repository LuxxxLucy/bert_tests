# bert_tests

We test the CPU latency of different BERT variants across input lengths.

```bash
./build.sh fetch      # fetch the models
./build.sh dry-run    # check it runs + estimate full-run time
./build.sh run        # full run (torch fp32) -> results/perf_torch.csv + TABLE.md
./build.sh onnx       # full run (onnx fp32)  -> results/perf_onnx.csv  + TABLE.md
```

Extra args pass through to `bench.py`, e.g. `./build.sh run --threads 16`.

Models are listed in `manifests/models.yaml`. Backends: `--backend torch` (default) and `--backend onnx`.
Lengths default to 64..8192; models capped at 512 positions skip the longer lengths.
