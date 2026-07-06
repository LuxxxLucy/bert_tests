"""Pre-fetch every model in manifests/models.yaml."""
from __future__ import annotations
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import all_models  # noqa: E402

BASE = [
    "config.json", "configuration_*.py", "modeling_*.py",
    "tokenizer*", "tokenization_*.py", "special_tokens_map.json",
    "vocab.*", "merges.txt", "spm.model", "sentencepiece.bpe.model",
]


def weights(files: list[str]) -> list[str]:
    if any(f.endswith(".safetensors") for f in files):
        return ["*.safetensors"]
    return ["pytorch_model*.bin"]


def main() -> int:
    api = HfApi()
    fails = []
    for m in all_models():
        print(f"fetch {m.name:<18} {m.hf_id}")
        try:
            files = api.list_repo_files(m.hf_id)
            snapshot_download(repo_id=m.hf_id, allow_patterns=BASE + weights(files))
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {exc!r}")
            fails.append(m.name)
    print(f"\ndone. {len(all_models()) - len(fails)} ok, {len(fails)} failed: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
