"""Shared model registry loaded from manifests/models.yaml."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class BenchModel:
    name: str
    hf_id: str
    group: str
    note: str = ""


def all_models() -> list[BenchModel]:
    doc = yaml.safe_load((ROOT / "manifests" / "models.yaml").read_text())
    return [BenchModel(**{k: m.get(k, "") for k in ("name", "hf_id", "group", "note")})
            for m in doc["models"]]


def pick(models_arg: str | None, group_arg: str) -> list[BenchModel]:
    mods = all_models()
    if models_arg:
        want = [n.strip() for n in models_arg.split(",") if n.strip()]
        by = {m.name: m for m in mods}
        return [by[n] for n in want]
    if group_arg in ("all", "", None):
        return mods
    return [m for m in mods if m.group == group_arg]
