from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

SPLITS = "manifests/extension/splits_prepared.csv"
SCANS = "manifests/extension/scan_masks.csv"
PREPARED = "manifests/extension/preparation.json"

def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))

def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)

def protocol(root: Path) -> dict:
    import yaml

    plan = json.loads((root / "configs/experiments/extension_plan.json").read_text())
    if digest(root / plan["split_manifest"]) != plan["manifest_sha256"]:
        raise ValueError("Frozen raw split manifest changed.")
    path = root / "configs/published/experiments/primary_benchmark.yaml"
    if digest(path) != plan["source_config_sha256"]:
        raise ValueError("Published experiment configuration changed.")
    expected = yaml.safe_load(path.read_text())
    expected.update(name="extension_plan", datasets=["ThyroidXL", "PKTN", "TNUI-2021"])
    expected.pop("approval_gate", None)
    if expected != plan["protocol"]:
        raise ValueError("New experiment settings differ from the frozen published protocol.")
    return plan["protocol"]
