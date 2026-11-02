from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

PROTOCOL = "configs/experiments/benchmark.yaml"
SPLIT_RECORD = "configs/experiments/pktn_tnui2021_thyroidxl_splits.json"
SPLITS = "manifests/pktn_tnui2021_thyroidxl/splits_prepared.csv"
SCANS = "manifests/pktn_tnui2021_thyroidxl/scan_masks.csv"
PREPARED = "manifests/pktn_tnui2021_thyroidxl/preparation.json"
DATASETS = ("TUCC", "TN3K", "DDTI", "PKTN", "TNUI-2021", "ThyroidXL")

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

    record = json.loads((root / SPLIT_RECORD).read_text())
    if digest(root / record["split_manifest"]) != record["manifest_sha256"]:
        raise ValueError("Frozen raw split manifest changed.")
    settings = yaml.safe_load((root / PROTOCOL).read_text())
    if tuple(settings["datasets"]) != DATASETS:
        raise ValueError("The benchmark protocol must list the six datasets.")
    return settings
