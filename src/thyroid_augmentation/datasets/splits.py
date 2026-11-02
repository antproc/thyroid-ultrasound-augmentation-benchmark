from __future__ import annotations

import csv
import random
import re
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit

from thyroid_augmentation.paths import ProjectPaths
from thyroid_augmentation.utils import write_csv, write_markdown

SPLIT_COLUMNS = [
    "dataset",
    "sample_id",
    "group_id",
    "case_id",
    "clip_id",
    "split_hint",
    "split",
    "image_path",
    "mask_path",
]

def create_splits(paths: ProjectPaths, seed: int = 0) -> Path:
    manifest_path = paths.project_root / "manifests" / "pairs.csv"
    if not manifest_path.exists():
        raise FileNotFoundError("Run dataset audit before creating splits.")
    rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8")))
    paired = [row for row in rows if row.get("status", "").startswith("paired")]
    output_rows: list[dict[str, object]] = []
    for dataset in sorted({row["dataset"] for row in paired}):
        dataset_rows = [row for row in paired if row["dataset"] == dataset]
        output_rows.extend(_split_dataset(dataset_rows, dataset, seed))
    out = paths.project_root / "manifests" / "splits.csv"
    write_csv(out, output_rows, SPLIT_COLUMNS)
    write_markdown(paths.project_root / "reports_SPLITS.md", _split_report(output_rows))
    return out

def create_arxiv_stanford_splits(paths: ProjectPaths) -> Path:
    manifest_path = paths.project_root / "manifests" / "pairs.csv"
    if not manifest_path.exists():
        raise FileNotFoundError("Run dataset audit before creating splits.")
    included_patients = _load_stanford_included_patients(paths)
    rows = [
        row
        for row in csv.DictReader(manifest_path.open("r", encoding="utf-8"))
        if row.get("status", "").startswith("paired")
        and row["dataset"] == "TUCC"
        and _tucc_patient_id(row["sample_id"]) in included_patients
    ]
    if not rows:
        raise RuntimeError("No paired TUCC rows found in manifests/pairs.csv.")

    train_rows, val_rows, test_rows = _stanford_group_shuffle_split(rows)
    assignments = {row["sample_id"]: "train" for row in train_rows}
    assignments.update({row["sample_id"]: "val" for row in val_rows})
    assignments.update({row["sample_id"]: "test" for row in test_rows})

    output_rows = [
        {
            "dataset": row["dataset"],
            "sample_id": row["sample_id"],
            "group_id": _tucc_patient_id(row["sample_id"]) or row.get("group_id", ""),
            "case_id": _tucc_patient_id(row["sample_id"]) or row.get("case_id", ""),
            "clip_id": _tucc_patient_id(row["sample_id"]) or row.get("clip_id", ""),
            "split_hint": row.get("split_hint", ""),
            "split": assignments[row["sample_id"]],
            "image_path": row["image_path"],
            "mask_path": row["mask_path"],
        }
        for row in rows
    ]
    out = paths.project_root / "manifests" / "splits_arxiv_stanford_thyroid.csv"
    write_csv(out, output_rows, SPLIT_COLUMNS)
    write_markdown(paths.project_root / "reports_ARXIV_STANFORD_SPLITS.md", _split_report(output_rows))
    return out

def _load_stanford_included_patients(paths: ProjectPaths) -> set[str]:
    patients_path = (
        paths.project_root
        / "external"
        / "ultrasound-augmentation"
        / "usaugment-experiments"
        / "scripts"
        / "data"
        / "thyroid_ultrasound_cineclip"
        / "patients.csv"
    )
    if not patients_path.exists():
        raise FileNotFoundError(f"Missing Stanford Thyroid included-patient list: {patients_path}")
    return {row["patient"] for row in csv.DictReader(patients_path.open("r", encoding="utf-8"))}

def _tucc_patient_id(sample_id: str) -> str:
    match = re.match(r"tucc_seq(?P<patient>\d+)_frame\d+_idx\d+", sample_id)
    return match.group("patient") if match else ""

def _stanford_group_shuffle_split(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    groups = [row.get("group_id") or row["sample_id"] for row in rows]
    indices = list(range(len(rows)))
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_val_indices, test_indices = next(splitter.split(X=indices, groups=groups))

    train_val_rows = [rows[i] for i in train_val_indices]
    train_val_groups = [row.get("group_id") or row["sample_id"] for row in train_val_rows]
    train_val_positions = list(range(len(train_val_rows)))
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_indices, val_indices = next(splitter.split(X=train_val_positions, groups=train_val_groups))

    train_rows = [train_val_rows[i] for i in train_indices]
    val_rows = [train_val_rows[i] for i in val_indices]
    test_rows = [rows[i] for i in test_indices]
    return train_rows, val_rows, test_rows

def _split_dataset(rows: list[dict[str, str]], dataset: str, seed: int) -> list[dict[str, object]]:
    if dataset == "TN3K" and any(row.get("split_hint") == "test" for row in rows):
        trainval = [row for row in rows if row.get("split_hint") == "trainval"]
        test = [row for row in rows if row.get("split_hint") == "test"]
        train, val = _group_split(trainval, seed, val_fraction=0.2)
        assignments = {row["sample_id"]: "train" for row in train}
        assignments.update({row["sample_id"]: "val" for row in val})
        assignments.update({row["sample_id"]: "test" for row in test})
    else:
        train, val, test = _three_way_group_split(rows, seed)
        assignments = {row["sample_id"]: "train" for row in train}
        assignments.update({row["sample_id"]: "val" for row in val})
        assignments.update({row["sample_id"]: "test" for row in test})
    return [
        {
            "dataset": row["dataset"],
            "sample_id": row["sample_id"],
            "group_id": row.get("group_id", ""),
            "case_id": row.get("case_id", ""),
            "clip_id": row.get("clip_id", ""),
            "split_hint": row.get("split_hint", ""),
            "split": assignments[row["sample_id"]],
            "image_path": row["image_path"],
            "mask_path": row["mask_path"],
        }
        for row in rows
    ]

def _group_split(rows: list[dict[str, str]], seed: int, val_fraction: float) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    by_group: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_group[row.get("group_id") or row["sample_id"]].append(row)
    groups = sorted(by_group)
    random.Random(seed).shuffle(groups)
    n_val = max(1, round(len(groups) * val_fraction)) if groups else 0
    val_groups = set(groups[:n_val])
    train = [row for group in groups if group not in val_groups for row in by_group[group]]
    val = [row for group in groups if group in val_groups for row in by_group[group]]
    return train, val

def _three_way_group_split(rows: list[dict[str, str]], seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    by_group: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_group[row.get("group_id") or row["sample_id"]].append(row)
    groups = sorted(by_group)
    random.Random(seed).shuffle(groups)
    n_total = len(groups)
    n_test = max(1, round(n_total * 0.2)) if n_total else 0
    n_val = max(1, round(n_total * 0.2)) if n_total else 0
    test_groups = set(groups[:n_test])
    val_groups = set(groups[n_test : n_test + n_val])
    train_groups = set(groups[n_test + n_val :])
    train = [row for group in groups if group in train_groups for row in by_group[group]]
    val = [row for group in groups if group in val_groups for row in by_group[group]]
    test = [row for group in groups if group in test_groups for row in by_group[group]]
    return train, val, test

def _split_report(rows: list[dict[str, object]]) -> str:
    counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    groups: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        key = (str(row["dataset"]), str(row["split"]))
        counts[key] += 1
        groups[key].add(str(row.get("group_id", "")))
    lines = ["# Split Report", "", "| Dataset | Split | Samples | Groups |", "|---|---|---:|---:|"]
    for dataset, split in sorted(counts):
        lines.append(f"| {dataset} | {split} | {counts[(dataset, split)]} | {len(groups[(dataset, split)])} |")
    lines.extend(["", "Splits are deterministic and grouped by available case/clip identifiers.", ""])
    return "\n".join(lines)
