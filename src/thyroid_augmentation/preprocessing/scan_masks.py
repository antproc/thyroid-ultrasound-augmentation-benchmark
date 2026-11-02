from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
from PIL import Image
import skimage.morphology

from thyroid_augmentation.paths import ProjectPaths
from thyroid_augmentation.utils import write_csv, write_markdown

SCAN_MASK_COLUMNS = [
    "dataset",
    "sample_id",
    "scan_mask_path",
    "image_path",
    "width",
    "height",
    "scan_pixels",
    "scan_fraction",
    "status",
]

def prepare_scan_masks(paths: ProjectPaths, threshold: int = 5, max_items: int | None = None) -> Path:
    manifest = paths.project_root / "manifests" / "pairs.csv"
    if not manifest.exists():
        raise FileNotFoundError("Run dataset audit before preparing scan masks.")
    rows = [row for row in csv.DictReader(manifest.open("r", encoding="utf-8")) if row.get("status", "").startswith("paired")]
    if max_items is not None:
        rows = rows[:max_items]
    output_rows: list[dict[str, object]] = []
    for row in rows:
        image_path = paths.project_root / row["image_path"]
        out_dir = paths.project_root / "scan_masks" / row["dataset"].lower()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{row['sample_id']}_scan_mask.png"
        output_rows.append(_make_scan_mask(row, image_path, out_path, paths, threshold))
    csv_path = paths.project_root / "manifests" / "scan_masks.csv"
    write_csv(csv_path, output_rows, SCAN_MASK_COLUMNS)
    report = paths.project_root / "reports_MASK_REPORT.md"
    write_markdown(report, _scan_mask_report(output_rows, threshold))
    return csv_path

def prepare_arxiv_stanford_scan_masks(paths: ProjectPaths, max_items: int | None = None) -> Path:
    manifest = paths.project_root / "manifests" / "pairs.csv"
    if not manifest.exists():
        raise FileNotFoundError("Run dataset audit before preparing scan masks.")
    included_patients = _load_stanford_included_patients(paths)
    rows = [
        row
        for row in csv.DictReader(manifest.open("r", encoding="utf-8"))
        if row.get("status", "").startswith("paired")
        and row["dataset"] == "TUCC"
        and _tucc_patient_id(row["sample_id"]) in included_patients
    ]
    if max_items is not None:
        rows = rows[:max_items]
    output_rows: list[dict[str, object]] = []
    for row in rows:
        image_path = paths.project_root / row["image_path"]
        out_dir = paths.project_root / "scan_masks_arxiv" / "tucc"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{row['sample_id']}_scan_mask.png"
        output_rows.append(_make_arxiv_stanford_scan_mask(row, image_path, out_path, paths))
    csv_path = paths.project_root / "manifests" / "scan_masks_arxiv_stanford_thyroid.csv"
    write_csv(csv_path, output_rows, SCAN_MASK_COLUMNS)
    report = paths.project_root / "reports_ARXIV_STANFORD_SCAN_MASKS.md"
    write_markdown(report, _scan_mask_report(output_rows, threshold=1))
    return csv_path

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

def _make_scan_mask(row: dict[str, str], image_path: Path, out_path: Path, paths: ProjectPaths, threshold: int) -> dict[str, object]:
    try:
        with Image.open(image_path) as img:
            gray = np.asarray(img.convert("L"))
            mask = (gray > threshold).astype(np.uint8) * 255
            Image.fromarray(mask).save(out_path)
            return {
                "dataset": row["dataset"],
                "sample_id": row["sample_id"],
                "scan_mask_path": out_path.relative_to(paths.project_root).as_posix(),
                "image_path": row["image_path"],
                "width": img.width,
                "height": img.height,
                "scan_pixels": int(np.count_nonzero(mask)),
                "scan_fraction": float(np.count_nonzero(mask) / mask.size) if mask.size else 0.0,
                "status": "created",
            }
    except Exception as exc:
        return {
            "dataset": row["dataset"],
            "sample_id": row["sample_id"],
            "scan_mask_path": out_path.relative_to(paths.project_root).as_posix(),
            "image_path": row["image_path"],
            "width": "",
            "height": "",
            "scan_pixels": "",
            "scan_fraction": "",
            "status": f"error:{type(exc).__name__}",
        }

def _make_arxiv_stanford_scan_mask(
    row: dict[str, str], image_path: Path, out_path: Path, paths: ProjectPaths
) -> dict[str, object]:
    try:
        with Image.open(image_path) as img:
            gray = np.asarray(img.convert("L"))
            mask = _generate_arxiv_scan_mask(gray)
            Image.fromarray(mask).save(out_path)
            return {
                "dataset": row["dataset"],
                "sample_id": row["sample_id"],
                "scan_mask_path": out_path.relative_to(paths.project_root).as_posix(),
                "image_path": row["image_path"],
                "width": img.width,
                "height": img.height,
                "scan_pixels": int(np.count_nonzero(mask)),
                "scan_fraction": float(np.count_nonzero(mask) / mask.size) if mask.size else 0.0,
                "status": "created",
            }
    except Exception as exc:
        return {
            "dataset": row["dataset"],
            "sample_id": row["sample_id"],
            "scan_mask_path": out_path.relative_to(paths.project_root).as_posix(),
            "image_path": row["image_path"],
            "width": "",
            "height": "",
            "scan_pixels": "",
            "scan_fraction": "",
            "status": f"error:{type(exc).__name__}",
        }

def _generate_arxiv_scan_mask(image: np.ndarray) -> np.ndarray:
    mask = image > 1
    mask = skimage.morphology.remove_small_holes(mask, area_threshold=64)
    for _ in range(30):
        mask = skimage.morphology.binary_erosion(mask)
    for _ in range(30):
        mask = skimage.morphology.binary_dilation(mask)

    _, width = mask.shape
    midpoint = width // 2
    if width % 2:
        left_half = mask[:, :midpoint]
        middle = mask[:, midpoint : midpoint + 1]
        right_half = mask[:, midpoint + 1 :]
    else:
        left_half = mask[:, :midpoint]
        middle = None
        right_half = mask[:, midpoint:]
    left_half = np.logical_or(left_half, np.fliplr(right_half))
    right_half = np.logical_or(right_half, np.fliplr(left_half))
    mask = np.hstack((left_half, middle, right_half)) if middle is not None else np.hstack((left_half, right_half))
    mask = skimage.morphology.convex_hull_image(mask)
    return mask.astype(np.uint8)

def _scan_mask_report(rows: list[dict[str, object]], threshold: int) -> str:
    created = sum(1 for row in rows if row["status"] == "created")
    failed = len(rows) - created
    lines = [
        "# Scan Mask Report",
        "",
        f"- Threshold: grayscale > {threshold}",
        f"- Created: {created}",
        f"- Failed: {failed}",
        "",
        "| Dataset | Created | Failed | Mean scan fraction |",
        "|---|---:|---:|---:|",
    ]
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        ds = [row for row in rows if row["dataset"] == dataset]
        ok = [float(row["scan_fraction"]) for row in ds if row["status"] == "created"]
        lines.append(f"| {dataset} | {len(ok)} | {len(ds) - len(ok)} | {sum(ok) / len(ok) if ok else 0:.4f} |")
    lines.append("")
    return "\n".join(lines)
