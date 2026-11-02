from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from thyroid_augmentation.paths import ProjectPaths
from thyroid_augmentation.utils import write_markdown

def create_qc_galleries(paths: ProjectPaths, per_dataset: int = 12) -> list[Path]:
    manifest = paths.project_root / "manifests" / "pairs.csv"
    if not manifest.exists():
        raise FileNotFoundError("Run dataset audit before QC gallery generation.")
    rows = [row for row in csv.DictReader(manifest.open("r", encoding="utf-8")) if row.get("status", "").startswith("paired")]
    outputs: list[Path] = []
    for dataset in sorted({row["dataset"] for row in rows}):
        ds_rows = [row for row in rows if row["dataset"] == dataset][:per_dataset]
        if not ds_rows:
            continue
        tiles = [_overlay(paths.project_root / row["image_path"], paths.project_root / row["mask_path"]) for row in ds_rows]
        out = paths.project_root / "qc" / "dataset_audit" / f"{dataset.lower()}_overlay_contact_sheet.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        _contact_sheet(tiles).save(out, quality=90)
        outputs.append(out)
    write_markdown(paths.project_root / "reports_QC_GALLERIES.md", _qc_report(outputs, paths))
    return outputs

def _overlay(image_path: Path, mask_path: Path, size: int = 192) -> Image.Image:
    with Image.open(image_path) as img, Image.open(mask_path) as mask:
        base = img.convert("RGB").resize((size, size))
        mask_arr = np.asarray(mask.convert("L").resize((size, size))) > 0
        arr = np.asarray(base).copy()
        arr[mask_arr, 0] = 255
        arr[mask_arr, 1] = (arr[mask_arr, 1] * 0.35).astype(arr.dtype)
        arr[mask_arr, 2] = (arr[mask_arr, 2] * 0.35).astype(arr.dtype)
        return Image.fromarray(arr)

def _contact_sheet(images: list[Image.Image], columns: int = 4) -> Image.Image:
    if not images:
        return Image.new("RGB", (1, 1), "white")
    width, height = images[0].size
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * height), "white")
    for idx, image in enumerate(images):
        x = (idx % columns) * width
        y = (idx // columns) * height
        sheet.paste(image, (x, y))
    return sheet

def _qc_report(outputs: list[Path], paths: ProjectPaths) -> str:
    lines = ["# QC Galleries", ""]
    if not outputs:
        lines.append("No QC galleries were created.")
    for out in outputs:
        lines.append(f"- `{out.relative_to(paths.project_root).as_posix()}`")
    lines.append("")
    lines.append("QC images are derived from medical images and are ignored by Git.")
    lines.append("")
    return "\n".join(lines)
