from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import skimage.morphology
from PIL import Image

from thyroid_augmentation.extension.common import read_csv, write_csv

PAIRS = "manifests/tucc_tn3k_ddti/pairs.csv"
PATIENTS = "external/ultrasound-augmentation/usaugment-experiments/scripts/data/thyroid_ultrasound_cineclip/patients.csv"

def _paired(root: Path) -> list[dict]:
    return [row for row in read_csv(root / PAIRS) if row.get("status", "").startswith("paired")]

def _clip(sample_id: str) -> str:
    match = re.match(r"tucc_seq(?P<clip>\d+)_frame\d+_idx\d+", sample_id)
    return match.group("clip") if match else ""

def _record(root: Path, row: dict, out_path: Path, mask: np.ndarray, size: tuple[int, int]) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(out_path)
    return {"dataset": row["dataset"], "sample_id": row["sample_id"],
            "scan_mask_path": out_path.relative_to(root).as_posix(), "image_path": row["image_path"],
            "width": size[0], "height": size[1], "scan_pixels": int(np.count_nonzero(mask)),
            "scan_fraction": float(np.count_nonzero(mask) / mask.size), "status": "created"}

def prepare_scan_masks(root: Path, datasets: tuple[str, ...] = ("TN3K", "DDTI"), threshold: int = 5, limit: int | None = None) -> Path:
    rows = [row for row in _paired(root) if row["dataset"] in datasets][:limit]
    output = []
    for row in rows:
        with Image.open(root / row["image_path"]) as img:
            gray = np.asarray(img.convert("L"))
            size = img.size
        mask = (gray > threshold).astype(np.uint8) * 255
        out = root / "scan_masks" / row["dataset"].lower() / f"{row['sample_id']}_scan_mask.png"
        output.append(_record(root, row, out, mask, size))
    path = root / "results" / "scan_masks_tn3k_ddti.csv"
    write_csv(path, output)
    return path

def prepare_tucc_scan_masks(root: Path, limit: int | None = None) -> Path:
    patients = {row["patient"] for row in read_csv(root / PATIENTS)}
    rows = [row for row in _paired(root) if row["dataset"] == "TUCC" and _clip(row["sample_id"]) in patients][:limit]
    output = []
    for row in rows:
        with Image.open(root / row["image_path"]) as img:
            gray = np.asarray(img.convert("L"))
            size = img.size
        out = root / "scan_masks_arxiv" / "tucc" / f"{row['sample_id']}_scan_mask.png"
        output.append(_record(root, row, out, _fan_mask(gray), size))
    path = root / "results" / "scan_masks_tucc.csv"
    write_csv(path, output)
    return path

def _fan_mask(image: np.ndarray) -> np.ndarray:
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
