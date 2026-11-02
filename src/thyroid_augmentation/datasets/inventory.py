from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageStat
from scipy import ndimage

from thyroid_augmentation.paths import ProjectPaths, relative_to_root
from thyroid_augmentation.utils import write_csv, write_markdown

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
DATASET_ALIASES = {
    "TUCC": ("tucc", "stanford", "cine"),
    "TN3K": ("tn3k",),
    "DDTI": ("ddti", "tdid"),
}

PAIR_COLUMNS = [
    "dataset",
    "sample_id",
    "image_path",
    "mask_path",
    "group_id",
    "case_id",
    "clip_id",
    "frame_id",
    "panel_id",
    "split_hint",
    "image_width",
    "image_height",
    "mask_width",
    "mask_height",
    "image_mode",
    "mask_mode",
    "image_min",
    "image_max",
    "mask_min",
    "mask_max",
    "mask_nonzero_pixels",
    "mask_area_fraction",
    "mask_touches_edge",
    "mask_components",
    "status",
]

@dataclass(frozen=True)
class DatasetSelection:
    dataset: str
    selected_path: Path | None
    candidates: list[Path]
    reason: str
    blocked: bool

@dataclass(frozen=True)
class AuditResult:
    selections: list[DatasetSelection]
    rows: list[dict[str, object]]
    summary_rows: list[dict[str, object]]
    folder_report: Path
    inventory_report: Path
    inventory_csv: Path
    manifest_csv: Path

def audit_datasets(paths: ProjectPaths, sample_limit: int | None = None) -> AuditResult:
    paths.ensure_output_dirs()
    selections = select_dataset_folders(paths.data_root)
    folder_report = paths.project_root / "reports_FOLDER_SELECTION.md"
    write_markdown(folder_report, _folder_selection_report(selections, paths))

    rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for selection in selections:
        if selection.blocked or selection.selected_path is None:
            summary_rows.append(_blocked_summary(selection))
            continue
        dataset_rows = inspect_dataset(selection.dataset, selection.selected_path, paths, sample_limit)
        rows.extend(dataset_rows)
        summary_rows.append(_summary_from_rows(selection.dataset, selection.selected_path, dataset_rows, paths))

    inventory_csv = paths.project_root / "manifests_inventory.csv"
    write_csv(inventory_csv, summary_rows, list(summary_rows[0].keys()) if summary_rows else ["dataset"])

    manifest_csv = paths.project_root / "manifests" / "pairs.csv"
    write_csv(manifest_csv, rows, PAIR_COLUMNS)

    inventory_report = paths.project_root / "reports_INVENTORY.md"
    write_markdown(inventory_report, _inventory_report(summary_rows, rows))

    quality_report = paths.project_root / "reports_QUALITY_REPORT.md"
    write_markdown(quality_report, _quality_report(summary_rows))

    return AuditResult(selections, rows, summary_rows, folder_report, inventory_report, inventory_csv, manifest_csv)

def select_dataset_folders(data_root: Path) -> list[DatasetSelection]:
    candidates = [p for p in data_root.rglob("*") if p.is_dir() and (p.name == "VSE_CLEAN" or p.name.endswith("_VSE_CLEAN"))]
    selections: list[DatasetSelection] = []
    for dataset, aliases in DATASET_ALIASES.items():
        matched = [p for p in candidates if _matches_dataset(p, aliases)]
        if len(matched) == 1:
            selections.append(
                DatasetSelection(dataset, matched[0], matched, "Exactly one matching VSE_CLEAN folder.", False)
            )
        elif not matched:
            selections.append(DatasetSelection(dataset, None, [], "No matching VSE_CLEAN folder found.", True))
        else:
            selections.append(
                DatasetSelection(dataset, None, matched, "Multiple matching VSE_CLEAN folders; manual selection required.", True)
            )
    return selections

def inspect_dataset(
    dataset: str,
    folder: Path,
    paths: ProjectPaths,
    sample_limit: int | None = None,
) -> list[dict[str, object]]:
    image_files: dict[str, Path] = {}
    mask_files: dict[str, Path] = {}
    all_files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    for file in all_files:
        stem = file.stem
        if stem.endswith("_image"):
            image_files[stem[: -len("_image")]] = file
        elif stem.endswith("_mask"):
            mask_files[stem[: -len("_mask")]] = file

    keys = sorted(set(image_files) | set(mask_files))
    if sample_limit is not None:
        keys = keys[:sample_limit]

    rows: list[dict[str, object]] = []
    for key in keys:
        image_path = image_files.get(key)
        mask_path = mask_files.get(key)
        metadata = parse_sample_metadata(dataset, key)
        row = {
            "dataset": dataset,
            "sample_id": key,
            "image_path": relative_to_root(image_path, paths.project_root) if image_path else "",
            "mask_path": relative_to_root(mask_path, paths.project_root) if mask_path else "",
            **metadata,
            "image_width": "",
            "image_height": "",
            "mask_width": "",
            "mask_height": "",
            "image_mode": "",
            "mask_mode": "",
            "image_min": "",
            "image_max": "",
            "mask_min": "",
            "mask_max": "",
            "mask_nonzero_pixels": "",
            "mask_area_fraction": "",
            "mask_touches_edge": "",
            "mask_components": "",
            "status": "paired" if image_path and mask_path else "missing_mask" if image_path else "missing_image",
        }
        if image_path and mask_path:
            row.update(_inspect_pair(image_path, mask_path))
        rows.append(row)
    return rows

def parse_sample_metadata(dataset: str, sample_id: str) -> dict[str, object]:
    if dataset == "TUCC":
        match = re.match(r"tucc_seq(?P<seq>\d+)_frame(?P<frame>\d+)_idx(?P<idx>\d+)", sample_id)
        if match:
            return {
                "group_id": f"seq{match.group('seq')}",
                "case_id": f"seq{match.group('seq')}",
                "clip_id": f"seq{match.group('seq')}",
                "frame_id": match.group("frame"),
                "panel_id": "",
                "split_hint": "",
            }
    if dataset == "TN3K":
        match = re.match(r"tn3k_(?P<split>trainval|test)_(?P<idx>\d+)", sample_id)
        if match:
            case_id = f"{match.group('split')}_{match.group('idx')}"
            return {
                "group_id": case_id,
                "case_id": case_id,
                "clip_id": "",
                "frame_id": "",
                "panel_id": "",
                "split_hint": match.group("split"),
            }
    if dataset == "DDTI":
        match = re.match(r"ddti_(?P<case>\d+)_(?P<panel>\d+)", sample_id)
        if match:
            return {
                "group_id": match.group("case"),
                "case_id": match.group("case"),
                "clip_id": "",
                "frame_id": "",
                "panel_id": match.group("panel"),
                "split_hint": "",
            }
    return {"group_id": sample_id, "case_id": sample_id, "clip_id": "", "frame_id": "", "panel_id": "", "split_hint": ""}

def _inspect_pair(image_path: Path, mask_path: Path) -> dict[str, object]:
    try:
        with Image.open(image_path) as img, Image.open(mask_path) as mask:
            image_stat = ImageStat.Stat(img.convert("L"))
            mask_gray = mask.convert("L")
            mask_array = np.asarray(mask_gray)
            nonzero = int(np.count_nonzero(mask_array))
            total = int(mask_array.size)
            touches = bool(
                nonzero
                and (
                    np.any(mask_array[0, :] > 0)
                    or np.any(mask_array[-1, :] > 0)
                    or np.any(mask_array[:, 0] > 0)
                    or np.any(mask_array[:, -1] > 0)
                )
            )
            components = count_components(mask_array > 0) if nonzero else 0
            return {
                "image_width": img.width,
                "image_height": img.height,
                "mask_width": mask.width,
                "mask_height": mask.height,
                "image_mode": img.mode,
                "mask_mode": mask.mode,
                "image_min": int(image_stat.extrema[0][0]),
                "image_max": int(image_stat.extrema[0][1]),
                "mask_min": int(mask_array.min()) if total else "",
                "mask_max": int(mask_array.max()) if total else "",
                "mask_nonzero_pixels": nonzero,
                "mask_area_fraction": nonzero / total if total else "",
                "mask_touches_edge": touches,
                "mask_components": components,
                "status": "paired" if img.size == mask.size else "paired_dimension_mismatch",
            }
    except Exception as exc:
        return {"status": f"read_error:{type(exc).__name__}"}

def count_components(mask: np.ndarray) -> int:
    structure = np.ones((3, 3), dtype=np.uint8)
    _, components = ndimage.label(np.asarray(mask, dtype=bool), structure=structure)
    return int(components)

def _matches_dataset(path: Path, aliases: Iterable[str]) -> bool:
    text = " ".join(part.lower() for part in path.parts)
    return any(alias in text for alias in aliases)

def _folder_selection_report(selections: list[DatasetSelection], paths: ProjectPaths) -> str:
    lines = ["# Folder Selection Report", ""]
    for selection in selections:
        lines.extend([f"## {selection.dataset}", "", f"- Blocked: {selection.blocked}", f"- Reason: {selection.reason}"])
        if selection.selected_path:
            lines.append(f"- Selected: `{relative_to_root(selection.selected_path, paths.project_root)}`")
        if selection.candidates:
            lines.append("")
            lines.append("| Candidate | File count | Last modified |")
            lines.append("|---|---:|---|")
            for candidate in selection.candidates:
                files = sum(1 for p in candidate.rglob("*") if p.is_file())
                lines.append(
                    f"| `{relative_to_root(candidate, paths.project_root)}` | {files} | {candidate.stat().st_mtime} |"
                )
        lines.append("")
    return "\n".join(lines)

def _summary_from_rows(dataset: str, folder: Path, rows: list[dict[str, object]], paths: ProjectPaths) -> dict[str, object]:
    statuses = Counter(str(row.get("status", "")) for row in rows)
    paired = [row for row in rows if str(row.get("status", "")).startswith("paired")]
    groups = {str(row.get("group_id", "")) for row in paired if row.get("group_id")}
    clips = {str(row.get("clip_id", "")) for row in paired if row.get("clip_id")}
    frames = [row for row in paired if row.get("frame_id")]
    cases = {str(row.get("case_id", "")) for row in paired if row.get("case_id")}
    empty = sum(1 for row in paired if int(row.get("mask_nonzero_pixels") or 0) == 0)
    nonempty = len(paired) - empty
    edge = sum(1 for row in paired if str(row.get("mask_touches_edge")) == "True")
    panel_counts: defaultdict[str, int] = defaultdict(int)
    for row in paired:
        if row.get("case_id"):
            panel_counts[str(row["case_id"])] += 1
    split_cases = sum(1 for count in panel_counts.values() if count > 1) if dataset == "DDTI" else 0
    return {
        "dataset": dataset,
        "selected_folder": relative_to_root(folder, paths.project_root),
        "images": sum(1 for row in rows if row.get("image_path")),
        "masks": sum(1 for row in rows if row.get("mask_path")),
        "paired_pairs": len(paired),
        "problematic_pairs": len(rows) - len(paired) + statuses.get("paired_dimension_mismatch", 0),
        "missing_masks": statuses.get("missing_mask", 0),
        "missing_images": statuses.get("missing_image", 0),
        "empty_masks": empty,
        "nonempty_masks": nonempty,
        "unique_patients_or_cases": len(cases or groups),
        "unique_clips": len(clips),
        "frames": len(frames),
        "masks_touching_edge": edge,
        "ddti_split_cases": split_cases,
        "manual_review_needed": split_cases,
        "statuses": dict(statuses),
    }

def _blocked_summary(selection: DatasetSelection) -> dict[str, object]:
    return {
        "dataset": selection.dataset,
        "selected_folder": "",
        "images": 0,
        "masks": 0,
        "paired_pairs": 0,
        "problematic_pairs": 0,
        "missing_masks": 0,
        "missing_images": 0,
        "empty_masks": 0,
        "nonempty_masks": 0,
        "unique_patients_or_cases": 0,
        "unique_clips": 0,
        "frames": 0,
        "masks_touching_edge": 0,
        "ddti_split_cases": 0,
        "manual_review_needed": 0,
        "statuses": {"blocked": selection.reason},
    }

def _inventory_report(summary_rows: list[dict[str, object]], rows: list[dict[str, object]]) -> str:
    lines = ["# Inventory Report", "", "| Dataset | Images | Masks | Paired | Problematic | Cases | Clips | Frames | Empty masks | DDTI split cases |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary_rows:
        lines.append(
            f"| {row['dataset']} | {row['images']} | {row['masks']} | {row['paired_pairs']} | "
            f"{row['problematic_pairs']} | {row['unique_patients_or_cases']} | {row['unique_clips']} | "
            f"{row['frames']} | {row['empty_masks']} | {row['ddti_split_cases']} |"
        )
    lines.extend(["", f"Pair-level manifest rows: {len(rows)}", ""])
    return "\n".join(lines)

def _quality_report(summary_rows: list[dict[str, object]]) -> str:
    lines = ["# Quality Report", ""]
    for row in summary_rows:
        ready = row["problematic_pairs"] == 0 and row["paired_pairs"] > 0
        lines.extend(
            [
                f"## {row['dataset']}",
                "",
                f"- Ready for split generation: {ready}",
                f"- Problematic pairs: {row['problematic_pairs']}",
                f"- Empty masks: {row['empty_masks']}",
                f"- Masks touching image edge: {row['masks_touching_edge']}",
                f"- Manual review needed: {row['manual_review_needed']}",
                "",
            ]
        )
    return "\n".join(lines)
