from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from thyroid_augmentation.paths import ProjectPaths
from thyroid_augmentation.utils import write_csv, write_markdown

PANEL_COLUMNS = [
    "dataset",
    "case_id",
    "panel_count",
    "sample_ids",
    "requires_manual_review",
    "reason",
]

def create_ddti_panel_maps(paths: ProjectPaths) -> Path:
    manifest = paths.project_root / "manifests" / "pairs.csv"
    if not manifest.exists():
        raise FileNotFoundError("Run dataset audit before preparing DDTI panel maps.")
    rows = [row for row in csv.DictReader(manifest.open("r", encoding="utf-8")) if row.get("dataset") == "DDTI"]
    by_case: defaultdict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row.get("status", "").startswith("paired"):
            by_case[row.get("case_id") or row["sample_id"]].append(row["sample_id"])
    output_rows = []
    for case_id, sample_ids in sorted(by_case.items()):
        panel_count = len(sample_ids)
        output_rows.append(
            {
                "dataset": "DDTI",
                "case_id": case_id,
                "panel_count": panel_count,
                "sample_ids": ";".join(sorted(sample_ids)),
                "requires_manual_review": panel_count > 1,
                "reason": "multiple derived panels for one DDTI case" if panel_count > 1 else "single panel",
            }
        )
    out = paths.project_root / "panel_maps" / "ddti_panel_map.csv"
    write_csv(out, output_rows, PANEL_COLUMNS)
    write_markdown(paths.project_root / "reports_SPLIT_SCREEN_REPORT.md", _panel_report(output_rows))
    return out

def _panel_report(rows: list[dict[str, object]]) -> str:
    split_cases = sum(1 for row in rows if str(row["requires_manual_review"]) == "True")
    lines = [
        "# DDTI Split-Screen / Panel Report",
        "",
        f"- DDTI cases: {len(rows)}",
        f"- Cases with multiple derived panels: {split_cases}",
        f"- Cases requiring manual review: {split_cases}",
        "",
        "The `VSE_CLEAN` folder appears to contain panel-level files named `ddti_CASE_PANEL_image.png`.",
        "Cases with more than one panel are flagged so panel-aware augmentations can avoid treating panels as independent patients.",
        "",
    ]
    return "\n".join(lines)
