from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from thyroid_augmentation.augmentations.factory import DISPLAY_NAMES, normalize_augmentation_name
from thyroid_augmentation.paths import ProjectPaths

@dataclass(frozen=True)
class RankingOutputs:
    ranking_csv: Path
    policies_csv: Path
    report_md: Path

def rank_individual_augmentations(
    paths: ProjectPaths,
    model: str,
    phase: str = "individual_augmentations",
    top_n_min: int = 2,
    top_n_max: int = 14,
) -> RankingOutputs:
    run_root = paths.project_root / "runs" / phase
    results_dir = paths.project_root / "results" / "rankings"
    reports_dir = paths.project_root
    results_dir.mkdir(parents=True, exist_ok=True)

    runs = _collect_runs(run_root, model)
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for run in runs:
        grouped[(str(run["dataset"]), str(run["augmentation"]))].append(run)

    ranking_rows: list[dict[str, object]] = []
    for (dataset, augmentation), rows in grouped.items():
        best_dices = [float(row["best_dice"]) for row in rows]
        best_ious = [float(row["best_iou"]) for row in rows]
        ranking_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "augmentation": augmentation,
                "display_name": DISPLAY_NAMES.get(augmentation, augmentation),
                "n_seeds": len(rows),
                "mean_best_dice": sum(best_dices) / len(best_dices),
                "mean_best_iou": sum(best_ious) / len(best_ious),
                "best_dice_max": max(best_dices),
                "best_dice_min": min(best_dices),
            }
        )

    ranking_rows = sorted(
        ranking_rows,
        key=lambda row: (str(row["dataset"]), -float(row["mean_best_dice"]), str(row["augmentation"])),
    )
    ranked_with_position: list[dict[str, object]] = []
    for dataset in sorted({str(row["dataset"]) for row in ranking_rows}):
        rank = 1
        for row in [r for r in ranking_rows if r["dataset"] == dataset]:
            row = dict(row)
            row["rank"] = rank
            ranked_with_position.append(row)
            rank += 1

    policies = _make_topn_policies(ranked_with_position, model, top_n_min, top_n_max)

    ranking_csv = results_dir / "individual_augmentation_ranking.csv"
    policies_csv = results_dir / "topn_policies.csv"
    report_md = reports_dir / "reports_AUGMENTATION_RANKING.md"
    _write_csv(ranking_csv, ranked_with_position)
    _write_csv(policies_csv, policies)
    _write_report(report_md, ranked_with_position, policies, phase)
    return RankingOutputs(ranking_csv=ranking_csv, policies_csv=policies_csv, report_md=report_md)

def _collect_runs(run_root: Path, model: str) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    if not run_root.exists():
        return runs
    for log_path in run_root.glob(f"*/{model}/*/seed_*/train_log.csv"):
        parts = log_path.relative_to(run_root).parts
        if len(parts) < 5:
            continue
        dataset, _, augmentation, seed_dir = parts[:4]
        rows = list(csv.DictReader(log_path.open("r", encoding="utf-8")))
        if not rows:
            continue
        best = max(rows, key=lambda row: float(row["val_dice"]))
        runs.append(
            {
                "dataset": dataset,
                "model": model,
                "augmentation": normalize_augmentation_name(augmentation.replace("__", "+")),
                "seed": seed_dir.replace("seed_", ""),
                "epochs": len(rows),
                "best_epoch": int(best["epoch"]),
                "best_dice": float(best["val_dice"]),
                "best_iou": float(best["val_iou"]),
                "log": str(log_path),
            }
        )
    return runs

def _make_topn_policies(
    ranking_rows: list[dict[str, object]],
    model: str,
    top_n_min: int,
    top_n_max: int,
) -> list[dict[str, object]]:
    policies: list[dict[str, object]] = []
    datasets = sorted({str(row["dataset"]) for row in ranking_rows})
    for dataset in datasets:
        ranked_augs = [
            str(row["augmentation"])
            for row in sorted(
                [r for r in ranking_rows if r["dataset"] == dataset and r["augmentation"] != "none"],
                key=lambda row: int(row["rank"]),
            )
        ]
        upper = min(top_n_max, len(ranked_augs))
        for top_n in range(top_n_min, upper + 1):
            selected = ranked_augs[:top_n]
            policies.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "top_n": top_n,
                    "augmentation": "+".join(selected),
                    "selected_augmentations": ",".join(selected),
                }
            )
    return policies

def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def _write_report(
    path: Path,
    ranking_rows: list[dict[str, object]],
    policies: list[dict[str, object]],
    phase: str,
) -> None:
    lines = [
        "# Augmentation Ranking",
        "",
        f"- Source phase: `{phase}`",
        f"- Ranked individual augmentation rows: {len(ranking_rows)}",
        f"- Top-N policies: {len(policies)}",
        "",
    ]
    for dataset in sorted({str(row["dataset"]) for row in ranking_rows}):
        lines += [f"## {dataset}", "", "| Rank | Augmentation | Seeds | Mean best Dice | Mean best IoU |"]
        lines += ["|---:|---|---:|---:|---:|"]
        for row in sorted(
            [r for r in ranking_rows if r["dataset"] == dataset],
            key=lambda row: int(row["rank"]),
        ):
            lines.append(
                "| {rank} | {display_name} | {n_seeds} | {dice:.4f} | {iou:.4f} |".format(
                    rank=row["rank"],
                    display_name=row["display_name"],
                    n_seeds=row["n_seeds"],
                    dice=float(row["mean_best_dice"]),
                    iou=float(row["mean_best_iou"]),
                )
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
