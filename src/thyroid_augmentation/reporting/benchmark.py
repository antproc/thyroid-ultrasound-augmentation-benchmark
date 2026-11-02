from __future__ import annotations

import csv
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from torch.utils.data import DataLoader

from thyroid_augmentation.augmentations.factory import DISPLAY_NAMES, normalize_augmentation_name
from thyroid_augmentation.models.registry import build_model
from thyroid_augmentation.paths import ProjectPaths
from thyroid_augmentation.training.runner import SegmentationManifestDataset, _evaluate, _resolve_device
from thyroid_augmentation.training.metrics import METRIC_PROTOCOL, require_protocol
from thyroid_augmentation.utils import write_markdown

PRIMARY_MODEL = "monai_efficientnetb0_unet"
DEFAULT_PHASES = [
    "primary_50_individual",
    "primary_50_trivialaugment",
    "primary_50_confirmatory",
    "architecture_50_validation",
]
EXPECTED_PHASE_COUNTS = {
    "primary_50_individual": 135,
    "primary_50_trivialaugment": 117,
    "primary_50_confirmatory": 18,
    "architecture_50_validation": 81,
}
UPSTREAM_ORDER = [
    "none",
    "bilateral_filter",
    "brightness",
    "contrast",
    "depth_attenuation",
    "gamma",
    "gaussian_noise",
    "gaussian_shadow",
    "haze_artifact",
    "horizontal_flip",
    "vertical_flip",
    "random_crop",
    "rotate",
    "translate",
    "zoom",
]

@dataclass(frozen=True)
class BenchmarkReportOutputs:
    combined_csv: Path
    outputs_json: Path
    reports: list[Path]
    tables: list[Path]
    figures: list[Path]

def evaluate_and_report_benchmark(
    paths: ProjectPaths,
    phases: Iterable[str] | None = None,
    model: str = "all",
    device: str = "cuda",
    batch_size: int = 128,
    num_workers: int = 0,
    resume: bool = True,
    dry_run: bool = False,
    evaluate_test: bool = True,
) -> BenchmarkReportOutputs:
    selected_phases = list(phases or DEFAULT_PHASES)
    paths.ensure_output_dirs()
    results_dir = paths.project_root / "results" / "benchmark_50"
    figures_dir = paths.project_root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    records = collect_run_records(paths, selected_phases, model_filter=model)
    if dry_run:
        print(f"Dry run: collected {len(records)} completed summaries.")
    elif evaluate_test:
        for record in records:
            _ensure_test_metrics(paths, record, device, batch_size, num_workers, resume)

    records = collect_run_records(paths, selected_phases, model_filter=model)
    frame = pd.DataFrame(records)
    combined_csv = results_dir / "all_runs.csv"
    if not frame.empty:
        frame.to_csv(combined_csv, index=False)
    else:
        combined_csv.write_text("", encoding="utf-8")

    tables: list[Path] = [combined_csv]
    figures: list[Path] = []
    reports: list[Path] = []

    if not frame.empty:
        tables.extend(_write_individual_outputs(paths, frame))
        tables.extend(_write_trivial_outputs(paths, frame))
        tables.extend(_write_confirmatory_outputs(paths, frame))
        tables.extend(_write_architecture_outputs(paths, frame))
        figures.extend(_write_figures(paths, frame))

    reports.extend(_write_narrative_reports(paths, frame, selected_phases))
    outputs_json = results_dir / "outputs.json"
    outputs = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "phases": selected_phases,
        "combined_csv": combined_csv.as_posix(),
        "tables": [path.as_posix() for path in tables],
        "figures": [path.as_posix() for path in figures],
        "reports": [path.as_posix() for path in reports],
    }
    outputs_json.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    return BenchmarkReportOutputs(combined_csv, outputs_json, reports, tables, figures)

def collect_run_records(
    paths: ProjectPaths,
    phases: Iterable[str] | None = None,
    model_filter: str = "all",
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    selected_phases = list(phases or DEFAULT_PHASES)
    for phase in selected_phases:
        phase_root = paths.project_root / "runs" / phase
        if not phase_root.exists():
            continue
        for summary_path in sorted(phase_root.glob("*/*/*/seed_*/summary.json")):
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if model_filter != "all" and summary.get("model") != model_filter:
                continue
            record = _record_from_summary(paths, phase, summary_path, summary)
            records.append(record)
    return records

def _record_from_summary(
    paths: ProjectPaths,
    phase: str,
    summary_path: Path,
    summary: dict[str, object],
) -> dict[str, object]:
    augmentation = normalize_augmentation_name(str(summary.get("augmentation", "none")))
    if str(summary.get("augmentation", "")).lower().startswith("trivial:"):
        augmentation = str(summary.get("augmentation", ""))
    top_n = summary.get("top_n")
    log_path = _path_from_summary(paths, summary.get("log"))
    best_epoch, val_dice, val_iou = _best_validation_from_log(log_path)
    if val_dice is None:
        val_dice = _as_float(summary.get("best_dice"))
    test_metrics_path = summary_path.parent / "test_metrics_best.json"
    test_metrics = _read_json(test_metrics_path)
    seed = int(summary.get("seed", summary_path.parent.name.replace("seed_", "0")))
    model = str(summary.get("model", ""))
    dataset = str(summary.get("dataset", ""))
    strategy = _strategy_label(phase, augmentation, top_n)
    return {
        "phase": phase,
        "dataset": dataset,
        "model": model,
        "strategy": strategy,
        "augmentation": augmentation,
        "augmentation_display": _display_name(augmentation, top_n),
        "top_n": _none_to_blank(top_n),
        "seed": seed,
        "epochs": int(summary.get("epochs", 0) or 0),
        "scheduler_epochs": int(summary.get("scheduler_epochs", summary.get("epochs", 0)) or 0),
        "train_samples": int(summary.get("train_samples", 0) or 0),
        "val_samples": int(summary.get("val_samples", 0) or 0),
        "test_samples": int(test_metrics.get("samples", 0) or 0) if test_metrics else "",
        "best_val_epoch": best_epoch if best_epoch is not None else "",
        "best_val_dice": val_dice if val_dice is not None else "",
        "best_val_iou": val_iou if val_iou is not None else "",
        "test_dice": test_metrics.get("dice", "") if test_metrics else "",
        "test_iou": test_metrics.get("iou", "") if test_metrics else "",
        "total_seconds": float(summary.get("total_seconds", 0.0) or 0.0),
        "test_seconds": test_metrics.get("seconds", "") if test_metrics else "",
        "checkpoint": str(summary.get("checkpoint", "")),
        "summary": summary_path.as_posix(),
        "log": str(summary.get("log", "")),
        "split_manifest": str(summary.get("split_manifest", "")),
        "scan_mask_manifest": str(summary.get("scan_mask_manifest", "")),
    }

def _ensure_test_metrics(
    paths: ProjectPaths,
    record: dict[str, object],
    device_name: str,
    batch_size: int,
    num_workers: int,
    resume: bool,
) -> None:
    summary_path = Path(str(record["summary"]))
    metrics_path = summary_path.parent / "test_metrics_best.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require_protocol(summary, str(summary_path))
    if resume and metrics_path.exists():
        require_protocol(json.loads(metrics_path.read_text()), str(metrics_path))
        return
    if metrics_path.exists():
        require_protocol(json.loads(metrics_path.read_text()), str(metrics_path))
    split_manifest = _path_from_summary(paths, summary.get("split_manifest"))
    if split_manifest is None or not split_manifest.exists():
        raise FileNotFoundError(f"Missing split manifest for {summary_path}: {split_manifest}")
    rows = [
        row
        for row in csv.DictReader(split_manifest.open("r", encoding="utf-8-sig"))
        if row.get("dataset") == summary.get("dataset") and row.get("split") == "test"
    ]
    if not rows:
        raise RuntimeError(f"No test rows found for {summary.get('dataset')} in {split_manifest}")

    device = _resolve_device(device_name)
    eval_dataset = SegmentationManifestDataset(
        rows,
        paths.project_root,
        int(summary.get("image_size", 224) or 224),
        training=False,
    )
    loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    model = build_model(
        str(summary["model"]),
        in_channels=3,
        classes=1,
        encoder_weights=None,
        dropout=float(summary.get("dropout", 0.0) or 0.0),
        pretrained=False,
    ).to(device)
    checkpoint_path = _path_from_summary(paths, summary.get("checkpoint"))
    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint for {summary_path}: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    start = time.perf_counter()
    metrics = _evaluate(model, loader, device)
    seconds = time.perf_counter() - start
    payload = {
        "metric_protocol": METRIC_PROTOCOL,
        "phase": record["phase"],
        "dataset": summary["dataset"],
        "model": summary["model"],
        "augmentation": summary.get("augmentation", "none"),
        "top_n": summary.get("top_n"),
        "seed": summary["seed"],
        "split": "test",
        "checkpoint": checkpoint_path.as_posix(),
        "dice": metrics["dice"],
        "iou": metrics["iou"],
        "samples": len(rows),
        "batch_size": batch_size,
        "device": str(device),
        "seconds": seconds,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def _write_individual_outputs(paths: ProjectPaths, frame: pd.DataFrame) -> list[Path]:
    subset = frame[(frame["phase"] == "primary_50_individual") & (frame["model"] == PRIMARY_MODEL)]
    if subset.empty:
        return []
    results_dir = paths.project_root / "results" / "benchmark_50"
    summary = _aggregate(subset, ["dataset", "model", "augmentation", "augmentation_display"])
    summary = _add_delta_vs_none(summary, ["dataset", "model"])
    validation = _metric_table(summary, "best_val")
    test = _metric_table(summary, "test")
    paths_out = [
        results_dir / "results_model_individual_augmentations_validation.csv",
        results_dir / "results_model_individual_augmentations_test.csv",
    ]
    validation.to_csv(paths_out[0], index=False)
    test.to_csv(paths_out[1], index=False)
    report_paths = _write_table_bundle(
        paths,
        base_name="reports_INDIVIDUAL_AUGMENTATIONS",
        title="Individual Augmentations",
        table=summary,
        sort_columns=["dataset", "best_val_dice_mean"],
        descending=[False, True],
    )
    return paths_out + report_paths

def _write_trivial_outputs(paths: ProjectPaths, frame: pd.DataFrame) -> list[Path]:
    subset = frame[(frame["phase"] == "primary_50_trivialaugment") & (frame["model"] == PRIMARY_MODEL)]
    if subset.empty:
        return []
    results_dir = paths.project_root / "results" / "benchmark_50"
    summary = _aggregate(subset, ["dataset", "model", "top_n", "augmentation", "augmentation_display"])
    summary = _add_delta_vs_primary_none(frame, summary, ["dataset", "model"])
    csv_path = results_dir / "results_model_trivialaugment_top_n.csv"
    summary.to_csv(csv_path, index=False)
    report_paths = _write_table_bundle(
        paths,
        base_name="reports_TRIVIALAUGMENT_TOP_N",
        title="TrivialAugment Top-N",
        table=summary,
        sort_columns=["dataset", "top_n"],
        descending=[False, False],
    )
    return [csv_path] + report_paths

def _write_confirmatory_outputs(paths: ProjectPaths, frame: pd.DataFrame) -> list[Path]:
    subset = frame[(frame["phase"] == "primary_50_confirmatory") & (frame["model"] == PRIMARY_MODEL)]
    if subset.empty:
        return []
    results_dir = paths.project_root / "results" / "benchmark_50"
    summary = _aggregate(subset, ["dataset", "model", "strategy", "top_n", "augmentation", "augmentation_display"])
    summary = _add_delta_vs_none(summary, ["dataset", "model"])
    csv_path = results_dir / "results_model_confirmatory_finalists.csv"
    summary.to_csv(csv_path, index=False)
    return [csv_path]

def _write_architecture_outputs(paths: ProjectPaths, frame: pd.DataFrame) -> list[Path]:
    subset = frame[frame["phase"] == "architecture_50_validation"]
    if subset.empty:
        return []
    results_dir = paths.project_root / "results" / "benchmark_50"
    summary = _aggregate(subset, ["dataset", "model", "strategy", "top_n", "augmentation", "augmentation_display"])
    summary = _add_delta_vs_none(summary, ["dataset", "model"])
    csv_path = results_dir / "results_architecture_validation.csv"
    summary.to_csv(csv_path, index=False)
    return [csv_path]

def _aggregate(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    numeric = frame.copy()
    for col in ["best_val_dice", "best_val_iou", "test_dice", "test_iou", "total_seconds"]:
        numeric[col] = pd.to_numeric(numeric[col], errors="coerce")
    grouped = numeric.groupby(group_cols, dropna=False)
    result = grouped.agg(
        n_seeds=("seed", "nunique"),
        best_val_dice_mean=("best_val_dice", "mean"),
        best_val_dice_sd=("best_val_dice", _sample_sd),
        best_val_iou_mean=("best_val_iou", "mean"),
        best_val_iou_sd=("best_val_iou", _sample_sd),
        test_dice_mean=("test_dice", "mean"),
        test_dice_sd=("test_dice", _sample_sd),
        test_iou_mean=("test_iou", "mean"),
        test_iou_sd=("test_iou", _sample_sd),
        total_hours=("total_seconds", lambda values: float(values.sum()) / 3600.0),
    ).reset_index()
    for metric in ["best_val_dice", "best_val_iou", "test_dice", "test_iou"]:
        result[f"{metric}_mean_sd"] = [
            _mean_sd(mean, sd)
            for mean, sd in zip(result[f"{metric}_mean"], result[f"{metric}_sd"], strict=False)
        ]
    return result

def _add_delta_vs_none(frame: pd.DataFrame, baseline_cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    out["delta_val_pct_vs_none"] = ""
    out["delta_test_pct_vs_none"] = ""
    for key, group in out.groupby(baseline_cols, dropna=False):
        baseline = group[group["augmentation"] == "none"]
        if baseline.empty:
            continue
        base_val = float(baseline.iloc[0]["best_val_dice_mean"])
        base_test = _safe_float(baseline.iloc[0].get("test_dice_mean"))
        mask = pd.Series(True, index=out.index)
        if not isinstance(key, tuple):
            key = (key,)
        for col, value in zip(baseline_cols, key, strict=False):
            mask &= out[col] == value
        if base_val:
            out.loc[mask, "delta_val_pct_vs_none"] = (
                (out.loc[mask, "best_val_dice_mean"].astype(float) - base_val) / base_val * 100.0
            )
        if base_test and not math.isnan(base_test):
            out.loc[mask, "delta_test_pct_vs_none"] = (
                (out.loc[mask, "test_dice_mean"].astype(float) - base_test) / base_test * 100.0
            )
    return out

def _add_delta_vs_primary_none(
    all_runs: pd.DataFrame,
    summary: pd.DataFrame,
    baseline_cols: list[str],
) -> pd.DataFrame:
    baseline = all_runs[
        (all_runs["phase"] == "primary_50_individual")
        & (all_runs["model"] == PRIMARY_MODEL)
        & (all_runs["augmentation"] == "none")
    ]
    if baseline.empty:
        return summary
    baseline_summary = _aggregate(baseline, baseline_cols)
    out = summary.copy()
    out["delta_val_pct_vs_none"] = ""
    out["delta_test_pct_vs_none"] = ""
    for _, base in baseline_summary.iterrows():
        mask = pd.Series(True, index=out.index)
        for col in baseline_cols:
            mask &= out[col] == base[col]
        base_val = float(base["best_val_dice_mean"])
        base_test = _safe_float(base.get("test_dice_mean"))
        if base_val:
            out.loc[mask, "delta_val_pct_vs_none"] = (
                (out.loc[mask, "best_val_dice_mean"].astype(float) - base_val) / base_val * 100.0
            )
        if base_test and not math.isnan(base_test):
            out.loc[mask, "delta_test_pct_vs_none"] = (
                (out.loc[mask, "test_dice_mean"].astype(float) - base_test) / base_test * 100.0
            )
    return out

def _metric_table(summary: pd.DataFrame, prefix: str) -> pd.DataFrame:
    cols = [
        "dataset",
        "model",
        "augmentation",
        "augmentation_display",
        "n_seeds",
        f"{prefix}_dice_mean",
        f"{prefix}_dice_sd",
        f"{prefix}_dice_mean_sd",
    ]
    delta_col = "delta_val_pct_vs_none" if prefix == "best_val" else "delta_test_pct_vs_none"
    if delta_col in summary.columns:
        cols.append(delta_col)
    return summary[[col for col in cols if col in summary.columns]].copy()

def _write_table_bundle(
    paths: ProjectPaths,
    base_name: str,
    title: str,
    table: pd.DataFrame,
    sort_columns: list[str],
    descending: list[bool],
) -> list[Path]:
    ordered = table.sort_values(
        by=sort_columns,
        ascending=[not value for value in descending],
        na_position="last",
    )
    md_path = paths.project_root / f"{base_name}.md"
    tex_path = paths.project_root / f"{base_name}.tex"
    xlsx_path = paths.project_root / f"{base_name}.xlsx"
    lines = [f"# {title}", "", _dataframe_to_markdown(ordered), ""]
    write_markdown(md_path, "\n".join(lines))
    tex_path.write_text(ordered.to_latex(index=False, float_format="%.4f"), encoding="utf-8")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        ordered.to_excel(writer, index=False, sheet_name="summary")
    return [md_path, tex_path, xlsx_path]

def _write_figures(paths: ProjectPaths, frame: pd.DataFrame) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputs: list[Path] = []
    individual = frame[(frame["phase"] == "primary_50_individual") & (frame["model"] == PRIMARY_MODEL)]
    if not individual.empty:
        results_dir = paths.project_root / "results" / "benchmark_50"
        summary = _aggregate(individual, ["dataset", "augmentation", "augmentation_display"])
        csv_path = results_dir / "figures_individual_augmentations.csv"
        summary.to_csv(csv_path, index=False)
        outputs.append(csv_path)
        outputs.extend(
            _plot_individual(
                paths,
                summary,
                "figures_individual_augmentations_upstream_order",
                order=UPSTREAM_ORDER,
                title="Individual augmentations, upstream order",
            )
        )
        ranked_order = (
            summary.groupby("augmentation", as_index=False)["best_val_dice_mean"]
            .mean()
            .sort_values("best_val_dice_mean", ascending=True)["augmentation"]
            .tolist()
        )
        outputs.extend(
            _plot_individual(
                paths,
                summary,
                "figures_individual_augmentations_ranked",
                order=ranked_order,
                title="Individual augmentations, ranked by validation Dice",
            )
        )

    trivial = frame[(frame["phase"] == "primary_50_trivialaugment") & (frame["model"] == PRIMARY_MODEL)]
    if not trivial.empty:
        summary = _aggregate(trivial, ["dataset", "top_n", "augmentation_display"])
        csv_path = paths.project_root / "results" / "benchmark_50" / "figures_trivialaugment_top_n.csv"
        summary.to_csv(csv_path, index=False)
        outputs.append(csv_path)
        outputs.extend(_plot_trivial(paths, summary))
    return outputs

def _plot_individual(
    paths: ProjectPaths,
    summary: pd.DataFrame,
    name: str,
    order: list[str],
    title: str,
) -> list[Path]:
    import matplotlib.pyplot as plt

    datasets = sorted(summary["dataset"].unique())
    fig, axes = plt.subplots(len(datasets), 1, figsize=(12, max(3.2, 2.7 * len(datasets))), sharex=True)
    if len(datasets) == 1:
        axes = [axes]
    order = [aug for aug in order if aug in set(summary["augmentation"])]
    x = list(range(len(order)))
    for axis, dataset in zip(axes, datasets, strict=False):
        ds = summary[summary["dataset"] == dataset].set_index("augmentation")
        means = [float(ds.loc[aug, "best_val_dice_mean"]) if aug in ds.index else math.nan for aug in order]
        sds = [float(ds.loc[aug, "best_val_dice_sd"]) if aug in ds.index else 0.0 for aug in order]
        axis.errorbar(x, means, yerr=sds, fmt="o", capsize=3, color="#2f6fb0", ecolor="#8099b8")
        axis.set_ylabel(f"{dataset}\nVal Dice")
        axis.grid(True, axis="y", alpha=0.3)
    labels = [_display_name(aug, None) for aug in order]
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=65, ha="right")
    fig.suptitle(title)
    fig.tight_layout()
    png = paths.project_root / "figures" / f"{name}.png"
    svg = paths.project_root / "figures" / f"{name}.svg"
    fig.savefig(png, dpi=220)
    fig.savefig(svg)
    plt.close(fig)
    return [png, svg]

def _plot_trivial(paths: ProjectPaths, summary: pd.DataFrame) -> list[Path]:
    import matplotlib.pyplot as plt

    datasets = sorted(summary["dataset"].unique())
    fig, axis = plt.subplots(figsize=(9, 5))
    for dataset in datasets:
        ds = summary[summary["dataset"] == dataset].sort_values("top_n")
        axis.errorbar(
            pd.to_numeric(ds["top_n"]),
            ds["best_val_dice_mean"],
            yerr=ds["best_val_dice_sd"].fillna(0.0),
            marker="o",
            capsize=3,
            label=dataset,
        )
    axis.set_xlabel("Top-N pool size")
    axis.set_ylabel("Validation Dice")
    axis.set_title("TrivialAugment Top-N")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    png = paths.project_root / "figures" / "figures_trivialaugment_top_n.png"
    svg = paths.project_root / "figures" / "figures_trivialaugment_top_n.svg"
    fig.savefig(png, dpi=220)
    fig.savefig(svg)
    plt.close(fig)
    return [png, svg]

def _write_narrative_reports(paths: ProjectPaths, frame: pd.DataFrame, phases: list[str]) -> list[Path]:
    reports = [
        paths.project_root / "reports_RESULTS.md",
        paths.project_root / "reports_ANALYSIS.md",
        paths.project_root / "reports_DRAFT_CZ.md",
        paths.project_root / "reports_DRAFT_EN.md",
    ]
    counts = _phase_counts(frame)
    best_individual = _best_by_validation(frame, "primary_50_individual", ["dataset"])
    best_trivial = _best_by_validation(frame, "primary_50_trivialaugment", ["dataset"])
    best_arch = _best_by_validation(frame, "architecture_50_validation", ["dataset", "model"])

    write_markdown(
        reports[0],
        "\n".join(
            [
                "# Results",
                "",
                f"Generated: {datetime.now().isoformat(timespec='seconds')}",
                "",
                "## Completion",
                "",
                _completion_table(counts, phases),
                "",
                "## Best individual augmentations by validation Dice",
                "",
                _dataframe_to_markdown(best_individual),
                "",
                "## Best TrivialAugment Top-N by validation Dice",
                "",
                _dataframe_to_markdown(best_trivial),
                "",
                "## Architecture validation winners by validation Dice",
                "",
                _dataframe_to_markdown(best_arch),
                "",
                "Selection is based on validation Dice. Test Dice is reported after selection only.",
                "",
            ]
        ),
    )
    write_markdown(
        reports[1],
        _analysis_text(frame, counts),
    )
    write_markdown(
        reports[2],
        _draft_text(frame, language="cz"),
    )
    write_markdown(
        reports[3],
        _draft_text(frame, language="en"),
    )
    return reports

def _phase_counts(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty:
        return {}
    return {str(phase): int(count) for phase, count in frame.groupby("phase").size().items()}

def _completion_table(counts: dict[str, int], phases: list[str]) -> str:
    rows = ["| Phase | Completed summaries | Planned |", "|---|---:|---:|"]
    for phase in phases:
        rows.append(f"| `{phase}` | {counts.get(phase, 0)} | {EXPECTED_PHASE_COUNTS.get(phase, '')} |")
    return "\n".join(rows)

def _best_by_validation(frame: pd.DataFrame, phase: str, group_cols: list[str]) -> pd.DataFrame:
    if frame.empty or "phase" not in frame.columns:
        return pd.DataFrame()
    subset = frame[frame["phase"] == phase]
    if subset.empty:
        return pd.DataFrame()
    base_cols = list(dict.fromkeys(group_cols))
    aggregate_cols = list(dict.fromkeys(base_cols + ["model", "strategy", "top_n", "augmentation", "augmentation_display"]))
    summary = _aggregate(subset, aggregate_cols)
    winners = []
    for _, group in summary.groupby(base_cols, dropna=False):
        winners.append(group.sort_values("best_val_dice_mean", ascending=False).iloc[0])
    return pd.DataFrame(winners) if winners else pd.DataFrame()

def _analysis_text(frame: pd.DataFrame, counts: dict[str, int]) -> str:
    lines = [
        "# Analysis Notes",
        "",
        "The analysis ranks augmentation choices by validation Dice. Test Dice is used only after the augmentation or policy has been selected.",
        "",
        "## Current completion",
        "",
        _completion_table(counts, DEFAULT_PHASES),
        "",
    ]
    if frame.empty or "phase" not in frame.columns:
        lines.extend(["No completed benchmark rows were found yet.", ""])
        return "\n".join(lines)
    individual = frame[(frame["phase"] == "primary_50_individual") & (frame["model"] == PRIMARY_MODEL)]
    if individual.empty:
        lines.extend(["Individual augmentation results are not complete yet.", ""])
        return "\n".join(lines)
    summary = _aggregate(individual, ["dataset", "augmentation", "augmentation_display"])
    for dataset in sorted(summary["dataset"].unique()):
        ds = summary[summary["dataset"] == dataset].sort_values("best_val_dice_mean", ascending=False)
        top = ds.head(5)
        bottom = ds.tail(3)
        lines.extend([f"## {dataset}", "", "Top validation augmentations:", ""])
        lines.append(_dataframe_to_markdown(top[["augmentation_display", "n_seeds", "best_val_dice_mean_sd", "test_dice_mean_sd"]]))
        lines.extend(["", "Lowest validation augmentations:", ""])
        lines.append(_dataframe_to_markdown(bottom[["augmentation_display", "n_seeds", "best_val_dice_mean_sd", "test_dice_mean_sd"]]))
        lines.append("")
    lines.extend(
        [
            "## Pattern to discuss",
            "",
            "After all datasets finish, compare whether high-ranking policies cluster around geometric changes, intensity changes, or ultrasound-specific artifacts. Report this as a cross-dataset pattern only when it is visible on validation metrics across seeds.",
            "",
        ]
    )
    return "\n".join(lines)

def _draft_text(frame: pd.DataFrame, language: str) -> str:
    counts = _phase_counts(frame)
    complete_primary = counts.get("primary_50_individual", 0) >= EXPECTED_PHASE_COUNTS["primary_50_individual"]
    complete_trivial = counts.get("primary_50_trivialaugment", 0) >= EXPECTED_PHASE_COUNTS["primary_50_trivialaugment"]
    if language == "cz":
        title = "# Pracovni rukopis"
        body = [
            "## Metody",
            "",
            "Experiment porovnava jednotlive augmentace a navazujici TrivialAugment Top-N policy na datasetech TUCC, TN3K a DDTI. Vyber nejlepsich augmentaci je proveden vyhradne podle validacniho Dice; testovaci sada slouzi az k finalnimu reportovani.",
            "",
            "## Vysledky",
            "",
            f"Primarni jednotlive augmentace hotove: {counts.get('primary_50_individual', 0)} / {EXPECTED_PHASE_COUNTS['primary_50_individual']}.",
            f"TrivialAugment Top-N hotove: {counts.get('primary_50_trivialaugment', 0)} / {EXPECTED_PHASE_COUNTS['primary_50_trivialaugment']}.",
            "",
            "## Diskuse",
            "",
            "Navazani na praci Tupper a Gagne bude stat na stejne logice: nejprve nezavisle zmerit jednotlive augmentace, pote skladat Top-N pool podle validacniho poradi a nakonec overit, zda se zlepseni prenasi napric datasety a architekturami.",
            "",
        ]
        if complete_primary and complete_trivial:
            body.append("Primarni benchmark je kompletni a lze doplnit definitivni ciselne zneni vysledku.")
        return "\n".join([title, "", *body])
    title = "# Working Manuscript"
    body = [
        "## Methods",
        "",
        "The experiment compares individual augmentations and validation-ranked TrivialAugment Top-N policies on TUCC, TN3K, and DDTI. Augmentation selection is based only on validation Dice; test Dice is used for final reporting after selection.",
        "",
        "## Results",
        "",
        f"Completed individual augmentation runs: {counts.get('primary_50_individual', 0)} / {EXPECTED_PHASE_COUNTS['primary_50_individual']}.",
        f"Completed TrivialAugment Top-N runs: {counts.get('primary_50_trivialaugment', 0)} / {EXPECTED_PHASE_COUNTS['primary_50_trivialaugment']}.",
        "",
        "## Discussion",
        "",
        "The study extends the Tupper and Gagne design by applying the same validation-first augmentation ranking logic across three thyroid ultrasound datasets and then testing whether the selected policies transfer across segmentation architectures.",
        "",
    ]
    if complete_primary and complete_trivial:
        body.append("The primary benchmark is complete and the final numeric results can be inserted.")
    return "\n".join([title, "", *body])

def _best_validation_from_log(log_path: Path | None) -> tuple[int | None, float | None, float | None]:
    if log_path is None or not log_path.exists():
        return None, None, None
    rows = list(csv.DictReader(log_path.open("r", encoding="utf-8")))
    if not rows:
        return None, None, None
    best = max(rows, key=lambda row: float(row.get("val_dice", 0.0) or 0.0))
    return (
        int(float(best.get("epoch", 0) or 0)),
        float(best.get("val_dice", 0.0) or 0.0),
        float(best.get("val_iou", 0.0) or 0.0),
    )

def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _path_from_summary(paths: ProjectPaths, value: object) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = paths.project_root / path
    return path

def _display_name(augmentation: str, top_n: object | None) -> str:
    if top_n not in (None, "", "None"):
        return f"TrivialAugment Top-{top_n}"
    normalized = normalize_augmentation_name(augmentation)
    return DISPLAY_NAMES.get(normalized, augmentation)

def _strategy_label(phase: str, augmentation: str, top_n: object | None) -> str:
    if top_n not in (None, "", "None"):
        return "trivialaugment"
    if normalize_augmentation_name(augmentation) == "none":
        return "none"
    if phase in {"primary_50_confirmatory", "architecture_50_validation"}:
        return "best_single"
    return "individual"

def _sample_sd(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1))

def _mean_sd(mean: object, sd: object) -> str:
    mean_float = _safe_float(mean)
    sd_float = _safe_float(sd)
    if mean_float is None or math.isnan(mean_float):
        return ""
    if sd_float is None or math.isnan(sd_float):
        sd_float = 0.0
    return f"{mean_float:.4f} +/- {sd_float:.4f}"

def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _as_float(value: object) -> float | None:
    converted = _safe_float(value)
    return converted

def _none_to_blank(value: object) -> object:
    return "" if value is None else value

def _dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No completed rows yet._"
    display = frame.copy()
    max_rows = 60
    if len(display) > max_rows:
        display = display.head(max_rows)
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    columns = [str(col) for col in display.columns]
    rows = []
    for _, row in display.iterrows():
        rows.append([_markdown_cell(row[col]) for col in display.columns])
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])

def _markdown_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("|", "\\|")
