from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from thyroid_augmentation.augmentations.factory import DISPLAY_NAMES, normalize_augmentation_name
from thyroid_augmentation.models.registry import build_model
from thyroid_augmentation.paths import ProjectPaths
from thyroid_augmentation.training.runner import (
    SegmentationManifestDataset,
    _evaluate,
    _load_scan_mask_map,
    _resolve_device,
)

DATASETS = ("TUCC", "TN3K", "DDTI")
ARCHITECTURES = ("res34_unet", "deeplabv3plus_resnet50", "mit_b2_unet")
SEEDS = (0, 1, 2)
STRATEGIES = ("baseline", "best_single", "trivialaugment")
LOO_STRATEGIES = ("baseline", "best_single_consensus", "trivialaugment_consensus")
PHASE_ARCH_INDIVIDUAL = "architecture_full_50_individual"
PHASE_ARCH_TRIVIAL = "architecture_full_50_trivialaugment"
PHASE_LOO_INDIVIDUAL = "ood_leave_one_out_50_individual"
PHASE_LOO_TRIVIAL = "ood_leave_one_out_50_trivialaugment"
PHASE_LOO_CONSENSUS = "ood_leave_one_out_consensus_50"

MODEL_DISPLAY = {
    "res34_unet": "ResNet34-U-Net",
    "deeplabv3plus_resnet50": "DeepLabV3+",
    "mit_b2_unet": "MiT-B2-U-Net",
}

LEAVE_ONE_OUT_SPECS = [
    ("TN3K_DDTI_to_TUCC", ("TN3K", "DDTI"), "TUCC"),
    ("TN3K_TUCC_to_DDTI", ("TN3K", "TUCC"), "DDTI"),
    ("DDTI_TUCC_to_TN3K", ("DDTI", "TUCC"), "TN3K"),
]

RESULT_COLUMNS_SINGLE = [
    "source_dataset",
    "target_dataset",
    "architecture",
    "strategy",
    "augmentation_name",
    "top_n",
    "seed",
    "dice",
    "iou",
    "checkpoint_path",
]

RESULT_COLUMNS_LOO = [
    "source_dataset_1",
    "source_dataset_2",
    "target_dataset",
    "architecture",
    "strategy",
    "augmentation_name",
    "top_n",
    "seed",
    "source1_best_val_dice",
    "source2_best_val_dice",
    "macro_best_val_dice",
    "target_test_dice",
    "target_test_iou",
    "checkpoint_path",
]

@dataclass(frozen=True)
class SummaryRecord:
    phase: str
    dataset: str
    model: str
    augmentation_raw: str
    augmentation_key: str
    top_n: int | None
    seed: int
    best_val_dice: float
    best_val_iou: float
    checkpoint: Path
    summary_path: Path
    log_path: Path | None
    split_manifest: str
    scan_mask_manifest: str
    image_size: int
    dropout: float
    source_datasets: tuple[str, ...]
    target_dataset: str | None

@dataclass(frozen=True)
class SelectedRun:
    source_dataset: str | None
    source_datasets: tuple[str, ...]
    target_dataset: str
    model: str
    strategy: str
    augmentation_name: str
    augmentation_key: str
    top_n: int | None
    seed: int
    record: SummaryRecord

def run_single_source_ood(
    paths: ProjectPaths,
    source_dataset: str | None = None,
    target_dataset: str | None = None,
    model: str | None = None,
    strategy: str | None = None,
    seed: int | None = None,
    device: str = "cuda",
    batch_size: int = 128,
    num_workers: int = 0,
    resume: bool = True,
    dry_run: bool = False,
    smoke: bool = False,
    max_target_samples: int | None = None,
    output_dir: str | None = None,
) -> list[Path]:
    out_dir = _output_dir(paths, output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if smoke:
        source_dataset = source_dataset or "TN3K"
        target_dataset = target_dataset or "DDTI"
        model = model or "res34_unet"
        strategy = strategy or "baseline"
        seed = 0 if seed is None else seed
        results_path = out_dir / "ood_single_source_smoke_results.csv"
    else:
        results_path = out_dir / "ood_single_source_results.csv"

    selected = select_single_source_runs(paths)
    selected = [
        run
        for run in selected
        if (source_dataset is None or run.source_dataset == source_dataset)
        and (target_dataset is None or run.target_dataset == target_dataset)
        and (model is None or run.model == model)
        and (strategy is None or run.strategy == strategy)
        and (seed is None or run.seed == seed)
    ]
    if dry_run:
        print(f"Planned single-source OOD evaluations: {len(selected)}")
        return [results_path]

    completed = _completed_keys(results_path, RESULT_COLUMNS_SINGLE[:-3]) if resume else set()
    for run in selected:
        key = (
            run.source_dataset,
            run.target_dataset,
            _model_display(run.model),
            run.strategy,
            run.augmentation_name,
            "" if run.top_n is None else str(run.top_n),
            str(run.seed),
        )
        if key in completed:
            print(f"Skipping completed OOD evaluation: {key}", flush=True)
            continue
        metrics = evaluate_checkpoint_on_target(
            paths,
            run.record,
            run.target_dataset,
            device_name=device,
            batch_size=batch_size,
            num_workers=num_workers,
            max_target_samples=max_target_samples,
            inspect_shape=smoke,
        )
        row = {
            "source_dataset": run.source_dataset or "",
            "target_dataset": run.target_dataset,
            "architecture": _model_display(run.model),
            "strategy": run.strategy,
            "augmentation_name": run.augmentation_name,
            "top_n": "" if run.top_n is None else run.top_n,
            "seed": run.seed,
            "dice": metrics["dice"],
            "iou": metrics["iou"],
            "checkpoint_path": run.record.checkpoint.as_posix(),
        }
        _append_csv(results_path, row, RESULT_COLUMNS_SINGLE)
        if smoke:
            print(
                "Smoke OOD "
                f"{run.source_dataset}->{run.target_dataset} {_model_display(run.model)} "
                f"{run.strategy} seed {run.seed}: "
                f"dice={metrics['dice']:.5f} iou={metrics['iou']:.5f} "
                f"output_shape={metrics.get('output_shape', '')}",
                flush=True,
            )

    outputs = [results_path]
    if not smoke:
        outputs.extend(analyze_ood_outputs(paths, output_dir=output_dir))
    return outputs

def prepare_leave_one_out_policies(paths: ProjectPaths, output_dir: str | None = None) -> list[Path]:
    out_dir = _output_dir(paths, output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    individual, trivial = _read_architecture_validation_summaries(paths)
    rows: list[dict[str, object]] = []
    for _, source_datasets, target_dataset in LEAVE_ONE_OUT_SPECS:
        source1, source2 = source_datasets
        for model in ARCHITECTURES:
            base_source1 = _individual_val(individual, source1, model, "none")
            base_source2 = _individual_val(individual, source2, model, "none")
            single_scores = _consensus_single_scores(individual, source1, source2, model)
            if not single_scores:
                raise RuntimeError(f"No individual augmentation scores for {source1}+{source2} {model}.")
            best_single = single_scores[0]
            consensus_order = [row["augmentation"] for row in single_scores]
            topn_scores = _consensus_topn_scores(trivial, source1, source2, model)
            if not topn_scores:
                raise RuntimeError(f"No TrivialAugment Top-N scores for {source1}+{source2} {model}.")
            best_topn = topn_scores[0]
            topn = int(best_topn["top_n"])
            trivial_policy = "trivial:identity," + ",".join(consensus_order[:topn])
            rows.extend(
                [
                    {
                        "source_dataset_1": source1,
                        "source_dataset_2": source2,
                        "target_dataset": target_dataset,
                        "architecture": _model_display(model),
                        "model": model,
                        "strategy": "baseline",
                        "selected_augmentation": "none",
                        "train_augmentation": "identity",
                        "selected_top_n": "",
                        "source1_validation_score": base_source1,
                        "source2_validation_score": base_source2,
                        "consensus_validation_score": (base_source1 + base_source2) / 2.0,
                    },
                    {
                        "source_dataset_1": source1,
                        "source_dataset_2": source2,
                        "target_dataset": target_dataset,
                        "architecture": _model_display(model),
                        "model": model,
                        "strategy": "best_single_consensus",
                        "selected_augmentation": best_single["augmentation"],
                        "train_augmentation": best_single["augmentation"],
                        "selected_top_n": "",
                        "source1_validation_score": best_single["source1_validation_score"],
                        "source2_validation_score": best_single["source2_validation_score"],
                        "consensus_validation_score": best_single["consensus_validation_score"],
                    },
                    {
                        "source_dataset_1": source1,
                        "source_dataset_2": source2,
                        "target_dataset": target_dataset,
                        "architecture": _model_display(model),
                        "model": model,
                        "strategy": "trivialaugment_consensus",
                        "selected_augmentation": trivial_policy,
                        "train_augmentation": trivial_policy,
                        "selected_top_n": topn,
                        "source1_validation_score": best_topn["source1_validation_score"],
                        "source2_validation_score": best_topn["source2_validation_score"],
                        "consensus_validation_score": best_topn["consensus_validation_score"],
                    },
                ]
            )
    public_path = out_dir / "leave_one_out_selection.csv"
    internal_path = out_dir / "leave_one_out_selection_internal.csv"
    public_columns = [
        "source_dataset_1",
        "source_dataset_2",
        "target_dataset",
        "architecture",
        "strategy",
        "selected_augmentation",
        "selected_top_n",
        "source1_validation_score",
        "source2_validation_score",
        "consensus_validation_score",
    ]
    internal_columns = [
        *public_columns,
        "model",
        "train_augmentation",
    ]
    _write_csv(public_path, rows, public_columns)
    _write_csv(internal_path, rows, internal_columns)
    return [public_path, internal_path]

def run_leave_one_out_evaluation(
    paths: ProjectPaths,
    target_dataset: str | None = None,
    model: str | None = None,
    strategy: str | None = None,
    seed: int | None = None,
    device: str = "cuda",
    batch_size: int = 128,
    num_workers: int = 0,
    resume: bool = True,
    dry_run: bool = False,
    max_target_samples: int | None = None,
    output_dir: str | None = None,
    allow_missing: bool = False,
) -> list[Path]:
    out_dir = _output_dir(paths, output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "leave_one_out_results.csv"
    selected = select_leave_one_out_runs(
        paths,
        allow_missing=allow_missing,
        target_dataset=target_dataset,
        model=model,
        strategy=strategy,
        seed=seed,
    )
    if dry_run:
        print(f"Planned leave-one-out OOD evaluations: {len(selected)}")
        return [results_path]

    completed = _completed_keys(results_path, RESULT_COLUMNS_LOO[:8]) if resume else set()
    for run in selected:
        source1, source2 = run.source_datasets
        key = (
            source1,
            source2,
            run.target_dataset,
            _model_display(run.model),
            run.strategy,
            run.augmentation_name,
            "" if run.top_n is None else str(run.top_n),
            str(run.seed),
        )
        if key in completed:
            print(f"Skipping completed leave-one-out evaluation: {key}", flush=True)
            continue
        metrics = evaluate_checkpoint_on_target(
            paths,
            run.record,
            run.target_dataset,
            device_name=device,
            batch_size=batch_size,
            num_workers=num_workers,
            max_target_samples=max_target_samples,
            inspect_shape=False,
        )
        source_validation = _best_source_validation(run.record, source1, source2)
        row = {
            "source_dataset_1": source1,
            "source_dataset_2": source2,
            "target_dataset": run.target_dataset,
            "architecture": _model_display(run.model),
            "strategy": run.strategy,
            "augmentation_name": run.augmentation_name,
            "top_n": "" if run.top_n is None else run.top_n,
            "seed": run.seed,
            "source1_best_val_dice": source_validation["source1_best_val_dice"],
            "source2_best_val_dice": source_validation["source2_best_val_dice"],
            "macro_best_val_dice": source_validation["macro_best_val_dice"],
            "target_test_dice": metrics["dice"],
            "target_test_iou": metrics["iou"],
            "checkpoint_path": run.record.checkpoint.as_posix(),
        }
        _append_csv(results_path, row, RESULT_COLUMNS_LOO)

    outputs = [results_path]
    outputs.extend(analyze_ood_outputs(paths, output_dir=output_dir))
    return outputs

def _read_architecture_validation_summaries(paths: ProjectPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = paths.project_root / "results" / "architecture_full_50"
    individual_path = base / "individual_summary.csv"
    trivial_path = base / "trivialaugment_summary.csv"
    if not individual_path.exists():
        raise FileNotFoundError(individual_path)
    if not trivial_path.exists():
        raise FileNotFoundError(trivial_path)
    individual = pd.read_csv(individual_path)
    trivial = pd.read_csv(trivial_path)
    required_individual = {"dataset", "model", "augmentation", "best_val_dice_mean"}
    required_trivial = {"dataset", "model", "top_n", "augmentation", "best_val_dice_mean"}
    if not required_individual.issubset(individual.columns):
        missing = sorted(required_individual - set(individual.columns))
        raise RuntimeError(f"Missing columns in {individual_path}: {missing}")
    if not required_trivial.issubset(trivial.columns):
        missing = sorted(required_trivial - set(trivial.columns))
        raise RuntimeError(f"Missing columns in {trivial_path}: {missing}")
    individual = individual.copy()
    trivial = trivial.copy()
    individual["augmentation"] = individual["augmentation"].map(lambda value: normalize_augmentation_name(str(value)))
    individual["best_val_dice_mean"] = pd.to_numeric(individual["best_val_dice_mean"], errors="coerce")
    trivial["top_n"] = pd.to_numeric(trivial["top_n"], errors="coerce").astype("Int64")
    trivial["best_val_dice_mean"] = pd.to_numeric(trivial["best_val_dice_mean"], errors="coerce")
    return individual, trivial

def _individual_val(frame: pd.DataFrame, dataset: str, model: str, augmentation: str) -> float:
    augmentation = normalize_augmentation_name(augmentation)
    subset = frame[
        (frame["dataset"] == dataset)
        & (frame["model"] == model)
        & (frame["augmentation"] == augmentation)
    ]
    if subset.empty:
        raise RuntimeError(f"Missing individual validation score: {dataset} {model} {augmentation}.")
    return float(subset.iloc[0]["best_val_dice_mean"])

def _consensus_single_scores(
    frame: pd.DataFrame,
    source1: str,
    source2: str,
    model: str,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    augmentations = sorted(
        set(frame[(frame["dataset"].isin([source1, source2])) & (frame["model"] == model)]["augmentation"])
        - {"none"}
    )
    for augmentation in augmentations:
        source1_score = _individual_val(frame, source1, model, augmentation)
        source2_score = _individual_val(frame, source2, model, augmentation)
        rows.append(
            {
                "augmentation": augmentation,
                "source1_validation_score": source1_score,
                "source2_validation_score": source2_score,
                "consensus_validation_score": (source1_score + source2_score) / 2.0,
            }
        )
    rows.sort(key=lambda row: (-float(row["consensus_validation_score"]), str(row["augmentation"])))
    return rows

def _consensus_topn_scores(
    frame: pd.DataFrame,
    source1: str,
    source2: str,
    model: str,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    top_ns = sorted(
        int(value)
        for value in set(frame[(frame["dataset"].isin([source1, source2])) & (frame["model"] == model)]["top_n"].dropna())
    )
    for top_n in top_ns:
        source1_score = _trivial_val(frame, source1, model, top_n)
        source2_score = _trivial_val(frame, source2, model, top_n)
        rows.append(
            {
                "top_n": top_n,
                "source1_validation_score": source1_score,
                "source2_validation_score": source2_score,
                "consensus_validation_score": (source1_score + source2_score) / 2.0,
            }
        )
    rows.sort(key=lambda row: (-float(row["consensus_validation_score"]), int(row["top_n"])))
    return rows

def _trivial_val(frame: pd.DataFrame, dataset: str, model: str, top_n: int) -> float:
    subset = frame[
        (frame["dataset"] == dataset)
        & (frame["model"] == model)
        & (frame["top_n"].astype("Int64") == top_n)
    ]
    if subset.empty:
        raise RuntimeError(f"Missing TrivialAugment validation score: {dataset} {model} Top-{top_n}.")
    return float(subset.iloc[0]["best_val_dice_mean"])

def analyze_ood_outputs(paths: ProjectPaths, output_dir: str | None = None) -> list[Path]:
    out_dir = _output_dir(paths, output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    single_path = out_dir / "ood_single_source_results.csv"
    if single_path.exists() and single_path.stat().st_size > 0:
        single = pd.read_csv(single_path)
        single_summary = _summary(single, ["source_dataset", "target_dataset", "architecture"])
        single_summary_path = out_dir / "ood_single_source_summary.csv"
        single_summary.to_csv(single_summary_path, index=False)
        outputs.append(single_summary_path)
        outputs.extend(_write_single_source_paper_tables(out_dir, single_summary))

    loo_path = out_dir / "leave_one_out_results.csv"
    if loo_path.exists() and loo_path.stat().st_size > 0:
        loo = pd.read_csv(loo_path)
        loo_summary = _leave_one_out_summary(loo)
        loo_summary_path = out_dir / "leave_one_out_summary.csv"
        loo_summary.to_csv(loo_summary_path, index=False)
        outputs.append(loo_summary_path)
        outputs.extend(_write_leave_one_out_paper_table(out_dir, loo_summary))
        overall = _leave_one_out_overall_summary(loo_summary)
        overall_path = out_dir / "leave_one_out_overall_summary.csv"
        overall.to_csv(overall_path, index=False)
        outputs.append(overall_path)

    overall = _overall_summary(out_dir)
    if not overall.empty:
        overall_path = out_dir / "ood_overall_summary.csv"
        overall.to_csv(overall_path, index=False)
        outputs.append(overall_path)
    return outputs

def select_single_source_runs(paths: ProjectPaths) -> list[SelectedRun]:
    records = collect_summary_records(paths, [PHASE_ARCH_INDIVIDUAL, PHASE_ARCH_TRIVIAL])
    by_key = _index_records(records)
    best_single = _read_selection_table(paths.project_root / "results" / "architecture_full_50" / "best_single_by_validation.csv")
    best_trivial = _read_selection_table(
        paths.project_root / "results" / "architecture_full_50" / "best_trivialaugment_by_validation.csv"
    )
    selected: list[SelectedRun] = []
    for source in DATASETS:
        for target in DATASETS:
            if source == target:
                continue
            for model in ARCHITECTURES:
                for seed in SEEDS:
                    base_record = _require_record(by_key, PHASE_ARCH_INDIVIDUAL, source, model, "none", seed)
                    selected.append(
                        SelectedRun(
                            source,
                            (source,),
                            target,
                            model,
                            "baseline",
                            "None",
                            "none",
                            None,
                            seed,
                            base_record,
                        )
                    )
                    single_row = best_single[(source, model)]
                    single_aug = normalize_augmentation_name(str(single_row["augmentation"]))
                    single_record = _require_record(by_key, PHASE_ARCH_INDIVIDUAL, source, model, single_aug, seed)
                    selected.append(
                        SelectedRun(
                            source,
                            (source,),
                            target,
                            model,
                            "best_single",
                            _display_augmentation(single_aug),
                            single_aug,
                            None,
                            seed,
                            single_record,
                        )
                    )
                    trivial_row = best_trivial[(source, model)]
                    trivial_aug = str(trivial_row["augmentation"])
                    top_n = int(float(trivial_row["top_n"]))
                    trivial_record = _require_record(
                        by_key,
                        PHASE_ARCH_TRIVIAL,
                        source,
                        model,
                        trivial_aug,
                        seed,
                        top_n=top_n,
                    )
                    selected.append(
                        SelectedRun(
                            source,
                            (source,),
                            target,
                            model,
                            "trivialaugment",
                            f"TrivialAugment Top-{top_n}",
                            trivial_aug,
                            top_n,
                            seed,
                            trivial_record,
                        )
                    )
    return selected

def select_leave_one_out_runs(
    paths: ProjectPaths,
    allow_missing: bool = False,
    target_dataset: str | None = None,
    model: str | None = None,
    strategy: str | None = None,
    seed: int | None = None,
) -> list[SelectedRun]:
    selection_path = _output_dir(paths, None) / "leave_one_out_selection_internal.csv"
    if not selection_path.exists():
        prepare_leave_one_out_policies(paths)
    records = collect_summary_records(paths, [PHASE_LOO_CONSENSUS])
    if not records:
        if allow_missing:
            print(f"WARNING: No consensus leave-one-out summaries found in runs/{PHASE_LOO_CONSENSUS}.", flush=True)
            return []
        raise RuntimeError(f"No consensus leave-one-out summaries found in runs/{PHASE_LOO_CONSENSUS}.")
    by_key = _index_records(records)
    selection_rows = list(csv.DictReader(selection_path.open("r", encoding="utf-8-sig")))
    selection_rows = [
        row
        for row in selection_rows
        if (target_dataset is None or row["target_dataset"] == target_dataset)
        and (model is None or row["model"] == model)
        and (strategy is None or row["strategy"] == strategy)
    ]
    selected: list[SelectedRun] = []
    missing: list[str] = []
    for row in selection_rows:
        source_datasets = (row["source_dataset_1"], row["source_dataset_2"])
        target = row["target_dataset"]
        model = row["model"]
        strategy = row["strategy"]
        train_augmentation = row.get("train_augmentation") or row.get("selected_augmentation") or "identity"
        top_n = _optional_int(row.get("selected_top_n"))
        dataset_label = _dataset_label(source_datasets, target)
        for run_seed in SEEDS:
            if seed is not None and run_seed != seed:
                continue
            try:
                record = _require_record(
                    by_key,
                    PHASE_LOO_CONSENSUS,
                    dataset_label,
                    model,
                    train_augmentation,
                    run_seed,
                    top_n=top_n,
                )
            except RuntimeError as exc:
                if allow_missing:
                    missing.append(str(exc))
                    continue
                raise
            selected.append(
                SelectedRun(
                    None,
                    source_datasets,
                    target,
                    model,
                    strategy,
                    _strategy_augmentation_display(strategy, train_augmentation, top_n),
                    train_augmentation,
                    top_n,
                    run_seed,
                    record,
                )
            )
    for message in missing[:20]:
        print(f"WARNING: {message}", flush=True)
    return selected

def evaluate_checkpoint_on_target(
    paths: ProjectPaths,
    record: SummaryRecord,
    target_dataset: str,
    device_name: str,
    batch_size: int,
    num_workers: int,
    max_target_samples: int | None = None,
    inspect_shape: bool = False,
) -> dict[str, object]:
    split_path = _path_from_summary(paths, record.split_manifest)
    if split_path is None or not split_path.exists():
        raise FileNotFoundError(f"Missing split manifest: {split_path}")
    rows = [
        row
        for row in csv.DictReader(split_path.open("r", encoding="utf-8-sig"))
        if row.get("dataset") == target_dataset and row.get("split") == "test"
    ]
    if max_target_samples:
        rows = rows[:max_target_samples]
    if not rows:
        raise RuntimeError(f"No target test rows found for {target_dataset} in {split_path}.")

    device = _resolve_device(device_name)
    scan_mask_map = _load_scan_mask_map(paths, record.scan_mask_manifest)
    dataset = SegmentationManifestDataset(
        rows,
        paths.project_root,
        record.image_size,
        training=False,
        scan_mask_map=scan_mask_map,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    model = build_model(
        record.model,
        in_channels=3,
        classes=1,
        encoder_weights=None,
        dropout=record.dropout,
        pretrained=False,
    ).to(device)
    if not record.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {record.checkpoint}")
    checkpoint = torch.load(record.checkpoint, map_location="cpu")
    from thyroid_augmentation.training.metrics import METRIC_PROTOCOL, require_protocol

    require_protocol(checkpoint, str(record.checkpoint))
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    output_shape = ""
    if inspect_shape:
        model.eval()
        with torch.no_grad():
            images, masks = next(iter(loader))
            logits = model(images.to(device, non_blocking=True))
            output_shape = f"logits={tuple(logits.shape)}, mask={tuple(masks.shape)}"

    start = time.perf_counter()
    metrics = _evaluate(model, loader, device)
    seconds = time.perf_counter() - start
    return {
        "metric_protocol": METRIC_PROTOCOL,
        "dice": metrics["dice"],
        "iou": metrics["iou"],
        "samples": len(rows),
        "seconds": seconds,
        "device": str(device),
        "output_shape": output_shape,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }

def _best_source_validation(record: SummaryRecord, source1: str, source2: str) -> dict[str, float]:
    if record.log_path is None or not record.log_path.exists():
        return {
            "source1_best_val_dice": math.nan,
            "source2_best_val_dice": math.nan,
            "macro_best_val_dice": record.best_val_dice,
        }
    rows = pd.read_csv(record.log_path)
    if rows.empty or "val_dice" not in rows:
        return {
            "source1_best_val_dice": math.nan,
            "source2_best_val_dice": math.nan,
            "macro_best_val_dice": record.best_val_dice,
        }
    values = pd.to_numeric(rows["val_dice"], errors="coerce")
    if values.dropna().empty:
        return {
            "source1_best_val_dice": math.nan,
            "source2_best_val_dice": math.nan,
            "macro_best_val_dice": record.best_val_dice,
        }
    best = rows.loc[values.idxmax()]
    source1_value = _safe_row_float(best, f"val_{source1}_dice")
    source2_value = _safe_row_float(best, f"val_{source2}_dice")
    macro = float(best["val_dice"])
    if math.isnan(source1_value) or math.isnan(source2_value):
        source1_value = source1_value if not math.isnan(source1_value) else macro
        source2_value = source2_value if not math.isnan(source2_value) else macro
    return {
        "source1_best_val_dice": source1_value,
        "source2_best_val_dice": source2_value,
        "macro_best_val_dice": macro,
    }

def collect_summary_records(paths: ProjectPaths, phases: Iterable[str]) -> list[SummaryRecord]:
    records: list[SummaryRecord] = []
    for phase in phases:
        phase_root = paths.project_root / "runs" / phase
        if not phase_root.exists():
            continue
        for summary_path in sorted(phase_root.glob("*/*/*/seed_*/summary.json")):
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            records.append(_summary_record(paths, phase, summary_path, summary))
    return records

def _summary_record(
    paths: ProjectPaths,
    phase: str,
    summary_path: Path,
    summary: dict[str, object],
) -> SummaryRecord:
    augmentation_raw = str(summary.get("augmentation", "none"))
    augmentation_key = (
        augmentation_raw
        if augmentation_raw.lower().startswith("trivial:")
        else normalize_augmentation_name(augmentation_raw)
    )
    top_n_value = summary.get("top_n")
    top_n = None if top_n_value in ("", None) else int(top_n_value)
    log_path = _path_from_summary(paths, summary.get("log"))
    best_val_dice, best_val_iou = _best_validation_from_log(log_path)
    checkpoint = _path_from_summary(paths, summary.get("checkpoint"))
    if checkpoint is None:
        checkpoint = paths.project_root / "__missing_checkpoint__"
    source_datasets_raw = summary.get("source_datasets") or [summary.get("dataset", "")]
    source_datasets = tuple(str(item) for item in source_datasets_raw if str(item))
    return SummaryRecord(
        phase=phase,
        dataset=str(summary.get("dataset", "")),
        model=str(summary.get("model", "")),
        augmentation_raw=augmentation_raw,
        augmentation_key=augmentation_key,
        top_n=top_n,
        seed=int(summary.get("seed", 0)),
        best_val_dice=best_val_dice if not math.isnan(best_val_dice) else float(summary.get("best_dice", math.nan)),
        best_val_iou=best_val_iou,
        checkpoint=checkpoint,
        summary_path=summary_path,
        log_path=log_path,
        split_manifest=str(summary.get("split_manifest", "")),
        scan_mask_manifest=str(summary.get("scan_mask_manifest", "")),
        image_size=int(summary.get("image_size", 224) or 224),
        dropout=float(summary.get("dropout", 0.0) or 0.0),
        source_datasets=source_datasets,
        target_dataset=str(summary.get("target_dataset") or "") or None,
    )

def _best_validation_from_log(log_path: Path | None) -> tuple[float, float]:
    if log_path is None or not log_path.exists():
        return math.nan, math.nan
    try:
        rows = pd.read_csv(log_path)
    except Exception:
        return math.nan, math.nan
    if rows.empty or "val_dice" not in rows:
        return math.nan, math.nan
    values = pd.to_numeric(rows["val_dice"], errors="coerce")
    if values.dropna().empty:
        return math.nan, math.nan
    best = rows.loc[values.idxmax()]
    return float(best["val_dice"]), float(best.get("val_iou", math.nan))

def _index_records(records: list[SummaryRecord]) -> dict[tuple[str, str, str, str, int, int | None], SummaryRecord]:
    out: dict[tuple[str, str, str, str, int, int | None], SummaryRecord] = {}
    for record in records:
        out[(record.phase, record.dataset, record.model, record.augmentation_key, record.seed, record.top_n)] = record
    return out

def _require_record(
    index: dict[tuple[str, str, str, str, int, int | None], SummaryRecord],
    phase: str,
    dataset: str,
    model: str,
    augmentation: str,
    seed: int,
    top_n: int | None = None,
) -> SummaryRecord:
    key = (
        phase,
        dataset,
        model,
        augmentation if augmentation.lower().startswith("trivial:") else normalize_augmentation_name(augmentation),
        seed,
        top_n,
    )
    record = index.get(key)
    if record is None:
        raise RuntimeError(f"Missing summary for {key}.")
    if not record.checkpoint.exists():
        raise RuntimeError(f"Missing checkpoint for {key}: {record.checkpoint}")
    return record

def _records_frame(records: list[SummaryRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "phase": record.phase,
                "dataset": record.dataset,
                "model": record.model,
                "augmentation_raw": record.augmentation_raw,
                "augmentation_key": record.augmentation_key,
                "top_n": record.top_n,
                "seed": record.seed,
                "best_val_dice": record.best_val_dice,
                "best_val_iou": record.best_val_iou,
                "checkpoint": record.checkpoint.as_posix(),
                "source_datasets": "+".join(record.source_datasets),
                "target_dataset": record.target_dataset or "",
            }
            for record in records
        ]
    )

def _summary(frame: pd.DataFrame, combo_cols: list[str]) -> pd.DataFrame:
    numeric = frame.copy()
    numeric["dice"] = pd.to_numeric(numeric["dice"], errors="coerce")
    numeric["iou"] = pd.to_numeric(numeric["iou"], errors="coerce")
    grouped = numeric.groupby([*combo_cols, "strategy"], dropna=False)
    summary = grouped.agg(
        mean_dice=("dice", "mean"),
        std_dice=("dice", _sample_sd),
        mean_iou=("iou", "mean"),
        std_iou=("iou", _sample_sd),
    ).reset_index()
    summary["delta_dice_vs_baseline"] = 0.0
    for key, group in summary.groupby(combo_cols, dropna=False):
        baseline = group[group["strategy"] == "baseline"]
        if baseline.empty:
            continue
        baseline_dice = float(baseline.iloc[0]["mean_dice"])
        mask = pd.Series(True, index=summary.index)
        if not isinstance(key, tuple):
            key = (key,)
        for col, value in zip(combo_cols, key, strict=False):
            mask &= summary[col] == value
        summary.loc[mask, "delta_dice_vs_baseline"] = summary.loc[mask, "mean_dice"] - baseline_dice
    return summary

def _leave_one_out_summary(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.copy()
    numeric["target_test_dice"] = pd.to_numeric(numeric["target_test_dice"], errors="coerce")
    numeric["target_test_iou"] = pd.to_numeric(numeric["target_test_iou"], errors="coerce")
    numeric["source_datasets"] = numeric["source_dataset_1"].astype(str) + "+" + numeric["source_dataset_2"].astype(str)
    combo_cols = ["source_datasets", "target_dataset", "architecture"]
    grouped = numeric.groupby([*combo_cols, "strategy"], dropna=False)
    summary = grouped.agg(
        mean_target_dice=("target_test_dice", "mean"),
        std_target_dice=("target_test_dice", _sample_sd),
        mean_target_iou=("target_test_iou", "mean"),
        std_target_iou=("target_test_iou", _sample_sd),
    ).reset_index()
    summary["delta_dice_vs_baseline"] = 0.0
    for key, group in summary.groupby(combo_cols, dropna=False):
        baseline = group[group["strategy"] == "baseline"]
        if baseline.empty:
            continue
        baseline_dice = float(baseline.iloc[0]["mean_target_dice"])
        mask = pd.Series(True, index=summary.index)
        if not isinstance(key, tuple):
            key = (key,)
        for col, value in zip(combo_cols, key, strict=False):
            mask &= summary[col] == value
        summary.loc[mask, "delta_dice_vs_baseline"] = summary.loc[mask, "mean_target_dice"] - baseline_dice
    columns = [
        "source_datasets",
        "target_dataset",
        "architecture",
        "strategy",
        "mean_target_dice",
        "std_target_dice",
        "mean_target_iou",
        "std_target_iou",
        "delta_dice_vs_baseline",
    ]
    return summary[columns]

def _leave_one_out_overall_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for strategy in LOO_STRATEGIES:
        subset = summary[summary["strategy"] == strategy].copy()
        subset["mean_target_dice"] = pd.to_numeric(subset["mean_target_dice"], errors="coerce")
        subset["delta_dice_vs_baseline"] = pd.to_numeric(subset["delta_dice_vs_baseline"], errors="coerce")
        rows.append(
            {
                "strategy": strategy,
                "macro_average_dice": float(subset["mean_target_dice"].mean()) if not subset.empty else "",
                "mean_delta_dice_vs_baseline": float(subset["delta_dice_vs_baseline"].mean()) if not subset.empty else "",
                "median_delta_dice_vs_baseline": float(subset["delta_dice_vs_baseline"].median()) if not subset.empty else "",
                "cases_better_than_baseline": int((subset["delta_dice_vs_baseline"] > 0).sum()) if not subset.empty else "",
                "cases_worse_than_baseline": int((subset["delta_dice_vs_baseline"] < 0).sum()) if not subset.empty else "",
            }
        )
    return pd.DataFrame(rows)

def _overall_summary(out_dir: Path) -> pd.DataFrame:
    if not (out_dir / "ood_single_source_summary.csv").exists():
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for strategy in STRATEGIES:
        row: dict[str, object] = {"strategy": strategy}
        single_summary = _read_optional_frame(out_dir / "ood_single_source_summary.csv")
        _add_overall_metrics(row, single_summary, strategy, "single_source")
        rows.append(row)
    return pd.DataFrame(rows)

def _add_overall_metrics(row: dict[str, object], frame: pd.DataFrame, strategy: str, prefix: str) -> None:
    if frame.empty:
        row[f"{prefix}_mean_dice"] = ""
        row[f"{prefix}_cases_beats_baseline"] = ""
        row[f"{prefix}_mean_delta_dice_vs_baseline"] = ""
        row[f"{prefix}_median_delta_dice_vs_baseline"] = ""
        return
    subset = frame[frame["strategy"] == strategy].copy()
    if subset.empty:
        row[f"{prefix}_mean_dice"] = ""
        row[f"{prefix}_cases_beats_baseline"] = ""
        row[f"{prefix}_mean_delta_dice_vs_baseline"] = ""
        row[f"{prefix}_median_delta_dice_vs_baseline"] = ""
        return
    subset["mean_dice"] = pd.to_numeric(subset["mean_dice"], errors="coerce")
    subset["delta_dice_vs_baseline"] = pd.to_numeric(subset["delta_dice_vs_baseline"], errors="coerce")
    row[f"{prefix}_mean_dice"] = float(subset["mean_dice"].mean())
    row[f"{prefix}_cases_beats_baseline"] = int((subset["delta_dice_vs_baseline"] > 0).sum())
    row[f"{prefix}_mean_delta_dice_vs_baseline"] = float(subset["delta_dice_vs_baseline"].mean())
    row[f"{prefix}_median_delta_dice_vs_baseline"] = float(subset["delta_dice_vs_baseline"].median())

def _write_single_source_paper_tables(out_dir: Path, summary: pd.DataFrame) -> list[Path]:
    outputs: list[Path] = []
    for strategy in ["best_single", "trivialaugment"]:
        rows: list[dict[str, object]] = []
        for (source, target), group in summary.groupby(["source_dataset", "target_dataset"], dropna=False):
            row: dict[str, object] = {"direction": f"{source}->{target}"}
            for arch in MODEL_DISPLAY.values():
                base = group[(group["architecture"] == arch) & (group["strategy"] == "baseline")]
                aug = group[(group["architecture"] == arch) & (group["strategy"] == strategy)]
                row[f"{arch} baseline Dice"] = _mean_sd_from_row(base)
                row[f"{arch} {strategy} Dice"] = _mean_sd_from_row(aug)
                row[f"{arch} delta"] = _delta_from_row(aug)
            rows.append(row)
        path = out_dir / f"paper_table_ood_single_source_{strategy}.csv"
        _write_csv(path, rows, list(rows[0].keys()) if rows else ["direction"])
        outputs.append(path)
    return outputs

def _write_leave_one_out_paper_table(out_dir: Path, summary: pd.DataFrame) -> list[Path]:
    rows: list[dict[str, object]] = []
    for (sources, target), group in summary.groupby(["source_datasets", "target_dataset"], dropna=False):
        for arch in MODEL_DISPLAY.values():
            arch_group = group[group["architecture"] == arch]
            baseline = arch_group[arch_group["strategy"] == "baseline"]
            single = arch_group[arch_group["strategy"] == "best_single_consensus"]
            trivial = arch_group[arch_group["strategy"] == "trivialaugment_consensus"]
            rows.append(
                {
                    "Train domains": str(sources).replace("+", " + "),
                    "Unseen domain": target,
                    "Architecture": arch,
                    "Baseline": _mean_sd_from_row(baseline),
                    "Best Single": _mean_sd_from_row(single),
                    "TrivialAugment": _mean_sd_from_row(trivial),
                    "Delta Best Single vs Baseline": _delta_from_row(single),
                    "Delta TrivialAugment vs Baseline": _delta_from_row(trivial),
                }
            )
    path = out_dir / "leave_one_out_manuscript_table.csv"
    _write_csv(path, rows, list(rows[0].keys()) if rows else ["Train domains"])
    return [path]

def _mean_sd_from_row(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    row = frame.iloc[0]
    mean_col = "mean_target_dice" if "mean_target_dice" in row.index else "mean_dice"
    sd_col = "std_target_dice" if "std_target_dice" in row.index else "std_dice"
    return f"{float(row[mean_col]):.4f} +/- {float(row[sd_col]):.4f}"

def _delta_from_row(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    return f"{float(frame.iloc[0]['delta_dice_vs_baseline']):+.4f}"

def _sample_sd(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0

def _read_selection_table(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return {
        (row["dataset"], row["model"]): row
        for row in csv.DictReader(path.open("r", encoding="utf-8-sig"))
    }

def _read_optional_frame(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)

def _path_from_summary(paths: ProjectPaths, value: object) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = paths.project_root / path
    return path

def _output_dir(paths: ProjectPaths, output_dir: str | None) -> Path:
    if not output_dir:
        return paths.project_root / "results" / "ood"
    path = Path(output_dir)
    if not path.is_absolute():
        path = paths.project_root / path
    return path

def _model_display(model: str) -> str:
    return MODEL_DISPLAY.get(model, model)

def _display_augmentation(name: str) -> str:
    return DISPLAY_NAMES.get(normalize_augmentation_name(name), name)

def _strategy_augmentation_display(strategy: str, augmentation: str, top_n: int | None) -> str:
    if strategy == "baseline":
        return "None"
    if strategy == "trivialaugment_consensus":
        return f"TrivialAugment Top-{top_n}" if top_n is not None else "TrivialAugment"
    return _display_augmentation(augmentation)

def _dataset_label(source_datasets: tuple[str, ...], target_dataset: str) -> str:
    for label, sources, target in LEAVE_ONE_OUT_SPECS:
        if tuple(sources) == tuple(source_datasets) and target == target_dataset:
            return label
    return f"{'_'.join(source_datasets)}_to_{target_dataset}"

def _optional_int(value: object) -> int | None:
    if value in (None, "", "nan"):
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return int(float(str(value)))

def _safe_row_float(row: pd.Series, column: str) -> float:
    if column not in row:
        return math.nan
    try:
        value = float(row[column])
    except (TypeError, ValueError):
        return math.nan
    return value

def _completed_keys(path: Path, key_columns: list[str]) -> set[tuple[str, ...]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    rows = csv.DictReader(path.open("r", encoding="utf-8-sig"))
    return {tuple(str(row.get(col, "")) for col in key_columns) for row in rows}

def _append_csv(path: Path, row: dict[str, object], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
