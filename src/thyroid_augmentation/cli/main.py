from __future__ import annotations

import argparse
import sys
from pathlib import Path

from thyroid_augmentation.datasets.inventory import audit_datasets
from thyroid_augmentation.datasets.splits import create_arxiv_stanford_splits, create_splits
from thyroid_augmentation.models.base import count_parameters
from thyroid_augmentation.models.registry import available_models, build_model, model_specs
from thyroid_augmentation.models.testing import run_model_smoke_tests
from thyroid_augmentation.paths import ProjectPaths
from thyroid_augmentation.preprocessing.panel_maps import create_ddti_panel_maps
from thyroid_augmentation.preprocessing.qc import create_qc_galleries
from thyroid_augmentation.preprocessing.scan_masks import prepare_arxiv_stanford_scan_masks, prepare_scan_masks
from thyroid_augmentation.reporting.benchmark import DEFAULT_PHASES, evaluate_and_report_benchmark
from thyroid_augmentation.reporting.plan import write_runtime_plan
from thyroid_augmentation.reporting.ranking import rank_individual_augmentations
from thyroid_augmentation.training.runner import TrainConfig, run_training
from thyroid_augmentation.ood import (
    analyze_ood_outputs,
    prepare_leave_one_out_policies,
    run_leave_one_out_evaluation,
    run_single_source_ood,
)
from thyroid_augmentation.upstream import inspect_upstream

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = ProjectPaths.from_args(args.project_root, args.data_root)
    paths.ensure_output_dirs()
    try:
        return int(args.func(args, paths) or 0)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m thyroid_augmentation")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--data-root", default=None)
    sub = parser.add_subparsers(required=True)

    _add_common(sub.add_parser("inspect-upstream")).set_defaults(func=_cmd_inspect_upstream)
    _add_common(sub.add_parser("audit")).set_defaults(func=_cmd_audit)
    _add_common(sub.add_parser("create-splits")).set_defaults(func=_cmd_create_splits)
    _add_common(sub.add_parser("create-arxiv-stanford-splits")).set_defaults(func=_cmd_create_arxiv_stanford_splits)
    _add_common(sub.add_parser("prepare-scan-masks")).set_defaults(func=_cmd_prepare_scan_masks)
    _add_common(sub.add_parser("prepare-arxiv-stanford-scan-masks")).set_defaults(
        func=_cmd_prepare_arxiv_stanford_scan_masks
    )
    _add_common(sub.add_parser("review-ddti-split-screen")).set_defaults(func=_cmd_panel_maps)
    _add_common(sub.add_parser("qc")).set_defaults(func=_cmd_qc)
    _add_common(sub.add_parser("test-models")).set_defaults(func=_cmd_test_models)
    _add_common(sub.add_parser("plan")).set_defaults(func=_cmd_plan)
    _add_common(sub.add_parser("list-models")).set_defaults(func=_cmd_list_models)
    _add_common(sub.add_parser("train")).set_defaults(func=_cmd_train)
    _add_common(sub.add_parser("rank-augmentations")).set_defaults(func=_cmd_rank_augmentations)
    _add_common(sub.add_parser("run-trivialaugment")).set_defaults(func=_cmd_not_implemented_plan)
    _add_common(sub.add_parser("evaluate")).set_defaults(func=_cmd_evaluate)
    _add_common(sub.add_parser("ood-single-source")).set_defaults(func=_cmd_ood_single_source)
    _add_common(sub.add_parser("ood-prepare-leave-one-out-policies")).set_defaults(
        func=_cmd_ood_prepare_leave_one_out_policies
    )
    _add_common(sub.add_parser("ood-leave-one-out-evaluate")).set_defaults(func=_cmd_ood_leave_one_out_evaluate)
    _add_common(sub.add_parser("ood-analyze")).set_defaults(func=_cmd_ood_analyze)
    _add_common(sub.add_parser("report")).set_defaults(func=_cmd_plan)
    return parser

def _add_common(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--source-dataset", default=None)
    parser.add_argument("--target-dataset", default=None)
    parser.add_argument("--source-datasets", default=None)
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-target-samples", type=int, default=None)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--model", default="unet_efficientnet_b0")
    parser.add_argument("--phase", default="preparation")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--augmentation", default="none")
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--effective-batch-size", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config", default=None)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--scheduler-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=2.77414e-4)
    parser.add_argument("--weight-decay", type=float, default=3.43856e-4)
    parser.add_argument("--dropout", type=float, default=0.454402)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--split-manifest", default="manifests/splits.csv")
    parser.add_argument("--scan-mask-manifest", default="manifests/scan_masks.csv")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--skip-test-eval", action="store_true")
    return parser

def _cmd_inspect_upstream(args: argparse.Namespace, paths: ProjectPaths) -> int:
    result = inspect_upstream(paths)
    print(f"Wrote {result.csv_path}")
    print(f"Wrote {result.report_path}")
    return 0

def _cmd_audit(args: argparse.Namespace, paths: ProjectPaths) -> int:
    result = audit_datasets(paths, sample_limit=args.sample_limit)
    print(f"Wrote {result.folder_report}")
    print(f"Wrote {result.inventory_report}")
    print(f"Wrote {result.manifest_csv}")
    return 0

def _cmd_create_splits(args: argparse.Namespace, paths: ProjectPaths) -> int:
    out = create_splits(paths, seed=args.seed if args.seed is not None else 0)
    print(f"Wrote {out}")
    return 0

def _cmd_create_arxiv_stanford_splits(args: argparse.Namespace, paths: ProjectPaths) -> int:
    out = create_arxiv_stanford_splits(paths)
    print(f"Wrote {out}")
    return 0

def _cmd_prepare_scan_masks(args: argparse.Namespace, paths: ProjectPaths) -> int:
    out = prepare_scan_masks(paths, max_items=args.sample_limit)
    print(f"Wrote {out}")
    return 0

def _cmd_prepare_arxiv_stanford_scan_masks(args: argparse.Namespace, paths: ProjectPaths) -> int:
    out = prepare_arxiv_stanford_scan_masks(paths, max_items=args.sample_limit)
    print(f"Wrote {out}")
    return 0

def _cmd_panel_maps(args: argparse.Namespace, paths: ProjectPaths) -> int:
    out = create_ddti_panel_maps(paths)
    print(f"Wrote {out}")
    return 0

def _cmd_qc(args: argparse.Namespace, paths: ProjectPaths) -> int:
    outputs = create_qc_galleries(paths)
    for output in outputs:
        print(f"Wrote {output}")
    return 0

def _cmd_test_models(args: argparse.Namespace, paths: ProjectPaths) -> int:
    out = run_model_smoke_tests(paths, image_size=args.image_size, device=args.device)
    print(f"Wrote {out}")
    return 0

def _cmd_list_models(args: argparse.Namespace, paths: ProjectPaths) -> int:
    specs = model_specs()
    for name in available_models():
        spec = specs[name]
        try:
            parameters = count_parameters(build_model(name, encoder_weights=None))
        except Exception:
            parameters = "construction_failed"
        print(f"{name}\t{spec.architecture}\t{spec.encoder}\t{parameters}")
    return 0

def _cmd_plan(args: argparse.Namespace, paths: ProjectPaths) -> int:
    out = write_runtime_plan(paths)
    print(f"Wrote {out}")
    return 0

def _cmd_rank_augmentations(args: argparse.Namespace, paths: ProjectPaths) -> int:
    outputs = rank_individual_augmentations(paths, model=args.model, phase=args.phase)
    print(f"Wrote {outputs.ranking_csv}")
    print(f"Wrote {outputs.policies_csv}")
    print(f"Wrote {outputs.report_md}")
    return 0

def _cmd_evaluate(args: argparse.Namespace, paths: ProjectPaths) -> int:
    phases = DEFAULT_PHASES if args.phase in {"preparation", "all"} else [
        item.strip() for item in args.phase.split(",") if item.strip()
    ]
    outputs = evaluate_and_report_benchmark(
        paths,
        phases=phases,
        model=args.model,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resume=args.resume,
        dry_run=args.dry_run,
        evaluate_test=not args.skip_test_eval,
    )
    print(f"Wrote {outputs.combined_csv}")
    print(f"Wrote {outputs.outputs_json}")
    for path in outputs.reports:
        print(f"Wrote {path}")
    for path in outputs.tables:
        print(f"Wrote {path}")
    for path in outputs.figures:
        print(f"Wrote {path}")
    return 0

def _cmd_ood_single_source(args: argparse.Namespace, paths: ProjectPaths) -> int:
    outputs = run_single_source_ood(
        paths,
        source_dataset=args.source_dataset,
        target_dataset=args.target_dataset,
        model=_ood_model_filter(args.model),
        strategy=args.strategy,
        seed=args.seed if args.seed is not None else None,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resume=args.resume,
        dry_run=args.dry_run,
        smoke=args.smoke,
        max_target_samples=args.max_target_samples,
        output_dir=args.output_dir,
    )
    for path in outputs:
        print(f"Wrote {path}")
    return 0

def _cmd_ood_prepare_leave_one_out_policies(args: argparse.Namespace, paths: ProjectPaths) -> int:
    outputs = prepare_leave_one_out_policies(paths, output_dir=args.output_dir)
    for path in outputs:
        print(f"Wrote {path}")
    return 0

def _cmd_ood_leave_one_out_evaluate(args: argparse.Namespace, paths: ProjectPaths) -> int:
    outputs = run_leave_one_out_evaluation(
        paths,
        target_dataset=args.target_dataset,
        model=_ood_model_filter(args.model),
        strategy=args.strategy,
        seed=args.seed if args.seed is not None else None,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resume=args.resume,
        dry_run=args.dry_run,
        max_target_samples=args.max_target_samples,
        output_dir=args.output_dir,
        allow_missing=args.allow_missing,
    )
    for path in outputs:
        print(f"Wrote {path}")
    return 0

def _cmd_ood_analyze(args: argparse.Namespace, paths: ProjectPaths) -> int:
    outputs = analyze_ood_outputs(paths, output_dir=args.output_dir)
    for path in outputs:
        print(f"Wrote {path}")
    return 0

def _cmd_train(args: argparse.Namespace, paths: ProjectPaths) -> int:
    approval = paths.project_root / "APPROVED_TO_TRAIN.txt"
    train_seed = args.seed if args.seed is not None else 0
    if args.dry_run or not approval.exists():
        print("Training not launched.")
        print(f"Dry run: {args.dry_run}")
        print(f"Approval file exists: {approval.exists()} ({approval})")
        print(
            "Planned command: python -m thyroid_augmentation train "
            f"--dataset {args.dataset} --model {args.model} --augmentation {args.augmentation} "
            f"--seed {train_seed} --device {args.device} --batch-size {args.batch_size}"
        )
        write_runtime_plan(paths)
        return 0 if args.dry_run else 2
    if args.dataset is None:
        raise ValueError("--dataset is required for training")
    source_datasets = _parse_source_datasets(args.source_datasets)
    effective_batch_size = args.effective_batch_size or args.batch_size
    summary = run_training(
        paths,
        TrainConfig(
            dataset=args.dataset,
            model=args.model,
            augmentation=args.augmentation,
            seed=train_seed,
            epochs=args.epochs,
            scheduler_epochs=args.scheduler_epochs,
            batch_size=args.batch_size,
            effective_batch_size=effective_batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            pretrained=args.pretrained,
            top_n=args.top_n,
            split_manifest=args.split_manifest,
            scan_mask_manifest=args.scan_mask_manifest,
            device=args.device,
            resume=args.resume,
            phase=args.phase,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            source_datasets=source_datasets,
            target_dataset=args.target_dataset,
        ),
    )
    print(f"Training finished. Summary: {summary}")
    return 0

def _cmd_not_implemented_plan(args: argparse.Namespace, paths: ProjectPaths) -> int:
    print("This command is planned but not run in the first safe preparation phase.")
    write_runtime_plan(paths)
    return 0

def _parse_source_datasets(value: str | None) -> tuple[str, ...] | None:
    if not value:
        return None
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    return items or None

def _ood_model_filter(model: str | None) -> str | None:
    if model in (None, "", "all", "unet_efficientnet_b0"):
        return None
    return model
