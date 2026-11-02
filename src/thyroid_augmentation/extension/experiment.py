from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from thyroid_augmentation.extension.common import PREPARED, SCANS, SPLITS, digest, protocol, read_csv, write_csv, write_json

PHASE = "extension_single"
AUGMENTATIONS = ["none", "bilateral_filter", "brightness", "contrast", "depth_attenuation", "gamma",
                 "gaussian_noise", "gaussian_shadow", "haze_artifact", "horizontal_flip", "vertical_flip",
                 "random_crop", "rotate", "translate", "zoom"]

def jobs(root: Path) -> list[dict]:
    p = protocol(root)
    common = {key: p[key] for key in ("epochs", "scheduler_epochs", "batch_size", "effective_batch_size", "image_size",
                                    "learning_rate", "weight_decay", "dropout", "pretrained")}
    return [{**common, "dataset": dataset, "model": model, "augmentation": aug, "seed": seed,
             "num_workers": 0, "top_n": None, "split_manifest": SPLITS, "scan_mask_manifest": SCANS,
             "device": "cuda", "resume": True, "phase": PHASE}
            for dataset in ("PKTN", "TNUI-2021", "ThyroidXL") for model in p["models"]
            for aug in AUGMENTATIONS for seed in p["seeds"]]

def run_directory(root: Path, config: dict, directory="runs") -> Path:
    return root / directory / config["phase"] / config["dataset"] / config["model"] / config["augmentation"] / f"seed_{config['seed']}"

def source_hashes(root: Path) -> dict:
    files = [root / SPLITS, root / SCANS, root / PREPARED,
             root / "manifests/extension/preparation_issues.json",
             root / "manifests/extension/preparation_audit.csv",
             root / "configs/experiments/extension_plan.json",
             root / "configs/experiments/extension_export_contract.json",
             root / "scripts/01_prepare_extension_datasets.py"]
    for folder in (root / "src/thyroid_augmentation/extension", root / "src/thyroid_augmentation/training",
                   root / "src/thyroid_augmentation/augmentations"):
        files.extend(sorted(folder.glob("*.py")))
    files.extend(sorted((root / "configs/published/models").glob("*.yaml")))
    files.extend(sorted((root / "external/ultrasound-augmentation/usaugment-experiments/src/usaugment").rglob("*.py")))
    return {str(f.relative_to(root)): digest(f) for f in files}

def verify_prepared(root: Path, check_pixels=True) -> dict:
    import numpy as np
    from PIL import Image

    p = protocol(root)
    issues = json.loads((root / "manifests/extension/preparation_issues.json").read_text())
    if issues["status"] != "passed" or issues["issues"]:
        raise ValueError("Unresolved image/mask preparation issues; training is blocked.")
    info = json.loads((root / PREPARED).read_text())
    if digest(root / SPLITS) != info["manifest_sha256"] or digest(root / SCANS) != info["scan_manifest_sha256"]:
        raise ValueError("Prepared manifests changed.")
    audit_path = root / "manifests/extension/preparation_audit.csv"
    if digest(audit_path) != info["audit_sha256"]:
        raise ValueError("Preparation audit changed.")
    raw = read_csv(root / "manifests/extension/splits_raw.csv")
    rows, scans = read_csv(root / SPLITS), read_csv(root / SCANS)
    if len(rows) != 14021 or len(scans) != len(rows) or len({(r['dataset'], r['sample_id']) for r in rows}) != len(rows):
        raise ValueError("Incomplete prepared dataset.")
    keys = ("dataset", "sample_id", "group_id", "group_unit", "split", "case_id", "clip_id", "image_path")
    if [{k: x[k] for k in keys} for x in raw] != [{k: x[k] for k in keys} for x in rows]:
        raise ValueError("Data preparation changed the frozen split or IDs.")
    scan_map = {(r["dataset"], r["sample_id"]): r for r in scans}
    audit_map = {(r["dataset"], r["sample_id"]): r for r in read_csv(audit_path)}
    for row in rows:
        image, mask = root / row["image_path"], root / row["mask_path"]
        scan = root / scan_map[row["dataset"], row["sample_id"]]["scan_mask_path"]
        if not all(x.is_file() for x in (image, mask, scan)):
            raise FileNotFoundError(row["sample_id"])
        if check_pixels:
            audit = audit_map[row["dataset"], row["sample_id"]]
            for path, col in ((image, "source_image_sha256"), (mask, "mask_sha256"), (scan, "scan_sha256")):
                if digest(path) != audit[col]:
                    raise ValueError(f"Prepared file changed: {path}")
            with Image.open(image) as im, Image.open(mask) as ma, Image.open(scan) as sc:
                if not im.size == ma.size == sc.size:
                    raise ValueError(f"Misaligned prepared data: {row['sample_id']}")
                if not set(np.unique(ma)).issubset({0, 255}) or not set(np.unique(sc)).issubset({0, 255}):
                    raise ValueError("Nonbinary prepared masks.")
    all_jobs = jobs(root)
    if len(all_jobs) != 405 or set(p["seeds"]) != {0, 1, 2}:
        raise ValueError("Wrong first-stage grid.")
    longest = max(len(str(run_directory(root, j, d) / filename)) for j in all_jobs
                  for d in ("runs", "checkpoints") for filename in ("checkpoint_retention.json.tmp", "best.pt", "last.pt"))
    if longest >= 250:
        raise ValueError(f"Windows path too long: {longest}")
    return {"samples": len(rows), "jobs": len(all_jobs), "longest_planned_path": longest,
            "split_counts": dict(Counter(f"{r['dataset']}/{r['split']}" for r in rows))}

def check_runtime(root: Path, rows: list[dict], scan_manifest: str) -> dict:
    import importlib.metadata
    import random
    import numpy as np
    import torch
    from thyroid_augmentation.extension.models import build_model
    from thyroid_augmentation.extension.evaluate import infer_rows, validate_export
    from thyroid_augmentation.paths import ProjectPaths
    from thyroid_augmentation.training.runner import DiceLoss, SegmentationManifestDataset, _load_scan_mask_map

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU; CPU fallback is forbidden.")
    free = shutil.disk_usage(root).free
    if free < 160 * 1024**3:
        raise RuntimeError("Less than 160 GiB free before first-stage preflight.")
    paths = ProjectPaths(root, root / "data")
    scans = _load_scan_mask_map(paths, scan_manifest)
    augmentation_checks = []
    samples = {ds: [r for r in rows if r["dataset"] == ds and r["split"] == "train"][:2]
               for ds in ("PKTN", "TNUI-2021", "ThyroidXL")}
    for ds, selected in samples.items():
        for aug in AUGMENTATIONS:
            for seed in (0, 1, 2):
                random.seed(seed)
                np.random.seed(seed)
                data = SegmentationManifestDataset(selected, root, 224, augmentation=aug, scan_mask_map=scans, training=True)
                for i in range(len(data)):
                    x, y = data[i]
                    if x.shape != (3, 224, 224) or y.shape != (1, 224, 224) or not torch.isfinite(x).all() or not set(y.unique().tolist()).issubset({0, 1}):
                        raise ValueError(f"Failed transform smoke test {ds}/{aug}/{seed}")
            augmentation_checks.append(f"{ds}/{aug}")
    smoke = []
    for name in protocol(root)["models"]:
        torch.manual_seed(0)
        model = build_model(root, name, pretrained=True).cuda()
        model.train()
        for ds, selected in samples.items():
            data = SegmentationManifestDataset(selected, root, 224, augmentation="none", scan_mask_map=scans, training=True)
            images, masks = zip(*(data[i] for i in range(2)))
            x = torch.stack(images).repeat(64, 1, 1, 1).cuda()
            y = torch.stack(masks).repeat(64, 1, 1, 1).cuda()
            model.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            with torch.amp.autocast("cuda"):
                pred = model(x)
                loss = DiceLoss(sigmoid=True, include_background=False)(pred, y)
            loss.backward()
            if pred.shape != y.shape or not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise ValueError(f"Invalid model smoke test {name}/{ds}")
            smoke.append({"model": name, "dataset": ds, "batch_size": 128, "loss": float(loss.detach()),
                          "peak_memory_MiB": torch.cuda.max_memory_allocated() / 1024**2})
            print(f"Smoke OK: {name}/{ds}, batch 128", flush=True)
            del x, y, pred, loss
            config = {"phase": "diagnostic_only", "dataset": ds, "model": name, "augmentation": "none", "seed": 0, "image_size": 224}
            for split in ("val", "test"):
                sources = [r for r in rows if r["dataset"] == ds and r["split"] == split][:2]
                output = infer_rows(root, config, model, sources, split, 128)
                path = root / "results/extension/diagnostics" / ds / name / f"{split}_untrained.csv"
                write_csv(path, output)
                validate_export(read_csv(path), sources, config, split)
            smoke[-1]["per_image_inference_and_csv_roundtrip"] = "val and test; two diagnostic images each"
            evaluation = SegmentationManifestDataset(sources, root, 224, training=False)
            x = torch.stack([evaluation[i][0] for i in range(2)]).repeat(64, 1, 1, 1).cuda()
            model.zero_grad(set_to_none=True)
            with torch.inference_mode():
                pred = model(x)
                if pred.shape != (128, 1, 224, 224) or not torch.isfinite(pred).all():
                    raise ValueError(f"Invalid FP32 inference smoke test {name}/{ds}")
            smoke[-1]["float32_inference_batch_128"] = "passed"
            del x, pred
            model.train()
        del model
        torch.cuda.empty_cache()
    return dict(gpu=torch.cuda.get_device_name(), free_GiB=free / 1024**3,
                augmentations_checked=augmentation_checks, model_smoke_tests=smoke,
                versions={p: importlib.metadata.version(p) for p in ("torch", "numpy", "albumentations", "segmentation-models-pytorch", "monai", "torchmetrics")})

def diagnostic_smoke(root: Path) -> dict:
    report = check_runtime(root, read_csv(root / "manifests/extension/splits_diagnostic.csv"),
                           "manifests/extension/scans_diagnostic.csv")
    report.update(status="diagnostic_only", training_started=False)
    write_json(root / "results/extension/runtime_smoke.json", report)
    return report

def preflight(root: Path) -> dict:
    report = verify_prepared(root)
    report.update(check_runtime(root, read_csv(root / SPLITS), SCANS))
    report.update(status="ready_for_single_stage", source_hashes=source_hashes(root))
    write_json(root / "results/extension/preflight.json", report)
    return report

def require_preflight(root: Path):
    import importlib.metadata

    path = root / "results/extension/preflight.json"
    info = json.loads(path.read_text())
    if info.get("status") != "ready_for_single_stage" or info["source_hashes"] != source_hashes(root):
        raise ValueError("Preflight is missing or stale; rerun preflight before training.")
    if any(importlib.metadata.version(name) != value for name, value in info["versions"].items()):
        raise ValueError("Software environment changed since preflight.")

def run_one(root: Path, index: int):
    import torch
    from thyroid_augmentation.extension.evaluate import export_best
    from thyroid_augmentation.paths import ProjectPaths
    from thyroid_augmentation.training.runner import TrainConfig, run_training

    require_preflight(root)
    if not torch.cuda.is_available() or shutil.disk_usage(root).free < 30 * 1024**3:
        raise RuntimeError("GPU unavailable or less than 30 GiB free; stopping.")
    config = jobs(root)[index]
    summary = run_training(ProjectPaths(root, root / "data"), TrainConfig(**config))
    export_best(root, config, summary)

def finalize_rankings(root: Path) -> list[dict]:
    from thyroid_augmentation.extension.evaluate import export_best
    grouped = defaultdict(list)
    for config in jobs(root):
        summary = json.loads((run_directory(root, config) / "summary.json").read_text())
        export_best(root, config, summary)
        grouped[config["dataset"], config["model"], config["augmentation"]].append(summary)
    ranking = []
    for (ds, model, aug), records in grouped.items():
        if sorted(x["seed"] for x in records) != [0, 1, 2]:
            raise ValueError("Ranking needs all three seeds.")
        ranking.append({"dataset": ds, "model": model, "augmentation": aug, "n_seeds": 3,
                        "mean_validation_dice": sum(x["best_dice"] for x in records) / 3})
    ranking.sort(key=lambda x: (x["dataset"], x["model"], -x["mean_validation_dice"], x["augmentation"]))
    best, pools = [], []
    for ds, model in sorted({(r["dataset"], r["model"]) for r in ranking}):
        subset = [r for r in ranking if r["dataset"] == ds and r["model"] == model and r["augmentation"] != "none"]
        best.append(subset[0])
        for rank, row in enumerate(subset, 1):
            row["nonidentity_rank"] = rank
        pools.extend({"dataset": ds, "model": model, "top_n": n,
                      "operations": ["identity"] + [r["augmentation"] for r in subset[:n]],
                      "operations_per_image": 2, "sample_with_replacement": True, "selection_uses": "validation"}
                     for n in range(2, 15))
    for row in ranking:
        if row["augmentation"] == "none":
            row["nonidentity_rank"] = ""
    folder = root / "results/extension/single"
    write_csv(folder / "ranking_by_validation.csv", ranking)
    write_csv(folder / "single_best_by_validation.csv", best)
    write_json(folder / "trivialaugment_pools.json", {"selection_uses": "validation", "pools": pools})
    write_json(folder / "complete.json", {"runs": 405, "status": "complete", "selection_uses": "validation", "source_hashes": source_hashes(root)})
    return best

def run_single(root: Path, script: Path):
    require_preflight(root)
    verify_prepared(root, check_pixels=True)

    import msvcrt

    lock_path = root / "results/extension/single.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        handle.seek(0)
        handle.write(b"0")
        handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        for index, config in enumerate(jobs(root)):
            print(f"Single run {index + 1}/405: {config['dataset']} {config['model']} {config['augmentation']} seed {config['seed']}", flush=True)
            subprocess.run([sys.executable, "-B", str(script), "--action", "run-one", "--confirm-training", "--job-index", str(index)], cwd=root, check=True)
        finalize_rankings(root)
