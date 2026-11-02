from __future__ import annotations

import importlib.metadata
import json
import math
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

from thyroid_augmentation.extension.common import SCANS, SPLITS, digest, protocol, read_csv, write_csv, write_json
from thyroid_augmentation.extension.experiment import jobs as single_jobs
from thyroid_augmentation.extension.experiment import require_preflight as require_single_preflight
from thyroid_augmentation.extension.experiment import source_hashes, verify_prepared
from thyroid_augmentation.randaugment_extension.export import OUTPUT, export_best
from thyroid_augmentation.randaugment_extension.resources import admission, apply_memory_limit, gpu_snapshot, wait_for_gpu

CONFIG = "configs/experiments/extension_randaugment.json"
PHASE = "extension_randaugment"
VERSIONS = ("torch", "numpy", "albumentations", "monai", "torchmetrics", "segmentation-models-pytorch", "psutil")

def settings(root: Path) -> dict:
    value = json.loads((root / CONFIG).read_text())
    p = protocol(root)
    for key, expected in (("phase", PHASE), ("models", p["models"]), ("seeds", p["seeds"]),
                          ("n_values", p["randaugment"]["n_values"]), ("m_values", p["randaugment"]["m_values"]),
                          ("bins", p["randaugment"]["bins"]), ("expected_runs", 324)):
        if value[key] != expected:
            raise ValueError(f"RandAugment protocol mismatch: {key}")
    if set(value["datasets"]) != set(p["datasets"]) or len(value["datasets"]) != 3:
        raise ValueError("Unexpected RandAugment datasets.")
    return value

def jobs(root: Path) -> list[dict]:
    s = settings(root)
    base = single_jobs(root)[0]
    return [{**base, "phase": PHASE, "dataset": ds, "model": model, "seed": seed,
             "augmentation": f"randaugment:n={n},m={m},bins={s['bins']}"}
            for model in s["models"] for ds in s["datasets"] for n in s["n_values"]
            for m in s["m_values"] for seed in s["seeds"]]

def run_directory(root: Path, config: dict, checkpoint=False) -> Path:
    from thyroid_augmentation.paths import ProjectPaths
    from thyroid_augmentation.training.runner import TrainConfig, _run_dir

    run = _run_dir(ProjectPaths(root, root / "data"), TrainConfig(**config))
    return root / "checkpoints" / run.relative_to(root / "runs") if checkpoint else run

def fingerprint(root: Path) -> dict:
    result = source_hashes(root)
    paths = [root / CONFIG, root / "scripts/02_extension_randaugment_queue.py"]
    paths.extend(sorted((root / "src/thyroid_augmentation/randaugment_extension").glob("*.py")))
    result.update({str(path.relative_to(root)): digest(path) for path in paths})
    return result

def require_preflight(root: Path) -> dict:
    require_single_preflight(root)
    report = json.loads((root / OUTPUT / "preflight.json").read_text())
    if report["status"] != "ready_for_randaugment" or report["source_hashes"] != fingerprint(root):
        raise ValueError("RandAugment preflight is missing or stale.")
    if any(importlib.metadata.version(k) != v for k, v in report["versions"].items()):
        raise ValueError("RandAugment software environment changed.")
    return report

@contextmanager
def exclusive_lock(path: Path):
    import msvcrt

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

def storage_requirement(root: Path) -> dict:
    s = settings(root)
    remaining = sum(not (run_directory(root, c, checkpoint=True) / "best.pt").exists()
                    for c in jobs(root) + single_jobs(root))
    required = remaining * s["checkpoint_budget_GiB_per_remaining_run"] + s["disk_reserve_GiB"]
    free = shutil.disk_usage(root).free / 1024**3
    if free < required:
        raise RuntimeError(f"Need {required:.1f} GiB for remaining single+RA models/reserve, have {free:.1f} GiB.")
    return {"free_GiB": free, "required_GiB": required, "remaining_best_files": remaining}

def gpu_smoke(root: Path) -> dict:
    import numpy as np
    import torch
    from thyroid_augmentation.extension.models import build_model
    from thyroid_augmentation.paths import ProjectPaths
    from thyroid_augmentation.training.runner import DiceLoss, SegmentationManifestDataset, _load_scan_mask_map

    s = settings(root)
    state = gpu_snapshot()
    allowed, reason = admission("res34_unet", state, s, concurrent=True)
    if not allowed:
        raise RuntimeError(f"Cannot safely run diagnostic GPU check now: {reason}")
    apply_memory_limit("res34_unet", s)
    torch.manual_seed(0)
    np.random.seed(0)
    scans = _load_scan_mask_map(ProjectPaths(root, root / "data"), SCANS)
    rows = read_csv(root / SPLITS)
    model = build_model(root, "res34_unet", pretrained=True).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000277414, weight_decay=0.000343856)
    scaler = torch.amp.GradScaler("cuda")
    diagnostics = []
    for ds in s["datasets"]:
        selected = [r for r in rows if r["dataset"] == ds and r["split"] == "train"][:2]
        data = SegmentationManifestDataset(selected, root, 224, "randaugment:n=4,m=30,bins=31", scans, training=True)
        images, masks = zip(*(data[i] for i in range(2)))
        x, y = torch.stack(images).repeat(64, 1, 1, 1).cuda(), torch.stack(masks).repeat(64, 1, 1, 1).cuda()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            pred = model(x)
            loss = DiceLoss(sigmoid=True, include_background=False)(pred, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise ValueError("Non-finite RandAugment diagnostic loss/gradients.")
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        del pred
        model.eval()
        with torch.inference_mode():
            pred = model(x)
        if pred.shape != y.shape or not torch.isfinite(pred).all():
            raise ValueError("Invalid RandAugment diagnostic inference.")
        diagnostics.append({"dataset": ds, "loss": float(loss.detach()), "batch": 128,
                            "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 1024**3})
        print(f"RA diagnostic OK: res34_unet/{ds}, N=4 M=30, batch 128", flush=True)
        del x, y, pred, loss
    report = {"status": "passed_diagnostic_only", "model": "res34_unet", "checks": diagnostics,
              "memory_budget_GiB": 10, "single_state_before": state, "experiment_checkpoint_saved": False}
    write_json(root / OUTPUT / "gpu_smoke.json", report)
    return report

def preflight(root: Path, script: Path) -> dict:
    import random
    import numpy as np
    import torch
    from thyroid_augmentation.augmentations.factory import _parse_randaugment_config, build_article_randaugment_transform
    from thyroid_augmentation.paths import ProjectPaths
    from thyroid_augmentation.training.runner import SegmentationManifestDataset, _load_scan_mask_map

    require_single_preflight(root)
    before = fingerprint(root)
    prepared = verify_prepared(root)
    s, grid = settings(root), jobs(root)
    if len(grid) != 324:
        raise ValueError("Incomplete RandAugment grid.")
    longest = max(len(str(run_directory(root, c, checkpoint=b) / "checkpoint_retention.json.tmp"))
                  for c in grid for b in (False, True))
    if longest >= 250:
        raise ValueError(f"Windows path too long: {longest}")
    scans = _load_scan_mask_map(ProjectPaths(root, root / "data"), SCANS)
    rows = read_csv(root / SPLITS)
    checks, mappings = 0, []
    for n in s["n_values"]:
        for m in s["m_values"]:
            name = f"randaugment:n={n},m={m},bins=31"
            policy = _parse_randaugment_config(name)
            if len(policy.operations) != 14 or "identity" in policy.operations or policy.num_ops != n or policy.magnitude != m or policy.magnitude_denominator != 30:
                raise ValueError("Unexpected RandAugment policy mapping.")
            transform = build_article_randaugment_transform(root, 224, name)
            selections = [x for x in transform.transforms if type(x).__name__ == "OneOf"]
            if len(selections) != n or any(len(x.transforms) != 14 for x in selections):
                raise ValueError("Unexpected number of RandAugment operations.")
            mappings.append({"n": n, "m": m, "bins": 31, "scale": m / 30, "operations": list(policy.operations)})
            for ds in s["datasets"]:
                selected = [r for r in rows if r["dataset"] == ds and r["split"] == "train"][:2]
                for seed in s["seeds"]:
                    random.seed(seed)
                    np.random.seed(seed)
                    data = SegmentationManifestDataset(selected, root, 224, name, scans, training=True)
                    for index in range(2):
                        x, y = data[index]
                        if x.shape != (3, 224, 224) or y.shape != (1, 224, 224) or x.dtype != torch.float32 or not torch.isfinite(x).all():
                            raise ValueError(f"Invalid RA data transform {ds}/{name}/{seed}")
                        if x.min() < 0 or x.max() > 1 or not set(y.unique().tolist()).issubset({0, 1}):
                            raise ValueError("RA normalization or binary mask failed.")
                        checks += 1
    space = storage_requirement(root)
    subprocess.run([sys.executable, "-B", str(script), "--action", "smoke"], cwd=root, check=True)
    if before != fingerprint(root):
        raise ValueError("Sources changed during RA preflight; rerun it.")
    base = json.loads((root / "results/extension/preflight.json").read_text())
    report = {"status": "ready_for_randaugment", "source_hashes": before,
              "versions": {name: importlib.metadata.version(name) for name in VERSIONS},
              "samples": prepared["samples"], "runs": len(grid), "longest_path": longest,
              "transform_checks": checks, "policies": mappings, "storage": space,
              "ra_specific_gpu_smoke": json.loads((root / OUTPUT / "gpu_smoke.json").read_text()),
              "shared_model_checks": base["model_smoke_tests"],
              "concurrency": s["optional_concurrent_single"], "throughput_speedup_measured": False}
    write_json(root / OUTPUT / "jobs.json", {"jobs": grid})
    write_json(root / OUTPUT / "preflight.json", report)
    return report

def run_one(root: Path, index: int, concurrent=False):
    import torch
    from thyroid_augmentation.paths import ProjectPaths
    from thyroid_augmentation.training.runner import TrainConfig, run_training

    require_preflight(root)
    config, s = jobs(root)[index], settings(root)
    with exclusive_lock(run_directory(root, config) / "ra_worker.lock"):
        wait_for_gpu(config["model"], s, concurrent)
        apply_memory_limit(config["model"], s)
        if shutil.disk_usage(root).free < s["disk_reserve_GiB"] * 1024**3:
            raise RuntimeError("Disk reserve exhausted; RA stops without touching single.")
        summary = run_training(ProjectPaths(root, root / "data"), TrainConfig(**config))
        torch.cuda.empty_cache()
        export_best(root, config, summary)

def rank_records(records: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped = defaultdict(list)
    for record in records:
        if not math.isfinite(float(record["best_dice"])) or not 0 <= float(record["best_dice"]) <= 1:
            raise ValueError("Invalid validation Dice.")
        grouped[record["dataset"], record["model"], record["augmentation"]].append(record)
    ranking = []
    for (ds, model, aug), group in grouped.items():
        if sorted(x["seed"] for x in group) != [0, 1, 2]:
            raise ValueError("RA ranking needs exactly seeds 0,1,2 per configuration.")
        scores = [float(x["best_dice"]) for x in group]
        ranking.append({"dataset": ds, "model": model, "augmentation": aug, "n_seeds": 3,
                        "mean_validation_dice": statistics.mean(scores), "sd_validation_dice": statistics.stdev(scores)})
    ranking.sort(key=lambda r: (r["dataset"], r["model"], -r["mean_validation_dice"], r["augmentation"]))
    best = [next(r for r in ranking if (r["dataset"], r["model"]) == pair)
            for pair in sorted({(r["dataset"], r["model"]) for r in ranking})]
    return ranking, best

def finalize(root: Path):
    require_preflight(root)
    records = []
    for config in jobs(root):
        record = json.loads((run_directory(root, config) / "summary.json").read_text())
        export_best(root, config, record)
        records.append(record)
    ranking, best = rank_records(records)
    if len(ranking) != 108 or len(best) != 9:
        raise ValueError("Incomplete RA ranking.")
    write_csv(root / OUTPUT / "ranking_by_validation.csv", ranking)
    write_csv(root / OUTPUT / "best_randaugment_by_validation.csv", best)
    write_json(root / OUTPUT / "complete.json", {"status": "complete", "runs": 324, "source_hashes": fingerprint(root)})

def run_all(root: Path, script: Path, concurrent=False):
    require_preflight(root)
    with exclusive_lock(root / OUTPUT / "queue.lock"):
        verify_prepared(root)
        storage_requirement(root)
        for index, config in enumerate(jobs(root)):
            print(f"RandAugment {index + 1}/324: {config['dataset']} {config['model']} {config['augmentation']} seed {config['seed']}", flush=True)
            command = [sys.executable, "-B", str(script), "--action", "run-one", "--confirm-training", "--job-index", str(index)]
            if concurrent:
                command.append("--allow-concurrent-single")
            subprocess.run(command, cwd=root, check=True)
        finalize(root)
