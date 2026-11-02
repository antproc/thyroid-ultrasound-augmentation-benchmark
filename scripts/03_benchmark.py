from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from functools import lru_cache
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thyroid_augmentation.extension.common import SCANS, SPLITS, digest, protocol, read_csv, write_csv, write_json
from thyroid_augmentation.extension.evaluate import export_best, run_id
from thyroid_augmentation.extension.experiment import AUGMENTATIONS, source_hashes
from thyroid_augmentation.paths import ProjectPaths
from thyroid_augmentation.training.metrics import METRIC_PROTOCOL, require_protocol
from thyroid_augmentation.training.runner import TrainConfig, _run_dir, run_training
from thyroid_augmentation.extension.pools import ranked_pools, short_paths

OUT = ROOT / "results/benchmark"
MANIFEST = "manifests/all/splits.csv"
SCAN_MANIFEST = "manifests/all/scan_masks.csv"
DATASETS = ("TUCC", "TN3K", "DDTI", "PKTN", "TNUI-2021", "ThyroidXL")
STAGES = {"single": "m2_single", "ta": "m2_ta", "ra": "m2_ra"}
OLD_SPLITS = "manifests/tucc_tn3k_ddti/splits.csv"
OLD_SCANS = "manifests/tucc_tn3k_ddti/scan_masks.csv"

@lru_cache(maxsize=1)
def settings():
    return protocol(ROOT)

def fingerprint():
    result = source_hashes(ROOT)
    for name in (OLD_SPLITS, OLD_SCANS, MANIFEST, SCAN_MANIFEST,
                 "scripts/02_preflight.py", "scripts/03_benchmark.py"):
        result[name] = digest(ROOT / name)
    for name in ("src/thyroid_augmentation/parallel_queue.py", "scripts/04_parallel_gpu_smoke.py"):
        result[name] = digest(ROOT / name)
    return result

def base_config(ds, model, stage, aug, seed, top_n=None):
    p = settings()
    fields = ("epochs", "scheduler_epochs", "image_size", "batch_size", "effective_batch_size",
              "learning_rate", "weight_decay", "dropout", "pretrained")
    return asdict(TrainConfig(**{key: p[key] for key in fields}, dataset=ds, model=model, augmentation=aug,
                            seed=seed, num_workers=0, top_n=top_n, split_manifest=MANIFEST,
                            scan_mask_manifest=SCAN_MANIFEST, device="cuda", resume=True, phase=STAGES[stage]))

def grid(stage, pools=None):
    p = settings()
    if stage == "single":
        return [base_config(ds, model, stage, aug, seed) for ds in DATASETS for model in p["models"]
                for aug in AUGMENTATIONS for seed in p["seeds"]]
    if stage == "ra":
        return [base_config(ds, model, stage, f"randaugment:n={n},m={m},bins=31", seed)
                for ds in DATASETS for model in p["models"] for n in (2, 3, 4)
                for m in (5, 10, 15, 30) for seed in p["seeds"]]
    if stage != "ta" or pools is None:
        raise ValueError("TA requires completed, validation-ranked single results.")
    expected = {(ds, model, n) for ds in DATASETS for model in p["models"] for n in range(2, 15)}
    if len(pools) != len(expected) or {(p["dataset"], p["model"], p["top_n"]) for p in pools} != expected:
        raise ValueError("Incomplete or duplicate TA pools.")
    jobs = []
    for pool in pools:
        ops, n = pool["operations"], pool["top_n"]
        if len(ops) != n + 1 or ops[0] != "identity" or len(set(ops)) != len(ops) or not set(ops[1:]) <= set(AUGMENTATIONS[1:]):
            raise ValueError("Invalid TA operation pool.")
        if pool["selection_uses"] != "validation" or pool["operations_per_image"] != 2 or pool["sample_with_replacement"] is not True:
            raise ValueError("TA protocol differs.")
        jobs.extend(base_config(pool["dataset"], pool["model"], stage, "trivial:" + ",".join(ops), seed, n)
                    for seed in p["seeds"])
    return jobs

def paths(config):
    with short_paths():
        run = _run_dir(ProjectPaths(ROOT, ROOT / "data"), TrainConfig(**config))
    checkpoint = ROOT / "checkpoints" / run.relative_to(ROOT / "runs") / "best.pt"
    return run, checkpoint

def check_storage(jobs):
    sizes = {"res34_unet": 293542791, "deeplabv3plus_resnet50": 320668479, "mit_b2_unet": 330199367}
    remaining = [c for c in jobs if not paths(c)[1].exists()]
    required = sum(sizes[c["model"]] for c in remaining) * 1.05 + 30 * 1024**3
    free = shutil.disk_usage(ROOT).free
    result = {"remaining_best": len(remaining), "free_GiB": free/1024**3,
              "required_GiB_with_reserve": required/1024**3, "fits": free >= required,
              "checkpoint_size_reference": "best.pt bytes per architecture plus 5% and a 30 GiB reserve"}
    return result

def require_preflight():
    info = json.loads((OUT / "prepared.json").read_text())
    if info["source_hashes"] != fingerprint() or info["metric_protocol"] != METRIC_PROTOCOL:
        raise ValueError("Preparation is stale.")
    for name, value in info["versions"].items():
        if importlib.metadata.version(name) != value:
            raise ValueError(f"Environment changed: {name}")
    return info

def require_parallel_preflight():
    info = json.loads((OUT / "parallel_preflight.json").read_text())
    if info.get("status") != "passed" or info["source_hashes"] != fingerprint():
        raise ValueError("Concurrent GPU preflight is missing or stale.")
    for name, value in require_preflight()["versions"].items():
        if info["versions"][name] != value:
            raise ValueError("Concurrent GPU preflight environment changed.")
    expected = {("res34_unet", "res34_unet"), ("res34_unet", "deeplabv3plus_resnet50"),
                ("res34_unet", "mit_b2_unet"), ("deeplabv3plus_resnet50", "deeplabv3plus_resnet50")}
    if {tuple(pair["models"]) for pair in info["pairs"]} != expected:
        raise ValueError("Incomplete concurrent model-pair smoke tests.")
    return info

def prepare():
    smoke = json.loads((ROOT / "results/preflight/preflight.json").read_text())
    if len(smoke["gpu_checks"]) != 18 or smoke["transform_checks"] != 720:
        raise ValueError("All six datasets and three models must pass diagnostic preflight.")
    for name, sha in smoke["source_hashes"].items():
        if digest(ROOT / name) != sha:
            raise ValueError(f"Smoke source changed; rerun script 02: {name}")
    for name, version in smoke["versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Smoke environment changed: {name}")
    rows = read_csv(ROOT / OLD_SPLITS) + read_csv(ROOT / SPLITS)
    units = {"TUCC": "patient_group_from_original_manifest", "TN3K": "image_id_not_patient", "DDTI": "case"}
    for row in rows:
        row.setdefault("group_unit", units.get(row["dataset"], "unknown"))
    scans = read_csv(ROOT / OLD_SCANS) + read_csv(ROOT / SCANS)

    scan_keys = list(dict.fromkeys(k for row in scans for k in row))
    scans = [{k: row.get(k, "") for k in scan_keys} for row in scans]
    for name, records in ((MANIFEST, rows), (SCAN_MANIFEST, scans)):
        path = ROOT / name
        if path.exists() and read_csv(path) != records:
            raise ValueError(f"Refusing to overwrite a different frozen manifest: {path}")
        if not path.exists():
            write_csv(path, records)
    previous = OUT / "prepared.json"
    if previous.exists() and json.loads(previous.read_text())["source_hashes"] != fingerprint():
        if any(any((ROOT / folder / phase).glob("**/*")) for folder in ("runs", "checkpoints") for phase in STAGES.values()):
            raise ValueError("Runs already exist with a different code fingerprint. Use a new namespace, not a refreshed preflight.")
    placeholder = [{"dataset": ds, "model": model, "top_n": n, "operations": ["identity"] + AUGMENTATIONS[1:n+1],
                    "selection_uses": "validation", "operations_per_image": 2, "sample_with_replacement": True}
                   for ds in DATASETS for model in settings()["models"] for n in range(2, 15)]
    grids = {stage: grid(stage, placeholder if stage == "ta" else None) for stage in STAGES}
    longest = max(max(len(str(paths(c)[0] / "checkpoint_retention.json.tmp")),
                      len(str(paths(c)[1].with_name("last.pt.tmp")))) for jobs in grids.values() for c in jobs)
    if longest >= 250:
        raise ValueError(f"Unsafe Windows path length: {longest}")
    all_jobs = [c for jobs in grids.values() for c in jobs]
    if len({run_id(c) for c in all_jobs}) != 2160:
        raise ValueError("Incorrect or duplicate grid.")
    write_json(OUT / "single_jobs.json", grids["single"])
    write_json(OUT / "ra_jobs.json", grids["ra"])
    info = {"metric_protocol": METRIC_PROTOCOL, "source_hashes": fingerprint(), "versions": smoke["versions"],
            "training_started": False, "datasets": DATASETS, "protocol": settings(),
            "counts": {stage: len(jobs) for stage, jobs in grids.items()}, "longest_path": longest,
            "storage": {stage: check_storage(jobs) for stage, jobs in grids.items()},
            "all_stages_storage": check_storage(all_jobs),
            "single_plus_ra_storage": check_storage(grids["single"] + grids["ra"]),
            "ta_pools": "built from the single-stage validation ranking; the pools above only verify paths and counts",
            "storage_policy": "serial mode checks a full stage; parallel mode reserves all unfinished single checkpoints and admits RA from remaining capacity; 30 GiB reserve"}
    write_json(OUT / "prepared.json", info)
    print(json.dumps({k: info[k] for k in ("counts", "longest_path", "storage", "all_stages_storage")}, indent=2), flush=True)

def export_and_verify(config):
    directory, _ = paths(config)
    summary = json.loads((directory / "summary.json").read_text())
    require_protocol(summary, str(directory))
    receipt = export_best(ROOT, config, summary, config["batch_size"], output_root=OUT,
                          source_fingerprint=require_preflight()["source_hashes"])
    if abs(receipt["aggregates"]["val"]["dice"] - summary["best_dice"]) > 1e-6:
        raise ValueError("Selected validation Dice differs from reloaded checkpoint per-image export.")
    return summary

def finish_stage(stage, jobs):
    records = [export_and_verify(c) for c in jobs]
    if stage == "single":
        write_json(OUT / "ta_pools.json", ranked_pools(records))
    grouped = defaultdict(list)
    for record in records:
        grouped[record["dataset"], record["model"], record["augmentation"]].append(record)
    ranking = []
    for (ds, model, aug), values in grouped.items():
        if sorted(r["seed"] for r in values) != [0, 1, 2]:
            raise ValueError("Ranking requires all three seeds.")
        ranking.append({"dataset": ds, "model": model, "augmentation": aug,
                        "mean_validation_dice": sum(r["best_dice"] for r in values)/3})
    ranking.sort(key=lambda r: (r["dataset"], r["model"], -r["mean_validation_dice"], r["augmentation"]))
    write_csv(OUT / f"{stage}_ranking.csv", ranking)
    write_json(OUT / f"{stage}_complete.json", {"status": "complete", "runs": len(jobs),
               "source_hashes": require_preflight()["source_hashes"], "metric_protocol": METRIC_PROTOCOL,
               "ta_pools_sha256": digest(OUT / "ta_pools.json") if stage == "single" else None})

def stage_jobs(stage):
    if stage == "ta":
        receipt = json.loads((OUT / "single_complete.json").read_text())
        if receipt["source_hashes"] != require_preflight()["source_hashes"] or receipt["runs"] != 810 or receipt["status"] != "complete":
            raise ValueError("Single stage is not successfully complete.")
        if receipt["ta_pools_sha256"] != digest(OUT / "ta_pools.json"):
            raise ValueError("TA pools changed after validation ranking.")
        return grid(stage, json.loads((OUT / "ta_pools.json").read_text()))
    return grid(stage)

@contextmanager
def queue_lock():
    import msvcrt
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "queue.lock").open("a+b") as handle:
        handle.seek(0)
        handle.write(b"0")
        handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

def run_one(stage, index, memory_cap=False):
    import torch
    require_preflight()
    if not torch.cuda.is_available() or shutil.disk_usage(ROOT).free < 30*1024**3:
        raise RuntimeError("CUDA required and at least 30 GiB disk reserve. No fallback or deletion.")
    config = stage_jobs(stage)[index]
    if memory_cap:
        require_parallel_preflight()
        from thyroid_augmentation.parallel_queue import limit_memory
        limit_memory(config["model"])
    with short_paths():
        run_training(ProjectPaths(ROOT, ROOT / "data"), TrainConfig(**config))
    export_and_verify(config)

def run_queue(stage):
    require_preflight()
    stages = list(STAGES) if stage == "all" else [stage]
    with queue_lock():
        for current in stages:
            jobs = stage_jobs(current)
            storage = check_storage(jobs)
            if not storage["fits"]:
                write_json(OUT / "storage_blocked.json", {"stage": current, **storage})
                raise RuntimeError(f"Insufficient C: space for {current}: {storage}")
            for index, config in enumerate(jobs):
                print(f"{current} {index+1}/{len(jobs)}: {config['dataset']} {config['model']} {config['augmentation']} seed {config['seed']}", flush=True)
                subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "run-one", "--stage", current,
                                "--index", str(index), "--confirm-training"], cwd=ROOT, check=True)
            finish_stage(current, jobs)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "run-parallel", "run-one", "status"))
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    parser.add_argument("--index", type=int)
    parser.add_argument("--confirm-training", action="store_true")
    parser.add_argument("--gpu-memory-cap", action="store_true")
    args = parser.parse_args()
    if args.action in ("run", "run-parallel", "run-one") and not args.confirm_training:
        parser.error("Explicit --confirm-training is required.")
    if args.action == "prepare":
        prepare()
    elif args.action == "run":
        run_queue(args.stage)
    elif args.action == "run-parallel":
        if args.stage != "all":
            parser.error("run-parallel coordinates single + RA, followed by TA; use --stage all")
        from thyroid_augmentation.parallel_queue import run
        run(sys.modules[__name__])
    elif args.action == "run-one":
        if args.stage == "all" or args.index is None or args.index < 0:
            parser.error("run-one requires a specific stage and nonnegative --index")
        run_one(args.stage, args.index, args.gpu_memory_cap)
    else:
        for stage, phase in STAGES.items():
            count = len(list((ROOT / "runs" / phase).glob("**/summary.json")))
            print(f"{stage}: {count} training summaries; complete={bool((OUT / (stage+'_complete.json')).exists())}")

if __name__ == "__main__":
    main()
