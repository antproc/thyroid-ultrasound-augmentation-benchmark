from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from thyroid_augmentation.extension.common import SPLITS, SCANS, digest, protocol, read_csv, write_csv, write_json
from thyroid_augmentation.extension.experiment import AUGMENTATIONS, jobs as single_jobs, source_hashes, verify_prepared
from thyroid_augmentation.extension.evaluate import run_id
from thyroid_augmentation.randaugment_extension import pipeline as ra
from thyroid_augmentation.trivial_extension.resources import ProcessWatch, limit_memory, wait_for_slot

OUTPUT = "results/extension_trivialaugment"
PHASE = "extension_trivialaugment"
SCRIPT = "scripts/03_extension_trivialaugment_queue.py"

def fingerprint(root):
    hashes = ra.fingerprint(root)
    paths = [root / SCRIPT] + sorted((root / "src/thyroid_augmentation/trivial_extension").glob("*.py"))
    hashes.update({str(p.relative_to(root)): digest(p) for p in paths})
    return hashes

def ranked_pools(records):
    groups = defaultdict(list)
    for r in records:
        score = float(r["best_dice"])
        if not math.isfinite(score) or not 0 <= score <= 1 or r["augmentation"] not in AUGMENTATIONS:
            raise ValueError("Invalid single result.")
        groups[r["dataset"], r["model"], r["augmentation"]].append(r)
    means = {}
    for key, values in groups.items():
        if sorted(r["seed"] for r in values) != [0, 1, 2]:
            raise ValueError("Single ranking requires exactly seeds 0,1,2.")
        means[key] = sum(r["best_dice"] for r in values) / 3
    pools = []
    for ds, model in sorted({k[:2] for k in means}):
        if {k[2] for k in means if k[:2] == (ds, model)} != set(AUGMENTATIONS):
            raise ValueError("All 15 single conditions are required.")
        order = sorted(AUGMENTATIONS[1:], key=lambda a: (-means[ds, model, a], a))
        pools.extend({"dataset": ds, "model": model, "top_n": n, "operations": ["identity"] + order[:n],
                      "operations_per_image": 2, "sample_with_replacement": True, "selection_uses": "validation"}
                     for n in range(2, 15))
    return pools

def jobs(root, pools):
    base = single_jobs(root)[0]
    p = protocol(root)
    expected = {(ds, m, n) for ds in p["datasets"] for m in p["models"] for n in range(2, 15)}
    if len(pools) != 117 or {(r["dataset"], r["model"], r["top_n"]) for r in pools} != expected:
        raise ValueError("Expected 117 Top-N pools.")
    out = []
    for pool in sorted(pools, key=lambda r: (p["models"].index(r["model"]), r["dataset"], r["top_n"])):
        n, ops = pool["top_n"], pool["operations"]
        if len(ops) != n + 1 or ops[0] != "identity" or len(set(ops)) != len(ops) or not set(ops[1:]) <= set(AUGMENTATIONS[1:]):
            raise ValueError("Invalid Top-N operations.")
        if pool["operations_per_image"] != 2 or pool["sample_with_replacement"] is not True or pool["selection_uses"] != "validation":
            raise ValueError("Wrong TrivialAugment protocol.")
        for seed in p["seeds"]:
            out.append({**base, "phase": PHASE, "dataset": pool["dataset"], "model": pool["model"],
                        "seed": seed, "top_n": n, "augmentation": "trivial:" + ",".join(ops)})
    if len({run_id(c) for c in out}) != 351:
        raise ValueError("Nonunique TrivialAugment run identity.")
    return out

def short_name(value):
    if not value.startswith("trivial:"):
        raise ValueError("Only explicit TA policies may use the short path adapter.")
    n = len(value.split(":", 1)[1].split(",")) - 1
    return f"top{n:02d}_{hashlib.sha256(value.encode()).hexdigest()[:12]}"

def run_directory(root, config, checkpoint=False):
    return root / ("checkpoints" if checkpoint else "runs") / PHASE / config["dataset"] / config["model"] / short_name(config["augmentation"]) / f"seed_{config['seed']}"

@contextmanager
def short_paths():

    from thyroid_augmentation.training import runner
    old = runner._safe_name
    with patch.object(runner, "_safe_name", side_effect=lambda value: short_name(value) if value.startswith("trivial:") else old(value)):
        yield

def storage(root, grid):
    remaining = sum(not (run_directory(root, c, True) / "best.pt").exists() for c in grid)
    remaining += sum(not (ra.run_directory(root, c, True) / "best.pt").exists() for c in ra.jobs(root) + single_jobs(root))
    free, required = shutil.disk_usage(root).free / 1024**3, remaining * .4 + 30
    if free < required:
        raise RuntimeError(f"Insufficient disk: {free:.1f} GiB free, {required:.1f} GiB required for all remaining stages.")
    return {"free_GiB": free, "required_GiB": required, "remaining_best_files": remaining}

def completion_ok(receipt, hashes, runs):
    if receipt.get("status") != "complete" or receipt.get("runs") != runs or receipt.get("source_hashes") != hashes:
        raise ValueError("Completion receipt is absent, stale or incomplete.")

def require_ra_success(root):
    path = root / "results/extension_randaugment/complete.json"
    if not path.exists():
        raise RuntimeError("RA is not running and has no successful completion receipt; stop for inspection.")
    completion_ok(json.loads(path.read_text()), ra.fingerprint(root), 324)

def verify_single(root):
    from thyroid_augmentation.extension.evaluate import export_best
    ra.require_preflight(root)
    folder = root / "results/extension/single"
    completion_ok(json.loads((folder / "complete.json").read_text()), source_hashes(root), 405)
    records = []
    for i, c in enumerate(single_jobs(root)):
        run = ra.run_directory(root, c)
        summary = json.loads((run / "summary.json").read_text())
        if not (root / "results/extension/per_image" / run_id(c) / "complete.json").exists():
            raise ValueError("Missing single per-image completion; do not infer or train during gate verification.")
        retention = json.loads((run / "checkpoint_retention.json").read_text())
        if retention["status"] != "complete_best_retained_last_absent" or retention["epochs"] != 50:
            raise ValueError("Single training verification missing.")
        for key, path in (("summary_sha256", run / "summary.json"), ("log_sha256", run / "train_log.csv"),
                          ("best_sha256", Path(summary["checkpoint"]))):
            if digest(path) != retention[key]:
                raise ValueError("Single training artifact changed.")
        for key in ("dataset", "model", "augmentation", "seed", "top_n", "epochs"):
            if summary[key] != c[key]:
                raise ValueError("Wrong single summary identity.")
        export_best(root, c, summary)
        records.append(summary)
        if i % 30 == 0:
            print(f"Single gate: verified {i + 1}/405", flush=True)
    pools = ranked_pools(records)
    published = json.loads((folder / "trivialaugment_pools.json").read_text())
    if published != {"selection_uses": "validation", "pools": pools}:
        raise ValueError("Single Top-N pools differ from independent validation-only reconstruction.")
    return pools

def require_preflight(root, ready=True):
    ra.require_preflight(root)
    name = "preflight.json" if ready else "preparation.json"
    report = json.loads((root / OUTPUT / name).read_text())
    if report["source_hashes"] != fingerprint(root):
        raise ValueError("TA sources changed since verification.")
    if any(importlib.metadata.version(k) != v for k, v in report["versions"].items()):
        raise ValueError("TA environment changed.")
    if ready:
        if report["status"] != "ready_after_successful_single":
            raise ValueError("TA gate not satisfied.")
        for name, sha in report["frozen_inputs"].items():
            if digest(root / name) != sha:
                raise ValueError("Frozen single selection/TA jobs changed.")
        completion_ok(json.loads((root / "results/extension/single/complete.json").read_text()), source_hashes(root), 405)
    elif report["status"] != "prepared_waiting_for_single":
        raise ValueError("TA preparation not verified.")
    return report

def transform_checks(root, grid):
    import random
    import numpy as np
    import torch
    from thyroid_augmentation.augmentations.factory import build_article_trivial_augment_transform, _parse_trivial_order
    from thyroid_augmentation.paths import ProjectPaths
    from thyroid_augmentation.training.runner import SegmentationManifestDataset, _load_scan_mask_map
    rows = read_csv(root / SPLITS)
    scans = _load_scan_mask_map(ProjectPaths(root, root / "data"), SCANS)
    count = 0
    for c in grid:
        random.seed(c["seed"])
        np.random.seed(c["seed"])
        transform = build_article_trivial_augment_transform(c["top_n"], root, 224, c["augmentation"])
        choices = transform.transforms[3].transforms
        if len(choices) != 2 or any(len(x.transforms) != c["top_n"] + 1 for x in choices):
            raise ValueError("TA must draw two operations from identity plus Top-N.")
        if _parse_trivial_order(c["augmentation"]) != c["augmentation"].split(":", 1)[1].split(","):
            raise ValueError("TA policy parsing differs.")
        selected = [r for r in rows if r["dataset"] == c["dataset"] and r["split"] == "train"][:2]
        data = SegmentationManifestDataset(selected, root, 224, c["augmentation"], scans, True, c["top_n"])
        for i in range(2):
            x, y = data[i]
            if x.shape != (3, 224, 224) or y.shape != (1, 224, 224) or not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1 or not set(y.unique().tolist()) <= {0, 1}:
                raise ValueError("Invalid TA transformed image/mask.")
            count += 1
    return count

def prepare(root):
    ra.require_preflight(root)
    before = fingerprint(root)

    records = [{**c, "best_dice": .5} for c in single_jobs(root)]
    grid = jobs(root, ranked_pools(records))
    longest = max(len(str(run_directory(root, c, b) / "checkpoint_retention.json.tmp")) for c in grid for b in (False, True))
    if longest >= 250:
        raise ValueError(f"Windows path too long: {longest}")
    checks = transform_checks(root, grid)
    if before != fingerprint(root):
        raise ValueError("Source changed during preparation.")
    report = {"status": "prepared_waiting_for_single", "source_hashes": before, "runs": 351,
              "versions": {p: importlib.metadata.version(p) for p in ra.VERSIONS}, "longest_path": longest,
              "transform_checks": checks, "storage": storage(root, grid),
              "gpu_smoke": "deferred until single succeeds; no third training process now",
              "shared_model_checks": json.loads((root / "results/extension/preflight.json").read_text())["model_smoke_tests"]}
    write_json(root / OUTPUT / "preparation.json", report)
    return report

def gpu_smoke(root, grid):
    import numpy as np
    import torch
    from thyroid_augmentation.extension.models import build_model
    from thyroid_augmentation.paths import ProjectPaths
    from thyroid_augmentation.training.runner import DiceLoss, SegmentationManifestDataset, _load_scan_mask_map
    wait_for_slot(root, "res34_unet")
    limit_memory("res34_unet")
    torch.manual_seed(0)
    np.random.seed(0)
    rows = read_csv(root / SPLITS)
    scans = _load_scan_mask_map(ProjectPaths(root, root / "data"), SCANS)
    model = build_model(root, "res34_unet", pretrained=True).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=.000277414, weight_decay=.000343856)
    scaler = torch.amp.GradScaler("cuda")
    checks = []
    for ds in protocol(root)["datasets"]:
        c = next(c for c in grid if c["dataset"] == ds and c["model"] == "res34_unet" and c["top_n"] == 14)
        selected = [r for r in rows if r["dataset"] == ds and r["split"] == "train"][:2]
        data = SegmentationManifestDataset(selected, root, 224, c["augmentation"], scans, True, 14)
        x, y = zip(*(data[i] for i in range(2)))
        x, y = torch.stack(x).repeat(64, 1, 1, 1).cuda(), torch.stack(y).repeat(64, 1, 1, 1).cuda()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            pred = model(x)
            loss = DiceLoss(sigmoid=True, include_background=False)(pred, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise ValueError("Invalid TA smoke loss/gradients.")
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        with torch.inference_mode():
            pred = model(x)
        if pred.shape != y.shape or not torch.isfinite(pred).all():
            raise ValueError("Invalid TA smoke inference.")
        checks.append({"dataset": ds, "loss": float(loss.detach()), "peak_reserved_GiB": torch.cuda.max_memory_reserved()/1024**3})
        del x, y, pred, loss
    write_json(root / OUTPUT / "gpu_smoke.json", {"status": "passed", "checks": checks, "experimental_checkpoint_saved": False})

def gate(root, script):
    prepared = require_preflight(root, ready=False)
    pools = verify_single(root)
    grid = jobs(root, pools)
    verify_prepared(root)
    space = storage(root, grid)
    checks = transform_checks(root, grid)
    job_path = root / OUTPUT / "jobs.json"
    content = {"jobs": grid, "pools": pools}
    if job_path.exists() and json.loads(job_path.read_text()) != content:
        raise ValueError("Refusing to replace frozen TA jobs with changed pools.")
    write_json(job_path, content)
    subprocess.run([sys.executable, "-B", str(script), "--action", "smoke", "--confirm-training"], cwd=root, check=True)
    files = ["results/extension/single/complete.json", "results/extension/single/trivialaugment_pools.json", OUTPUT + "/jobs.json"]
    report = {**prepared, "status": "ready_after_successful_single", "transform_checks": checks, "storage": space,
              "frozen_inputs": {p: digest(root / p) for p in files},
              "gpu_smoke": json.loads((root / OUTPUT / "gpu_smoke.json").read_text())}
    write_json(root / OUTPUT / "preflight.json", report)

def run_one(root, index):
    import torch
    from thyroid_augmentation.paths import ProjectPaths
    from thyroid_augmentation.training.runner import TrainConfig, run_training
    from thyroid_augmentation.trivial_extension.export import export_best
    require_preflight(root)
    grid = json.loads((root / OUTPUT / "jobs.json").read_text())["jobs"]
    c = grid[index]
    with ra.exclusive_lock(root / OUTPUT / "worker.lock"):
        storage(root, grid)
        wait_for_slot(root, c["model"])
        limit_memory(c["model"])
        with short_paths():
            summary = run_training(ProjectPaths(root, root / "data"), TrainConfig(**c))
        torch.cuda.empty_cache()
        export_best(root, c, summary)

def run_queue(root, script):
    require_preflight(root)
    grid = json.loads((root / OUTPUT / "jobs.json").read_text())["jobs"]
    records = []
    for i, c in enumerate(grid):
        print(f"TrivialAugment {i+1}/351: {c['dataset']} {c['model']} Top-{c['top_n']} seed {c['seed']}", flush=True)
        subprocess.run([sys.executable, "-B", str(script), "--action", "run-one", "--job-index", str(i), "--confirm-training"], cwd=root, check=True)
        records.append(json.loads((run_directory(root, c) / "summary.json").read_text()))
    ranking, best = ra.rank_records(records)
    if len(ranking) != 117 or len(best) != 9:
        raise ValueError("Incomplete TA ranking.")
    for r in ranking:
        r["top_n"] = len(r["augmentation"].split(":", 1)[1].split(",")) - 1
    require_preflight(root)
    write_csv(root / OUTPUT / "ranking_by_validation.csv", ranking)
    write_csv(root / OUTPUT / "best_trivialaugment_by_validation.csv", best)
    write_json(root / OUTPUT / "complete.json", {"status": "complete", "runs": 351, "source_hashes": fingerprint(root)})

def arm(root, script, single_pid, ra_pid):
    require_preflight(root, ready=False)
    with ra.exclusive_lock(root / OUTPUT / "queue.lock"):
        single = ProcessWatch(single_pid, "01_prepare_extension_datasets.py", "run-single")
        random = None
        try:
            random = ProcessWatch(ra_pid, "02_extension_randaugment_queue.py", "run")
            write_json(root / OUTPUT / "armed.json", {"status": "waiting_for_single_success", "single": single.identity,
                                                      "randaugment": random.identity, "source_hashes": fingerprint(root)})
            print("Armed: waiting for single exit code 0 plus verified 405 runs; no TA training yet.", flush=True)
            while single.exit_code() is None:
                require_preflight(root, ready=False)
                code = random.exit_code()
                if code not in (None, 0):
                    raise RuntimeError(f"RA exited with code {code}; automatic TA start cancelled.")
                time.sleep(30)
            code = single.exit_code()
            write_json(root / OUTPUT / "single_exit.json", {"identity": single.identity, "exit_code": code})
            if code != 0:
                raise RuntimeError(f"Single exited with code {code}; TA will not start.")
            if random.exit_code() not in (None, 0):
                raise RuntimeError("RA failed before TA gate; cancelled.")
            print("Single exited successfully; checking exports, pools, disk and GPU smoke.", flush=True)
            gate(root, script)
            run_queue(root, script)
        except BaseException as error:
            write_json(root / OUTPUT / "stopped.json", {"status": "stopped_on_error", "error": repr(error)})
            raise
        finally:
            single.close()
            if random:
                random.close()
