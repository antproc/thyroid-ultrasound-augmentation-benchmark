from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import statistics
import time

import numpy as np
import pandas as pd
from scipy import stats

from thyroid_augmentation.extension.common import digest, read_csv, write_csv, write_json
from thyroid_augmentation.extension.evaluate import pixel_metrics, run_id, validate_export, verify_backup

DATASETS = ("TUCC", "TN3K", "DDTI", "PKTN", "TNUI-2021", "ThyroidXL")
MODELS = ("res34_unet", "deeplabv3plus_resnet50", "mit_b2_unet")
STRATEGIES = ("baseline", "best_single", "trivialaugment", "randaugment")
LABELS = dict(zip(MODELS, ("ResNet34-U-Net", "DeepLabV3+", "MiT-B2-U-Net")))
PHASES = {"m2_single": 15, "m2_ta": 13, "m2_ra": 12}
COMPLETE = "complete_exports_and_local_backup_verified"
MANIFEST = "manifests/metric_v2/splits.csv"
OUTPUT = "results/metric_v2/ood"
PROTOCOL = "per_image_sigmoid05_v2"

def now():
    return datetime.now(timezone.utc).isoformat()

def source_key(config):
    return tuple(config[k] for k in ("phase", "dataset", "model", "augmentation", "seed"))

def evaluation_id(config, target):
    text = json.dumps([run_id(config), target, PROTOCOL], separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]

def select_models(records):
    grouped = defaultdict(list)
    for record in records:
        config = record["config"]
        score = float(record["validation_dice"])
        if config["metric_protocol"] != PROTOCOL or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid corrected validation record.")
        grouped[source_key(config)[:-1]].append(record)
    means = defaultdict(list)
    for key, values in grouped.items():
        if sorted(v["config"]["seed"] for v in values) != [0, 1, 2]:
            raise ValueError(f"Selection requires exactly seeds 0,1,2: {key}")
        means[key[:3]].append((statistics.mean(v["validation_dice"] for v in values), key[3], values))
    selected = []
    for dataset in DATASETS:
        for model in MODELS:
            for strategy in STRATEGIES:
                phase = {"baseline": "m2_single", "best_single": "m2_single",
                         "trivialaugment": "m2_ta", "randaugment": "m2_ra"}[strategy]
                candidates = means[(phase, dataset, model)]
                if len(candidates) != PHASES[phase]:
                    raise ValueError(f"Incomplete configuration grid: {phase}/{dataset}/{model}")
                if strategy == "baseline":
                    candidates = [c for c in candidates if c[1] == "none"]
                elif strategy == "best_single":
                    candidates = [c for c in candidates if c[1] not in ("none", "identity")]
                score, _, values = sorted(candidates, key=lambda c: (-c[0], c[1]))[0]
                for record in sorted(values, key=lambda r: r["config"]["seed"]):
                    selected.append({**record, "strategy": strategy,
                                     "source_validation_mean": score})
    if len(selected) != 216:
        raise ValueError("Expected 216 selected checkpoints.")
    return selected

def implementation_hashes(root):
    names = ["scripts/08_cross_dataset_inference.py", "src/thyroid_augmentation/metric_v2_ood.py",
             "src/thyroid_augmentation/extension/evaluate.py",
             "src/thyroid_augmentation/extension/models.py",
             "src/thyroid_augmentation/training/runner.py",
             "src/thyroid_augmentation/training/metrics.py",
             "src/thyroid_augmentation/augmentations/factory.py"]
    names += [f"configs/published/models/{m}.yaml" for m in MODELS]
    return {name: digest(root / name) for name in names}

def versions():
    return {p: importlib.metadata.version(p) for p in
            ("torch", "numpy", "albumentations", "segmentation-models-pytorch", "scipy", "pandas")}

def prepare(root):
    base, out = root / "results/metric_v2", root / OUTPUT
    for phase, count in (("single", 810), ("ta", 702), ("ra", 648)):
        receipt = json.loads((base / f"{phase}_complete.json").read_text())
        if (receipt["status"], receipt["runs"], receipt["metric_protocol"]) != ("complete", count, PROTOCOL):
            raise ValueError(f"Incomplete corrected stage: {phase}")
    records = []
    for folder in sorted((base / "per_image").iterdir()):
        meta = json.loads((folder / "metadata.json").read_text())
        receipt = json.loads((folder / "complete.json").read_text())
        backup = base / "backup/per_image" / folder.name / "complete.json"
        if receipt["status"] != COMPLETE or digest(folder / "complete.json") != digest(backup):
            raise ValueError(f"Missing verified source export: {folder}")
        if digest(folder / "metadata.json") != receipt["files"]["metadata.json"]:
            raise ValueError("Changed source metadata.")
        summary = json.loads((folder / "training_summary.json").read_text())
        if abs(float(summary["best_dice"]) - float(meta["aggregates"]["val"]["dice"])) > 1e-6:
            raise ValueError("Checkpoint selection and exported validation disagree.")
        records.append({"config": meta["config"], "source_run_id": folder.name,
                        "checkpoint": summary["checkpoint"],
                        "checkpoint_sha256": meta["source_hashes"]["checkpoint_sha256"],
                        "validation_dice": float(meta["aggregates"]["val"]["dice"]),
                        "selected_epoch": meta["selected_epoch"]})
    if len(records) != 2160:
        raise ValueError("Expected all 2160 corrected source runs.")
    selected = select_models(records)
    targets = defaultdict(list)
    all_rows = read_csv(root / MANIFEST)
    for row in all_rows:
        if row["split"] == "test":
            targets[row["dataset"]].append(row)
    for dataset in DATASETS:
        rows = targets[dataset]
        if not rows or len({r["sample_id"] for r in rows}) != len(rows):
            raise ValueError(f"Empty or duplicate target test split: {dataset}")
        for row in rows:
            if not row["group_id"] or not row["group_unit"]:
                raise ValueError("Missing target grouping identity.")
            for field in ("image_path", "mask_path"):
                if not (root / row[field]).is_file():
                    raise FileNotFoundError(row[field])
    print("Verifying the 216 selected checkpoint hashes...", flush=True)
    for i, record in enumerate(selected, 1):
        if digest(Path(record["checkpoint"])) != record["checkpoint_sha256"]:
            raise ValueError("Selected checkpoint hash differs from its certified export.")
        if i % 36 == 0:
            print(f"Checkpoint verification {i}/216", flush=True)
    jobs = [{"evaluation_id": evaluation_id(r["config"], target),
             "source_run_id": r["source_run_id"], "source_dataset": r["config"]["dataset"],
             "target_dataset": target, "model": r["config"]["model"],
             "strategy": r["strategy"], "seed": r["config"]["seed"]}
            for r in selected for target in DATASETS if target != r["config"]["dataset"]]
    longest = max(len(str(out / "backup/per_image" / j["evaluation_id"] / "metadata.json.tmp")) for j in jobs)
    if len(jobs) != 1080 or len({j["evaluation_id"] for j in jobs}) != 1080 or longest >= 250:
        raise ValueError("Duplicate jobs or unsafe Windows paths.")
    if shutil.disk_usage(root).free < 10 * 1024**3:
        raise RuntimeError("At least 10 GiB of free space required for OOD exports.")
    plan = {"metric_protocol": PROTOCOL, "manifest_sha256": digest(root / MANIFEST),
            "implementation_hashes": implementation_hashes(root), "versions": versions(),
            "selected_models": selected, "jobs": jobs, "training": False,
            "selection": "highest mean SOURCE validation Dice over seeds 0,1,2; no target selection",
            "target_split": "test", "batch_size": 128, "precision": "float32",
            "targets": {ds: {"images": len(targets[ds]),
                             "groups": len({r['group_id'] for r in targets[ds]}),
                             "group_unit": sorted({r['group_unit'] for r in targets[ds]})} for ds in DATASETS},
            "longest_export_path": longest,
            "statistics": "image mean over seeds; cluster-robust paired t, G/(G-1), df=G-1; Holm per 90 comparisons",
            "sensitivity": "paired t and Wilcoxon at image and group-mean levels",
            "backup_on_same_disk": True}
    path = out / "plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Refusing to replace a different frozen OOD plan.")
    write_json(path, plan)
    write_csv(out / "selected_checkpoints.csv", [
        {"source_dataset": r["config"]["dataset"], "model": r["config"]["model"],
         "strategy": r["strategy"], "seed": r["config"]["seed"],
         "augmentation": r["config"]["augmentation"], "top_n": r["config"]["top_n"],
         "source_validation_mean": r["source_validation_mean"], "checkpoint": r["checkpoint"],
         "checkpoint_sha256": r["checkpoint_sha256"]} for r in selected])
    print(json.dumps({"selected_checkpoints": len(selected), "evaluations": len(jobs),
                      "targets": plan["targets"], "longest_path": longest}, indent=2), flush=True)
    return plan

def require_plan(root):
    plan = json.loads((root / OUTPUT / "plan.json").read_text())
    if (plan["manifest_sha256"] != digest(root / MANIFEST)
            or plan["implementation_hashes"] != implementation_hashes(root)
            or plan["versions"] != versions() or plan["training"]):
        raise ValueError("OOD source code, manifest or software changed after preparation.")
    return plan

def load_model(root, record):
    import torch
    from thyroid_augmentation.extension.models import build_model
    from thyroid_augmentation.training.metrics import require_protocol

    checkpoint = torch.load(record["checkpoint"], map_location="cpu", weights_only=True)
    require_protocol(checkpoint, record["checkpoint"])
    if checkpoint["config"] != record["config"]:
        raise ValueError("Wrong source checkpoint configuration.")
    model = build_model(root, record["config"]["model"], pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to("cuda").eval()
    del checkpoint
    return model

def smoke(root):
    import torch
    from thyroid_augmentation.extension.evaluate import infer_rows

    plan = require_plan(root)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    manifest = read_csv(root / MANIFEST)
    checks = []
    for record in plan["selected_models"]:
        if record["strategy"] != "baseline" or record["config"]["seed"] != 0:
            continue
        config = record["config"]
        sources = [r for r in manifest if r["dataset"] == config["dataset"] and r["split"] == "test"][:128]
        model = load_model(root, record)
        actual = infer_rows(root, config, model, sources, "test", 128)
        job = {"evaluation_id": evaluation_id(config, config["dataset"]),
               "source_dataset": config["dataset"], "target_dataset": config["dataset"]}
        cached = infer_target(model, record, job, sources, cached_target(root, sources))
        if any(a[k] != b[k] for a, b in zip(actual, cached, strict=True)
               for k in ("sample_id", "group_id", "tp", "fp", "fn", "tn", "dice", "iou")):
            raise ValueError("Cached OOD preprocessing differs from certified export preprocessing.")
        stored = {r["sample_id"]: r for r in read_csv(root / "results/metric_v2/per_image" / record["source_run_id"] / "test.csv")}
        error = max(abs(r["dice"] - float(stored[r["sample_id"]]["dice"])) for r in actual)
        if error > 1e-4:
            raise ValueError(f"Smoke inference disagrees with certified in-domain export: {error}")
        checks.append({"dataset": config["dataset"], "model": config["model"],
                       "samples": len(actual), "max_dice_error": error})
        print(f"Smoke {len(checks)}/18: {config['dataset']}/{config['model']} max error={error:.8f}", flush=True)
        del model
        torch.cuda.empty_cache()
    if len(checks) != 18:
        raise ValueError("Incomplete 18-combination smoke test.")
    write_json(root / OUTPUT / "smoke.json", {"status": "passed", "checks": checks,
               "plan_sha256": digest(root / OUTPUT / "plan.json"), "completed_at": now()})

def completed(root, job):
    folder = root / OUTPUT / "per_image" / job["evaluation_id"]
    receipt = folder / "complete.json"
    if not receipt.exists():
        return False
    data = json.loads(receipt.read_text())
    if data["status"] != COMPLETE:
        raise ValueError("Invalid OOD completion receipt.")
    for name, sha in data["files"].items():
        if digest(folder / name) != sha:
            raise ValueError(f"Changed completed OOD export: {folder / name}")
    metadata = json.loads((folder / "metadata.json").read_text())
    if any(metadata[k] != v for k, v in job.items()):
        raise ValueError("Completed OOD export belongs to a different job.")
    if metadata["plan_sha256"] != digest(root / OUTPUT / "plan.json"):
        raise ValueError("Completed OOD export belongs to a different frozen plan.")
    verify_backup(folder, root / OUTPUT / "backup/per_image" / job["evaluation_id"], data)
    return True

@contextmanager
def queue_lock(root):
    import msvcrt

    path = root / OUTPUT / "queue.lock"
    with path.open("a+b") as handle:
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

def cached_target(root, sources):
    import torch
    from torch.utils.data import TensorDataset
    from thyroid_augmentation.training.runner import SegmentationManifestDataset

    dataset = SegmentationManifestDataset(sources, root, 224, training=False)
    images = torch.empty((len(sources), 3, 224, 224), dtype=torch.float32)
    masks = torch.empty((len(sources), 1, 224, 224), dtype=torch.float32)
    for i in range(len(dataset)):
        images[i], masks[i] = dataset[i]
    return TensorDataset(images, masks)

def infer_target(model, record, job, sources, dataset):
    import torch
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader
    from thyroid_augmentation.training.metrics import per_image_metrics

    config = {**record["config"], "dataset": job["target_dataset"]}
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    device = next(model.parameters()).device
    output, offset = [], 0
    model.eval()
    with torch.inference_mode():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = functional.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            metrics = per_image_metrics(logits, masks)
            values = torch.stack([metrics[k] for k in ("tp", "fp", "fn", "tn")], dim=1).cpu().tolist()
            for source, counts in zip(sources[offset:offset + len(values)], values, strict=True):
                output.append({"run_id": run_id(config), "evaluation_id": job["evaluation_id"],
                               "source_run_id": record["source_run_id"],
                               "source_dataset": job["source_dataset"], "target_dataset": job["target_dataset"],
                               "dataset": job["target_dataset"], "split": "test", "model": config["model"],
                               "strategy": record["strategy"], "configuration_id": config["augmentation"],
                               "seed": config["seed"],
                               **{k: source[k] for k in ("sample_id", "group_id", "group_unit", "case_id", "clip_id")},
                               **pixel_metrics(*counts), "evaluated_height": 224, "evaluated_width": 224})
            offset += len(values)
    validate_export(output, sources, config, "test")
    return output

def save_evaluation(root, record, job, rows, sources, seconds):
    out = root / OUTPUT / "per_image" / job["evaluation_id"]
    backup = root / OUTPUT / "backup/per_image" / job["evaluation_id"]
    config = {**record["config"], "dataset": job["target_dataset"]}
    aggregates = validate_export(rows, sources, config, "test")
    write_csv(out / "test.csv", rows)
    validate_export(read_csv(out / "test.csv"), sources, config, "test")
    meta = {**job, "source_training_config": record["config"],
            "checkpoint": record["checkpoint"], "checkpoint_sha256": record["checkpoint_sha256"],
            "source_validation_mean": record["source_validation_mean"],
            "selected_epoch": record["selected_epoch"], "metric_protocol": PROTOCOL,
            "evaluation_config": config, "aggregates": aggregates, "seconds": seconds,
            "completed_at": now(), "plan_sha256": digest(root / OUTPUT / "plan.json"),
            "training": False, "target_split": "test", "precision": "float32"}
    write_json(out / "metadata.json", meta)
    files = {name: digest(out / name) for name in ("test.csv", "metadata.json")}
    receipt = {"status": COMPLETE, "files": files, "aggregates": aggregates,
               "backup_on_same_disk": True}
    write_json(out / "complete.json", receipt)
    verify_backup(out, backup, receipt)
    return meta

def run(root):
    import torch

    plan = require_plan(root)
    proof = json.loads((root / OUTPUT / "smoke.json").read_text())
    if proof["status"] != "passed" or proof["plan_sha256"] != digest(root / OUTPUT / "plan.json"):
        raise ValueError("Run the GPU smoke check first.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback.")
    torch.backends.cudnn.benchmark = False
    manifest = read_csv(root / MANIFEST)
    targets = {ds: [r for r in manifest if r["dataset"] == ds and r["split"] == "test"] for ds in DATASETS}
    out = root / OUTPUT
    with queue_lock(root):
        started = time.perf_counter()
        done = {j["evaluation_id"] for j in plan["jobs"] if completed(root, j)}
        cache = {}
        try:
            for ds in DATASETS:
                write_json(out / "status.json", {"status": "preparing_target_cache", "target": ds,
                           "complete": len(done), "total": 1080, "updated_at": now()})
                print(f"Caching deterministic evaluation tensors: {ds}, {len(targets[ds])} images", flush=True)
                cache[ds] = cached_target(root, targets[ds])
            for record in plan["selected_models"]:
                todo = [j for j in plan["jobs"] if j["source_run_id"] == record["source_run_id"]
                        and j["evaluation_id"] not in done]
                if not todo:
                    continue
                if shutil.disk_usage(root).free < 5 * 1024**3:
                    raise RuntimeError("Free disk space below 5 GiB; no deletion performed.")
                if digest(Path(record["checkpoint"])) != record["checkpoint_sha256"]:
                    raise ValueError("Selected checkpoint changed after preparation.")
                model = load_model(root, record)
                for job in todo:
                    write_json(out / "status.json", {"status": "running", "active": job,
                               "complete": len(done), "total": 1080, "updated_at": now(),
                               "elapsed_seconds_this_launch": time.perf_counter() - started})
                    tick = time.perf_counter()
                    rows = infer_target(model, record, job, targets[job["target_dataset"]], cache[job["target_dataset"]])
                    meta = save_evaluation(root, record, job, rows, targets[job["target_dataset"]], time.perf_counter() - tick)
                    done.add(job["evaluation_id"])
                    print(f"{len(done)}/1080 {job['source_dataset']} -> {job['target_dataset']} "
                          f"{job['model']} {job['strategy']} seed={job['seed']} "
                          f"Dice={meta['aggregates']['dice']:.6f} seconds={meta['seconds']:.1f}", flush=True)
                del model
                torch.cuda.empty_cache()
            write_json(out / "status.json", {"status": "analyzing", "complete": len(done), "total": 1080,
                       "updated_at": now()})
            analyze(root)
            write_json(out / "status.json", {"status": "complete", "complete": len(done), "total": 1080,
                       "updated_at": now(), "elapsed_seconds_this_launch": time.perf_counter() - started})
        except BaseException as error:
            write_json(out / "status.json", {"status": "failed", "error": repr(error),
                       "complete": len(done), "total": 1080, "updated_at": now()})
            raise

def image_mean(frame, expected_ids):
    if frame.duplicated(["sample_id", "seed"]).any() or set(frame.sample_id) != set(expected_ids):
        raise ValueError("Missing/duplicate image x seed records.")
    if set(frame.seed.unique()) != {0, 1, 2}:
        raise ValueError("Expected all three seeds.")
    if not frame.groupby("sample_id").seed.nunique().eq(3).all():
        raise ValueError("Incomplete image-specific seed triplet.")
    for name in ("group_id", "group_unit", "case_id", "clip_id", "target_pixels"):
        if not frame.groupby("sample_id")[name].nunique(dropna=False).eq(1).all():
            raise ValueError(f"Seed exports disagree on {name}.")
    return frame.groupby("sample_id", sort=True).agg(
        group_id=("group_id", "first"), group_unit=("group_unit", "first"),
        case_id=("case_id", "first"), clip_id=("clip_id", "first"),
        dice=("dice", "mean"), iou=("iou", "mean"), n_seeds=("seed", "nunique")).reset_index()

def cluster_stats(delta, groups):
    d = np.asarray(delta, dtype=float)
    groups = np.asarray(groups, dtype=str)
    if len(d) != len(groups) or len(d) < 2 or not np.isfinite(d).all():
        raise ValueError("Invalid paired differences.")
    _, inv = np.unique(groups, return_inverse=True)
    n, count = len(d), len(np.unique(inv))
    if count < 2:
        raise ValueError("At least two independent groups required.")
    mean = float(d.mean())
    sums = np.bincount(inv, weights=d - mean)
    se = float(np.sqrt(count / (count - 1) * np.sum(sums ** 2) / n ** 2))
    p = float(2 * stats.t.sf(abs(mean / se), count - 1)) if se > 0 else (1.0 if mean == 0 else 0.0)
    margin = float(stats.t.ppf(.975, count - 1)) * se
    return {"n_images": n, "n_units": count, "mean_delta": mean, "se_cluster": se,
            "df": count - 1, "ci95_low": mean - margin, "ci95_high": mean + margin,
            "p_cluster_t": p}

def holm(values):
    p = np.asarray(values, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid p-values.")
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted

def sensitivity(baseline, augmented, groups):
    d = np.asarray(augmented) - np.asarray(baseline)
    def pvalues(x):
        if np.all(x == 0):
            return 1., 1.
        return float(stats.ttest_1samp(x, 0).pvalue), float(stats.wilcoxon(x, zero_method="wilcox", method="auto").pvalue)
    pi, wi = pvalues(d)
    gd = pd.DataFrame({"delta": d, "group": groups}).groupby("group").delta.mean().to_numpy()
    pg, wg = pvalues(gd)
    return {"p_image_t": pi, "p_image_wilcoxon": wi,
            "p_group_t": pg, "p_group_wilcoxon": wg}

def analyze(root):
    plan = require_plan(root)
    out = root / OUTPUT
    manifest = read_csv(root / MANIFEST)
    frames = defaultdict(list)
    run_rows = []
    for job in plan["jobs"]:
        if not completed(root, job):
            raise ValueError("Final OOD analysis requires all 1080 verified evaluations.")
        folder = out / "per_image" / job["evaluation_id"]
        metadata = json.loads((folder / "metadata.json").read_text())
        frame = pd.read_csv(folder / "test.csv", dtype={"sample_id": str, "group_id": str,
                            "group_unit": str, "case_id": str, "clip_id": str}, keep_default_na=False)
        frames[(job["source_dataset"], job["target_dataset"], job["model"], job["strategy"])].append(frame)
        run_rows.append({**job, **metadata["aggregates"], "seconds": metadata["seconds"]})
    pd.DataFrame(run_rows).to_csv(out / "per_seed_summary.csv", index=False)
    means, summaries = {}, []
    for key, seed_frames in frames.items():
        combined = pd.concat(seed_frames, ignore_index=True)
        ids = [r["sample_id"] for r in manifest if r["dataset"] == key[1] and r["split"] == "test"]
        mean = image_mean(combined, ids)
        means[key] = mean
        per_seed = combined.groupby("seed").dice.mean()
        summaries.append({"source_dataset": key[0], "target_dataset": key[1], "model": key[2],
                          "architecture": LABELS[key[2]], "strategy": key[3],
                          "mean_dice": float(mean.dice.mean()), "sd_seed_dice": float(per_seed.std(ddof=1)),
                          "mean_iou": float(mean.iou.mean()), "n_images": len(mean),
                          "n_units": mean.group_id.nunique(), "n_seeds": 3})
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "summary.csv", index=False)
    all_means = pd.concat([frame.assign(source_dataset=key[0], target_dataset=key[1], model=key[2], strategy=key[3])
                           for key, frame in means.items()], ignore_index=True)
    all_means.to_csv(out / "per_image_mean_over_seeds.csv", index=False)
    comparisons = []
    for source in DATASETS:
        for target in DATASETS:
            if source == target:
                continue
            for model in MODELS:
                for comparison in ("best_single", "trivialaugment", "randaugment", "randaugment_minus_trivialaugment"):
                    left = "trivialaugment" if comparison == "randaugment_minus_trivialaugment" else "baseline"
                    right = "randaugment" if comparison == "randaugment_minus_trivialaugment" else comparison
                    baseline = means[(source, target, model, left)].sort_values("sample_id").reset_index(drop=True)
                    augmented = means[(source, target, model, right)].sort_values("sample_id").reset_index(drop=True)
                    if not baseline[["sample_id", "group_id"]].equals(augmented[["sample_id", "group_id"]]):
                        raise ValueError("Unpaired source/target images.")
                    d = augmented.dice.to_numpy() - baseline.dice.to_numpy()
                    comparisons.append({"source_dataset": source, "target_dataset": target, "model": model,
                                        "architecture": LABELS[model], "comparison": comparison,
                                        "reference_mean": float(baseline.dice.mean()),
                                        "augmentation_mean": float(augmented.dice.mean()),
                                        **cluster_stats(d, baseline.group_id),
                                        **sensitivity(baseline.dice, augmented.dice, baseline.group_id)})
    statistics_frame = pd.DataFrame(comparisons)
    for comparison, indices in statistics_frame.groupby("comparison").groups.items():
        if len(indices) != 90:
            raise ValueError("Holm families must each include all 90 source-target-model comparisons.")
        for col in ("p_cluster_t", "p_image_t", "p_image_wilcoxon", "p_group_t", "p_group_wilcoxon"):
            statistics_frame.loc[indices, col + "_holm"] = holm(statistics_frame.loc[indices, col])
    statistics_frame.to_csv(out / "paired_statistics.csv", index=False)
    for strategy in STRATEGIES[1:]:
        table = []
        for source in DATASETS:
            for target in DATASETS:
                if source == target:
                    continue
                row = {"Train": source, "Test": target}
                for model in MODELS:
                    subset = summary[(summary.source_dataset == source) & (summary.target_dataset == target) & (summary.model == model)]
                    for label in ("baseline", strategy):
                        result = subset[subset.strategy == label].iloc[0]
                        row[f"{LABELS[model]} {label}"] = f"{result.mean_dice:.4f} +/- {result.sd_seed_dice:.4f}"
                    s = statistics_frame[(statistics_frame.source_dataset == source) & (statistics_frame.target_dataset == target)
                                         & (statistics_frame.model == model) & (statistics_frame.comparison == strategy)].iloc[0]
                    row[f"{LABELS[model]} delta"] = f"{s.mean_delta:+.4f}"
                table.append(row)
        frame = pd.DataFrame(table)
        frame.to_csv(out / f"table_{strategy}.csv", index=False)
        lines = ["| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns)]
        lines += ["| " + " | ".join(map(str, r)) + " |" for r in frame.itertuples(index=False, name=None)]
        (out / f"table_{strategy}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pd.ExcelWriter(out / "ood_results.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        statistics_frame.to_excel(writer, sheet_name="paired_statistics", index=False)
        pd.DataFrame(run_rows).to_excel(writer, sheet_name="per_seed", index=False)
        for strategy in STRATEGIES[1:]:
            pd.read_csv(out / f"table_{strategy}.csv").to_excel(writer, sheet_name=strategy, index=False)
    files = ["summary.csv", "paired_statistics.csv", "per_seed_summary.csv", "per_image_mean_over_seeds.csv", "ood_results.xlsx"]
    write_json(out / "complete.json", {"status": "complete", "evaluations": 1080, "checkpoints": 216,
               "source_target_directions": 30, "training": False, "completed_at": now(),
               "plan_sha256": digest(out / "plan.json"), "files": {n: digest(out / n) for n in files}})

