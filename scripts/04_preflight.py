from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thyroid_augmentation.extension.common import SCANS, SPLITS, digest, protocol, read_csv, write_csv, write_json
from thyroid_augmentation.extension.experiment import AUGMENTATIONS, source_hashes, verify_prepared
from thyroid_augmentation.training.metrics import METRIC_PROTOCOL, METRIC_SPEC

OUT = ROOT / "results/metric_v2_preflight"
DATASETS = ("TUCC", "TN3K", "DDTI", "PKTN", "TNUI-2021", "ThyroidXL")
OLD_SPLITS = "manifests/splits_primary_benchmark.csv"
OLD_SCANS = "manifests/scan_masks_primary_benchmark.csv"

def manifests():
    rows, scans = read_csv(ROOT / OLD_SPLITS) + read_csv(ROOT / SPLITS), {}
    for row in rows:
        row.setdefault("group_unit", {"TUCC": "patient_group_from_original_manifest",
                                      "TN3K": "image_id_not_patient", "DDTI": "case"}[row["dataset"]]
                       if row["dataset"] in DATASETS[:3] else "unknown")
    for path in (OLD_SCANS, SCANS):
        for row in read_csv(ROOT / path):
            key = row["dataset"], row["sample_id"]
            if key in scans:
                raise ValueError(f"Duplicate scan key: {key}")
            scans[key] = row["scan_mask_path"]
    expected = {"TUCC": (7441, 807, 1970), "TN3K": (2303, 576, 614), "DDTI": (278, 102, 86),
                "PKTN": (603, 201, 201), "TNUI-2021": (966, 276, 139), "ThyroidXL": (7633, 1908, 2094)}
    counts, seen, groups = Counter(), set(), defaultdict(set)
    for row in rows:
        key = row["dataset"], row["sample_id"]
        if key in seen or row["dataset"] not in expected or row["split"] not in ("train", "val", "test"):
            raise ValueError(f"Invalid/duplicate split entry: {key}")
        seen.add(key)
        counts[row["dataset"], row["split"]] += 1
        groups[row["dataset"], row["group_id"]].add(row["split"])
        for name in (row["image_path"], row["mask_path"], scans[key]):
            if not (ROOT / name).is_file():
                raise FileNotFoundError(name)
    for ds, values in expected.items():
        if tuple(counts[ds, s] for s in ("train", "val", "test")) != values:
            raise ValueError(f"Unexpected frozen split count: {ds}")
    leaks = [key for key, splits in groups.items() if len(splits) > 1]
    if leaks:
        raise ValueError(f"Group overlap across splits: {leaks[:10]}")
    write_csv(OUT / "frozen_split_inventory.csv", [
        {"dataset": ds, "split": split, "images": counts[ds, split],
         "groups": len({r['group_id'] for r in rows if r['dataset'] == ds and r['split'] == split})}
        for ds in DATASETS for split in ("train", "val", "test")])
    return rows, scans

def validate_transforms(rows, scans):
    import numpy as np
    import torch
    from thyroid_augmentation.training.runner import SegmentationManifestDataset

    policies = AUGMENTATIONS + ["trivial:" + ",".join(["identity"] + AUGMENTATIONS[1:n+1]) for n in range(2, 15)]
    policies += [f"randaugment:n={n},m={m},bins=31" for n in (2, 3, 4) for m in (5, 10, 15, 30)]
    checked = []
    for ds in DATASETS:
        sources = [r for r in rows if r["dataset"] == ds and r["split"] == "train"][:2]
        for policy in policies:
            for seed in (0, 1, 2):
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                top_n = len(policy.split(":", 1)[1].split(",")) - 1 if policy.startswith("trivial:") else None
                dataset = SegmentationManifestDataset(sources, ROOT, 224, augmentation=policy,
                                                       training=True, scan_mask_map=scans, top_n=top_n)
                for image, mask in (dataset[i] for i in range(len(dataset))):
                    if image.shape != (3, 224, 224) or mask.shape != (1, 224, 224):
                        raise ValueError(f"Invalid transform shape: {ds}/{policy}")
                    if not torch.isfinite(image).all() or not torch.all((mask == 0) | (mask == 1)):
                        raise ValueError(f"Invalid transform values: {ds}/{policy}")
                checked.append({"dataset": ds, "policy": policy, "seed": seed, "images": len(sources)})
        print(f"Transforms OK: {ds}, 40 configurations x 3 seeds x 2 images", flush=True)
    write_csv(OUT / "transform_checks.csv", checked)
    return len(checked)

def validate_gpu(rows, scans, settings):
    import torch
    from torch.utils.data import DataLoader
    from thyroid_augmentation.extension.evaluate import infer_rows, validate_export
    from thyroid_augmentation.extension.models import build_model
    from thyroid_augmentation.training.runner import DiceLoss, SegmentationManifestDataset, _evaluate

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU fallback.")
    report = []
    for name in settings["models"]:
        torch.manual_seed(0)
        model = build_model(ROOT, name, pretrained=True).cuda()
        for ds in DATASETS:
            started = time.perf_counter()
            selected = [r for r in rows if r["dataset"] == ds and r["split"] == "train"][:2]
            data = SegmentationManifestDataset(selected, ROOT, 224, augmentation="none", training=True, scan_mask_map=scans)
            images, masks = zip(*(data[i] for i in range(2)))
            x, y = torch.stack(images).repeat(64, 1, 1, 1).cuda(), torch.stack(masks).repeat(64, 1, 1, 1).cuda()
            model.train()
            model.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            with torch.amp.autocast("cuda"):
                pred = model(x)
                loss = DiceLoss(sigmoid=True, include_background=False)(pred, y)
            loss.backward()
            if pred.shape != y.shape or not torch.isfinite(loss) or not all(
                torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
            ):
                raise ValueError(f"Invalid training smoke: {ds}/{name}")
            record = {"dataset": ds, "model": name, "batch_size": 128, "loss": float(loss.detach()),
                      "peak_training_MiB": torch.cuda.max_memory_allocated()/1024**2}
            del x, y, pred, loss
            model.zero_grad(set_to_none=True)
            config = {"phase": "metric_v2_diagnostic", "dataset": ds, "model": name,
                      "augmentation": "none", "seed": 0, "image_size": 224, "metric_protocol": METRIC_PROTOCOL}
            for split in ("val", "test"):
                sources = [r for r in rows if r["dataset"] == ds and r["split"] == split][:2]
                outputs = infer_rows(ROOT, config, model, sources, split, 2)
                aggregates = validate_export(outputs, sources, config, split)
                path = OUT / "diagnostic_per_image" / ds / name / f"{split}.csv"
                write_csv(path, outputs)
                validate_export(read_csv(path), sources, config, split)
                eval_data = SegmentationManifestDataset(sources, ROOT, 224, training=False)
                metric = _evaluate(model, DataLoader(eval_data, batch_size=2), torch.device("cuda"))
                for key in ("dice", "iou"):
                    if abs(metric[key] - aggregates[key]) > 1e-12:
                        raise ValueError(f"Selection/export mismatch: {ds}/{name}/{split}/{key}")
                record[split + "_selection_export_agreement"] = True
            x = torch.stack([eval_data[i][0] for i in range(2)]).repeat(64, 1, 1, 1).cuda()
            torch.cuda.reset_peak_memory_stats()
            model.eval()
            with torch.inference_mode():
                pred = model(x)
                if pred.shape != (128, 1, 224, 224) or not torch.isfinite(pred).all():
                    raise ValueError(f"Invalid full-batch FP32 inference: {ds}/{name}")
            record.update(peak_inference_MiB=torch.cuda.max_memory_allocated()/1024**2,
                          seconds=time.perf_counter() - started)
            report.append(record)
            write_json(OUT / "gpu_checks_partial.json", {"diagnostic_only": True, "checks": report})
            print(f"GPU + selection/export agreement OK: {ds}/{name}", flush=True)
            del x, pred
        del model
        torch.cuda.empty_cache()
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-smoke", action="store_true")
    args = parser.parse_args()
    initial_hashes = source_hashes(ROOT)
    settings = protocol(ROOT)
    prepared = verify_prepared(ROOT, check_pixels=True)
    rows, scans = manifests()
    checks = validate_transforms(rows, scans)
    gpu = validate_gpu(rows, scans, settings) if args.gpu_smoke else []
    hashes = source_hashes(ROOT)
    if hashes != initial_hashes:
        raise ValueError("Source files changed during diagnostics; rerun with the final code.")
    for path in (OLD_SPLITS, OLD_SCANS, "scripts/04_preflight.py"):
        hashes[path] = digest(ROOT / path)
    result = {"status": "diagnostics_passed", "training_queue_started": False,
              "metric_contract": METRIC_SPEC, "source_hashes": hashes, "prepared_data": prepared,
              "datasets": DATASETS, "protocol": settings, "transform_checks": checks, "gpu_checks": gpu,
              "counts_all_six": {"single": 810, "trivialaugment": 702, "randaugment": 648, "total": 2160},
              "counts_per_dataset": {"single": 135, "trivialaugment": 117, "randaugment": 108, "total": 360},
              "free_GiB": shutil.disk_usage(ROOT).free/1024**3,
              "versions": {name: importlib.metadata.version(name) for name in
                           ("torch", "numpy", "albumentations", "monai", "torchmetrics", "segmentation-models-pytorch")}}
    write_json(OUT / "preflight.json", result)
    print(json.dumps({k: result[k] for k in ("status", "counts_all_six", "free_GiB")}, indent=2), flush=True)

if __name__ == "__main__":
    main()
