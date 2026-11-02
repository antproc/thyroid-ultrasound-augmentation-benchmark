from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

from thyroid_augmentation.extension.common import PREPARED, PROTOCOL, SCANS, SPLIT_RECORD, SPLITS, digest, protocol, read_csv, write_csv, write_json

AUGMENTATIONS = ["none", "bilateral_filter", "brightness", "contrast", "depth_attenuation", "gamma",
                 "gaussian_noise", "gaussian_shadow", "haze_artifact", "horizontal_flip", "vertical_flip",
                 "random_crop", "rotate", "translate", "zoom"]
NEW_DATASETS = ("PKTN", "TNUI-2021", "ThyroidXL")

def source_hashes(root: Path) -> dict:
    files = [root / SPLITS, root / SCANS, root / PREPARED,
             root / "manifests/pktn_tnui2021_thyroidxl/preparation_issues.json",
             root / "manifests/pktn_tnui2021_thyroidxl/preparation_audit.csv",
             root / PROTOCOL, root / SPLIT_RECORD,
             root / "configs/experiments/export_contract.json",
             root / "scripts/01_prepare_pktn_tnui2021_thyroidxl.py"]
    for folder in (root / "src/thyroid_augmentation/extension", root / "src/thyroid_augmentation/training",
                   root / "src/thyroid_augmentation/augmentations"):
        files.extend(sorted(folder.glob("*.py")))
    files.extend(sorted((root / "configs/models").glob("*.yaml")))
    files.extend(sorted((root / "external/ultrasound-augmentation/usaugment-experiments/src/usaugment").rglob("*.py")))
    return {str(f.relative_to(root)): digest(f) for f in files}

def verify_prepared(root: Path, check_pixels=True) -> dict:
    import numpy as np
    from PIL import Image

    issues = json.loads((root / "manifests/pktn_tnui2021_thyroidxl/preparation_issues.json").read_text())
    if issues["status"] != "passed" or issues["issues"]:
        raise ValueError("Unresolved image/mask preparation issues; training is blocked.")
    info = json.loads((root / PREPARED).read_text())
    if digest(root / SPLITS) != info["manifest_sha256"] or digest(root / SCANS) != info["scan_manifest_sha256"]:
        raise ValueError("Prepared manifests changed.")
    audit_path = root / "manifests/pktn_tnui2021_thyroidxl/preparation_audit.csv"
    if digest(audit_path) != info["audit_sha256"]:
        raise ValueError("Preparation audit changed.")
    raw = read_csv(root / "manifests/pktn_tnui2021_thyroidxl/splits_raw.csv")
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
    return {"samples": len(rows), "split_counts": dict(Counter(f"{r['dataset']}/{r['split']}" for r in rows))}

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
        raise RuntimeError("Less than 160 GiB free before preflight.")
    paths = ProjectPaths(root, root / "data")
    scans = _load_scan_mask_map(paths, scan_manifest)
    augmentation_checks = []
    samples = {ds: [r for r in rows if r["dataset"] == ds and r["split"] == "train"][:2] for ds in NEW_DATASETS}
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
                path = root / "results/prepare/diagnostics" / ds / name / f"{split}_untrained.csv"
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
    report = check_runtime(root, read_csv(root / "manifests/pktn_tnui2021_thyroidxl/splits_diagnostic.csv"),
                           "manifests/pktn_tnui2021_thyroidxl/scans_diagnostic.csv")
    report.update(status="diagnostic_only", training_started=False)
    write_json(root / "results/prepare/runtime_smoke.json", report)
    return report

def preflight(root: Path) -> dict:
    report = verify_prepared(root)
    report.update(check_runtime(root, read_csv(root / SPLITS), SCANS))
    report.update(status="ready", source_hashes=source_hashes(root))
    write_json(root / "results/prepare/preflight.json", report)
    return report
