from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import shutil
from pathlib import Path

from thyroid_augmentation.extension.common import SPLITS, digest, read_csv, write_csv, write_json
from thyroid_augmentation.training.metrics import METRIC_PROTOCOL, per_image_metrics, require_protocol

def run_id(config: dict) -> str:
    keys = ("phase", "dataset", "model", "augmentation", "seed")
    return hashlib.sha256(json.dumps({k: config[k] for k in keys}, sort_keys=True).encode()).hexdigest()[:16]

def pixel_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    epsilon = 1e-7
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "target_pixels": tp + fn, "predicted_pixels": tp + fp,
            "dice": (2 * tp + epsilon) / (2 * tp + fp + fn + epsilon),
            "iou": (tp + epsilon) / (tp + fp + fn + epsilon)}

def validate_export(rows: list[dict], expected: list[dict], config: dict, split: str) -> dict:
    if not expected:
        raise ValueError("An export cannot certify an empty evaluation split.")
    by_id = {row["sample_id"]: row for row in expected}
    if len(rows) != len(expected) or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Export has missing or duplicated images.")
    if {r["sample_id"] for r in rows} != set(by_id):
        raise ValueError("Export image IDs differ from the frozen split.")
    for row in rows:
        source = by_id[row["sample_id"]]
        for name in ("dataset", "group_id", "group_unit", "case_id", "clip_id"):
            if str(row[name]) != str(source[name]):
                raise ValueError(f"Export ID mismatch: {name}")
        if row["split"] != split or int(row["seed"]) != config["seed"] or row["model"] != config["model"]:
            raise ValueError("Export run identity differs.")
        if row["run_id"] != run_id(config):
            raise ValueError("Export belongs to a different run.")
        if row["configuration_id"] != config["augmentation"]:
            raise ValueError("Export augmentation differs.")
        if (int(row["evaluated_height"]), int(row["evaluated_width"])) != (config["image_size"], config["image_size"]):
            raise ValueError("Export evaluation size differs from the protocol.")
        counts = [int(row[c]) for c in ("tp", "fp", "fn", "tn")]
        if min(counts) < 0 or sum(counts) != int(row["evaluated_height"]) * int(row["evaluated_width"]):
            raise ValueError("Pixel counts inconsistent with image dimensions.")
        calculated = pixel_metrics(*counts)
        for col in ("dice", "iou"):
            value = float(row[col])
            if not math.isfinite(value) or not 0 <= value <= 1 or not math.isclose(value, calculated[col], abs_tol=1e-12, rel_tol=0):
                raise ValueError(f"Invalid exported {col}")
        for col in ("target_pixels", "predicted_pixels"):
            if int(row[col]) != calculated[col]:
                raise ValueError(f"Inconsistent {col}")
    return {"samples": len(rows), "dice": sum(float(r["dice"]) for r in rows) / len(rows),
            "iou": sum(float(r["iou"]) for r in rows) / len(rows)}

def infer_rows(root: Path, config: dict, model, sources: list[dict], split: str, batch_size: int) -> list[dict]:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from thyroid_augmentation.training.runner import SegmentationManifestDataset

    if not sources:
        raise ValueError(f"Empty {split} split")
    device = next(model.parameters()).device
    model.eval()
    dataset = SegmentationManifestDataset(sources, root, config["image_size"], training=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    output, offset = [], 0
    with torch.inference_mode():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            if not torch.isfinite(logits).all():
                raise ValueError("Non-finite inference logits; export not marked complete.")
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            metrics = per_image_metrics(logits, masks)
            values = torch.stack([metrics[k] for k in ("tp", "fp", "fn", "tn")], dim=1).cpu().tolist()
            for source, counts in zip(sources[offset:offset + len(values)], values, strict=True):
                output.append({"run_id": run_id(config), "dataset": config["dataset"], "split": split,
                               "model": config["model"], "strategy": (
                                   "baseline" if config["augmentation"] in ("none", "identity") else
                                   "trivialaugment" if config["augmentation"].startswith("trivial:") else
                                   "randaugment" if config["augmentation"].startswith("randaugment:") else "single"),
                               "configuration_id": config["augmentation"], "seed": config["seed"],
                               **{k: source[k] for k in ("sample_id", "group_id", "group_unit", "case_id", "clip_id")},
                               **pixel_metrics(*counts), "evaluated_height": int(masks.shape[-2]), "evaluated_width": int(masks.shape[-1])})
            offset += len(values)
    validate_export(output, sources, config, split)
    return output

def verify_backup(out: Path, backup: Path, receipt: dict) -> None:

    backup.mkdir(parents=True, exist_ok=True)
    for name, sha in receipt["files"].items():
        if digest(out / name) != sha:
            raise ValueError(f"Changed canonical export: {name}")
        if not (backup / name).exists():
            tmp = backup / (name + ".tmp")
            shutil.copyfile(out / name, tmp)
            if digest(tmp) != sha:
                raise ValueError(f"Backup checksum mismatch: {name}")
            tmp.replace(backup / name)
        if digest(backup / name) != sha:
            raise ValueError(f"Changed export backup: {name}")
    if not (backup / "complete.json").exists():
        write_json(backup / "complete.json", receipt)
    if digest(out / "complete.json") != digest(backup / "complete.json"):
        raise ValueError("Backup receipt mismatch")

def export_best(root: Path, config: dict, summary: dict, batch_size: int = 128,
                *, output_root: Path | None = None, source_fingerprint: dict | None = None) -> dict:
    import torch
    from thyroid_augmentation.extension.models import build_model

    checkpoint_path = Path(summary["checkpoint"])
    identity = run_id(config)
    out = (output_root / "per_image" if output_root else root / "results/prepare/per_image") / identity
    backup = (output_root / "backup/per_image" if output_root else root / "results/prepare_backup/per_image") / identity
    manifest = root / config.get("split_manifest", SPLITS)
    expected = {split: [r for r in read_csv(manifest) if r["dataset"] == config["dataset"] and r["split"] == split]
                for split in ("val", "test")}
    hashes = {"checkpoint_sha256": digest(checkpoint_path), "manifest_sha256": digest(manifest),
              "exporter_sha256": digest(Path(__file__)), "log_sha256": digest(Path(summary["log"])),
              "summary_sha256": digest(Path(summary["log"]).with_name("summary.json"))}
    from thyroid_augmentation.extension.experiment import source_hashes
    fingerprint = source_hashes(root) if source_fingerprint is None else source_fingerprint
    hashes["pipeline_sha256"] = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    receipt_path = out / "complete.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["source_hashes"] != hashes:
            raise ValueError(f"Completed export source changed; refusing silent overwrite: {out}")
        for split in expected:
            path = out / f"{split}.csv"
            if digest(path) != receipt["files"][path.name]:
                raise ValueError(f"Export checksum mismatch: {path}")
            validate_export(read_csv(path), expected[split], config, split)
        verify_backup(out, backup, receipt)
        return receipt
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing silent CPU inference.")
    device = torch.device("cuda")
    model = build_model(root, config["model"], pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if config.get("metric_protocol") == METRIC_PROTOCOL:
        require_protocol(checkpoint, str(checkpoint_path))
        require_protocol(summary, "export summary")
    for key in ("dataset", "model", "augmentation", "seed", "phase"):
        if checkpoint["config"][key] != config[key]:
            raise ValueError(f"Wrong evaluation checkpoint: {key}")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    aggregates = {}
    for split, sources in expected.items():
        output = infer_rows(root, config, model, sources, split, batch_size)
        aggregates[split] = validate_export(output, sources, config, split)
        write_csv(out / f"{split}.csv", output)
        validate_export(read_csv(out / f"{split}.csv"), sources, config, split)
    metadata = {"run_id": identity, "config": checkpoint["config"], "source_hashes": hashes,
                "selected_epoch": int(checkpoint["epoch"]) + 1, "training_selection_dice": summary["best_dice"],
                "metric_definition": "sigmoid(logits)>0.5; per-image Dice/IoU from integer pixel counts; epsilon=1e-7",
                "validation_selection_metric": checkpoint.get("metric_protocol", "legacy_raw_logits_dice"),
                "evaluation_preprocessing": "build_article_eval_transform: aspect-preserving resize to 224, zero padding, normalization 0..1, binary mask",
                "inference_batch_size": batch_size, "inference_precision": "float32", "aggregates": aggregates,
                "versions": {p: importlib.metadata.version(p) for p in ("torch", "numpy", "albumentations", "segmentation-models-pytorch")}}
    write_json(out / "metadata.json", metadata)
    shutil.copyfile(summary["log"], out / "train_log.csv")
    write_json(out / "training_summary.json", summary)
    files = {name: digest(out / name) for name in ("val.csv", "test.csv", "metadata.json", "train_log.csv", "training_summary.json")}
    backup.mkdir(parents=True, exist_ok=True)
    for name, sha in files.items():
        shutil.copyfile(out / name, backup / name)
        if digest(backup / name) != sha:
            raise ValueError(f"Backup verification failed: {name}")
    receipt = {"status": "complete_exports_and_local_backup_verified", "source_hashes": hashes,
               "files": files, "aggregates": aggregates,
               "backup": str(backup.relative_to(root)), "backup_on_same_disk": True}
    write_json(out / "complete.json", receipt)
    verify_backup(out, backup, receipt)
    del model, checkpoint
    torch.cuda.empty_cache()
    return receipt
