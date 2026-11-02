from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

NEW_PHASES = frozenset({
    "extension_single", "extension_trivialaugment", "extension_randaugment",
    "m2_single", "m2_ta", "m2_ra",
})
NEW_DATASETS = frozenset({"PKTN", "TNUI-2021", "ThyroidXL", "TUCC", "TN3K", "DDTI"})
IDENTITY_FIELDS = (
    "dataset", "model", "augmentation", "seed", "epochs", "scheduler_epochs",
    "image_size", "batch_size", "effective_batch_size", "learning_rate", "weight_decay",
    "dropout", "pretrained", "top_n", "split_manifest", "scan_mask_manifest", "metric_protocol",
)

def uses_verified_last_cleanup(config: dict) -> bool:
    return config.get("phase") in NEW_PHASES

def _hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

def _stamp(path: Path) -> tuple:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

def _contained(path: Path, allowed: Path) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(allowed) or path.is_symlink():
        raise ValueError(f"Retention path escapes the new experiment: {path}")
    return resolved

def _write_receipt(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)

def finalize_completed_run(root: Path, run_dir: Path, config: dict, model=None) -> dict:
    import torch

    if not uses_verified_last_cleanup(config) or config.get("dataset") not in NEW_DATASETS:
        raise ValueError("Cleanup is restricted to the new segmentation experiment.")
    if config.get("epochs") != 50 or config.get("seed") not in (0, 1, 2):
        raise ValueError("New experiment requires 50 epochs and seed 0, 1 or 2.")
    root = root.resolve(strict=True)
    allowed_runs = root / "runs" / config["phase"]
    allowed_models = root / "checkpoints" / config["phase"]
    run_dir = _contained(run_dir, allowed_runs)
    summary_path = _contained(run_dir / "summary.json", allowed_runs)
    log_path = _contained(run_dir / "train_log.csv", allowed_runs)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best_path = Path(summary["checkpoint"])
    if not best_path.is_absolute():
        best_path = root / best_path
    best_path = _contained(best_path, allowed_models)
    if best_path.name != "best.pt":
        raise ValueError("Expected a best.pt checkpoint.")
    last_path = best_path.with_name("last.pt")
    if last_path.is_symlink():
        raise ValueError("Refusing a linked last checkpoint.")
    had_last = last_path.exists()
    if had_last:
        _contained(last_path, allowed_models)
        if last_path.samefile(best_path):
            raise ValueError("best.pt and last.pt must be distinct files.")
    protected = [summary_path, log_path, best_path] + ([last_path] if had_last else [])
    before = {path: _stamp(path) for path in protected}
    for field in IDENTITY_FIELDS:
        if summary.get(field) != config.get(field):
            raise ValueError(f"Summary configuration differs: {field}")
    with log_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [int(row["epoch"]) for row in rows] != list(range(1, config["epochs"] + 1)):
        raise ValueError("Missing, duplicate or unordered training epochs; keeping last.pt.")
    for row in rows:
        for field in ("train_loss", "val_dice", "val_iou"):
            value = float(row[field])
            if not math.isfinite(value) or (field != "train_loss" and not 0 <= value <= 1):
                raise ValueError(f"Invalid {field} in epoch {row['epoch']}; keeping last.pt.")
    best_row = max(rows, key=lambda row: float(row["val_dice"]))
    best_dice = float(best_row["val_dice"])
    if not math.isclose(float(summary["best_dice"]), best_dice, rel_tol=0, abs_tol=1e-7):
        raise ValueError("Summary best Dice differs from the complete log.")
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    if config.get("metric_protocol") is not None:
        from thyroid_augmentation.training.metrics import require_protocol

        require_protocol(summary, str(summary_path))
        require_protocol(checkpoint, str(best_path))
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError("Best checkpoint does not contain a model state dictionary.")
    for field in (*IDENTITY_FIELDS, "phase"):
        if checkpoint.get("config", {}).get(field) != config.get(field):
            raise ValueError(f"Best checkpoint configuration differs: {field}")
    if int(checkpoint["epoch"]) + 1 != int(best_row["epoch"]):
        raise ValueError("Checkpoint is not from the validation-selected epoch.")
    if not math.isclose(float(checkpoint["best_dice"]), best_dice, rel_tol=0, abs_tol=1e-7):
        raise ValueError("Best checkpoint Dice differs from the complete log.")
    for name, tensor in checkpoint["model"].items():
        if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all().item():
            raise ValueError(f"Invalid model tensor: {name}")
    if model is None:
        from thyroid_augmentation.extension.models import build_model

        model = build_model(root, config["model"], pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    receipt = {
        "status": "verified_before_last_cleanup", "phase": config["phase"],
        "dataset": config["dataset"], "model": config["model"], "seed": config["seed"],
        "epochs": config["epochs"], "best_epoch": int(best_row["epoch"]),
        "best_checkpoint": str(best_path.relative_to(root)), "best_sha256": _hash(best_path),
        "summary_sha256": _hash(summary_path), "log_sha256": _hash(log_path),
        "last_existed": had_last, "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "per_image_exports_verified": False,
    }
    receipt_path = run_dir / "checkpoint_retention.json"
    if receipt_path.is_symlink() or receipt_path.with_suffix(".json.tmp").is_symlink():
        raise ValueError("Refusing a linked retention receipt.")

    _write_receipt(receipt_path, receipt)
    if any(_stamp(path) != before[path] for path in protected):
        raise RuntimeError("Run artifacts changed during verification; keeping last.pt.")
    if had_last:
        _contained(last_path, allowed_models)
        last_path.unlink()
    receipt["status"] = "complete_best_retained_last_absent"
    _write_receipt(receipt_path, receipt)
    return summary
