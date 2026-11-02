from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from thyroid_augmentation.augmentations.factory import (
    build_article_autoaugment_transform,
    build_article_eval_transform,
    build_article_randaugment_transform,
    build_article_train_transform,
    build_article_trivial_augment_transform,
    is_autoaugment_policy,
    is_randaugment_policy,
    normalize_augmentation_name,
)
from thyroid_augmentation.compat.monai_py312 import ensure_monai_py312_compat
from thyroid_augmentation.models.registry import build_model
from thyroid_augmentation.paths import ProjectPaths
from thyroid_augmentation.training.checkpoint_retention import (
    finalize_completed_run,
    uses_verified_last_cleanup,
)
from thyroid_augmentation.training.metrics import (
    METRIC_PROTOCOL, guard_run_namespace, per_image_metrics, require_protocol,
)

ensure_monai_py312_compat()
from monai.losses import DiceLoss

@dataclass(frozen=True)
class TrainConfig:
    dataset: str
    model: str
    augmentation: str
    seed: int
    epochs: int
    scheduler_epochs: int | None
    batch_size: int
    effective_batch_size: int
    num_workers: int
    image_size: int
    learning_rate: float
    weight_decay: float
    dropout: float
    pretrained: bool
    top_n: int | None
    split_manifest: str
    scan_mask_manifest: str
    device: str
    resume: bool
    phase: str
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    source_datasets: tuple[str, ...] | None = None
    target_dataset: str | None = None
    metric_protocol: str = METRIC_PROTOCOL

class SegmentationManifestDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        project_root: Path,
        image_size: int,
        augmentation: str | None = None,
        scan_mask_map: dict[tuple[str, str], str] | None = None,
        training: bool = False,
        top_n: int | None = None,
    ) -> None:
        self.rows = rows
        self.project_root = project_root
        self.image_size = image_size
        self.augmentation = normalize_augmentation_name(augmentation)
        if training and is_randaugment_policy(augmentation):
            if top_n is not None:
                raise ValueError("RandAugment policies do not use --top-n.")
            self.transform = build_article_randaugment_transform(
                project_root,
                image_size,
                augmentation=augmentation,
            )
        elif training and is_autoaugment_policy(augmentation):
            if top_n is not None:
                raise ValueError("AutoAugment policies do not use --top-n.")
            self.transform = build_article_autoaugment_transform(
                project_root,
                image_size,
                augmentation=augmentation,
            )
        elif training and top_n is not None:
            self.transform = build_article_trivial_augment_transform(
                top_n,
                project_root,
                image_size,
                augmentation=augmentation,
            )
        elif training:
            self.transform = build_article_train_transform(self.augmentation, project_root, image_size)
        else:
            self.transform = build_article_eval_transform(image_size)
        self.scan_mask_map = scan_mask_map or {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        image_path = self.project_root / row["image_path"]
        mask_path = self.project_root / row["mask_path"]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_arr = np.asarray(image, dtype=np.uint8)
        with Image.open(mask_path) as mask:
            mask = mask.convert("L")
            mask_arr = np.asarray(mask, dtype=np.uint8)
        mask_arr = np.expand_dims(mask_arr, axis=-1)
        scan_mask_arr = self._load_scan_mask(row, image_arr.shape[:2])
        transformed = self.transform(image=image_arr, mask=mask_arr, scan_mask=scan_mask_arr)
        image_arr = transformed["image"].astype(np.float32)
        mask_arr = transformed["mask"]
        if mask_arr.ndim == 3:
            mask_arr = np.squeeze(mask_arr, axis=-1)
        mask_arr = (mask_arr.astype(np.float32) > 0).astype(np.float32)

        image_tensor = torch.from_numpy(image_arr).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)
        return image_tensor, mask_tensor

    def _load_scan_mask(self, row: dict[str, str], fallback_shape: tuple[int, int]) -> np.ndarray:
        key = (row["dataset"], row["sample_id"])
        rel_path = self.scan_mask_map.get(key)
        if rel_path:
            path = self.project_root / rel_path
            if path.exists():
                with Image.open(path) as scan_mask:
                    scan_mask = scan_mask.convert("L")
                    return (np.asarray(scan_mask, dtype=np.uint8) > 0).astype(np.uint8)
        return np.ones(fallback_shape, dtype=np.uint8)

def run_training(paths: ProjectPaths, config: TrainConfig) -> dict[str, object]:
    require_protocol(config.__dict__, "training configuration")
    normalized_augmentation = normalize_augmentation_name(config.augmentation)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = _resolve_device(config.device)
    run_dir = _run_dir(paths, config)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.csv"
    summary_path = run_dir / "summary.json"
    checkpoint_dir = (
        paths.project_root
        / "checkpoints"
        / config.phase
        / config.dataset
        / config.model
        / _safe_name(normalized_augmentation)
        / f"seed_{config.seed}"
    )
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    guard_run_namespace(run_dir, checkpoint_dir)
    log_has_invalid_values = _log_has_nonfinite_metrics(log_path)
    if config.resume and log_has_invalid_values:
        raise RuntimeError("Invalid saved training metrics; refusing automatic restart or cleanup.")
    if config.resume and summary_path.exists() and uses_verified_last_cleanup(config.__dict__):
        summary = finalize_completed_run(paths.project_root, run_dir, config.__dict__)
        print(f"Training already complete and verified. Summary: {summary_path}", flush=True)
        return summary
    if config.resume and summary_path.exists() and last_path.exists() and _log_has_epochs(log_path, config.epochs):
        print(f"Training already complete. Summary: {summary_path}", flush=True)
        return json.loads(summary_path.read_text(encoding="utf-8"))

    train_rows, val_rows = _load_rows(paths, config)
    if not train_rows or not val_rows:
        raise RuntimeError(f"Missing train/val rows for {config.dataset}. Run create-splits first.")

    train_sampler = _build_balanced_source_sampler(train_rows, config)
    train_loader = DataLoader(
        SegmentationManifestDataset(
            train_rows,
            paths.project_root,
            config.image_size,
            augmentation=normalized_augmentation,
            scan_mask_map=_load_scan_mask_map(paths, config.scan_mask_manifest),
            training=True,
            top_n=config.top_n,
        ),
        batch_size=config.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=config.num_workers > 0,
        prefetch_factor=4 if config.num_workers > 0 else None,
    )
    val_loader = _build_eval_loader(val_rows, paths, config, device)
    source_val_loaders = _build_source_val_loaders(val_rows, paths, config, device)

    if uses_verified_last_cleanup(config.__dict__):
        from thyroid_augmentation.extension.models import build_model as build_extension_model

        model = build_extension_model(paths.project_root, config.model, pretrained=config.pretrained).to(device)
    else:
        model = build_model(
            config.model,
            in_channels=3,
            classes=1,
            encoder_weights="imagenet" if config.pretrained else None,
            dropout=config.dropout,
            pretrained=config.pretrained,
        ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_function = DiceLoss(sigmoid=True, include_background=False)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    accumulation = max(1, math.ceil(config.effective_batch_size / config.batch_size))
    scheduler_epochs = config.scheduler_epochs or config.epochs
    total_optimizer_steps = max(1, scheduler_epochs * math.ceil(len(train_loader) / accumulation))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_optimizer_steps)

    start_epoch = 0
    best_dice = -1.0
    save_checkpoint = torch.save
    if uses_verified_last_cleanup(config.__dict__):
        from thyroid_augmentation.extension.checkpoint_io import restore_state, save_checkpoint as save_extension_checkpoint

        save_checkpoint = lambda payload, path: save_extension_checkpoint(payload, path, scaler)
        if config.resume and not last_path.exists() and (log_path.exists() or best_path.exists()):
            raise RuntimeError("Incomplete new run without last checkpoint; refusing to overwrite existing progress.")
    if config.resume and last_path.exists() and log_has_invalid_values:
        print(
            f"WARNING: Existing log contains non-finite train/validation values: {log_path}. "
            "Restarting this run from epoch 1 instead of resuming the checkpoint.",
            flush=True,
        )
    elif config.resume and last_path.exists():
        try:
            checkpoint = torch.load(last_path, map_location=device)
            require_protocol(checkpoint, str(last_path))
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            if "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            best_dice = float(checkpoint.get("best_dice", -1.0))
            if uses_verified_last_cleanup(config.__dict__):
                restore_state(checkpoint, config.__dict__, scaler)
        except (EOFError, OSError, RuntimeError, ValueError, KeyError) as exc:
            raise RuntimeError("Cannot resume checkpoint; keeping artifacts for inspection.") from exc

    _write_log_header(log_path, append=start_epoch > 0)
    total_start = time.perf_counter()
    max_cuda_memory_mb = 0 if device.type == "cuda" else ""
    for epoch in range(start_epoch, config.epochs):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_start = time.perf_counter()
        train_loss = _train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, loss_function, device, accumulation
        )
        val_metrics = _evaluate_validation(model, val_loader, source_val_loaders, device)
        if uses_verified_last_cleanup(config.__dict__) and not all(
            math.isfinite(value) for value in (train_loss, val_metrics["dice"], val_metrics["iou"])
        ):
            raise RuntimeError("Non-finite new-run metrics; keeping the previous checkpoints without cleanup.")
        epoch_seconds = time.perf_counter() - epoch_start
        max_cuda_memory_mb = (
            torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else ""
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "epoch_seconds": epoch_seconds,
            "images_per_second": len(train_rows) / epoch_seconds if epoch_seconds > 0 else 0,
            "max_cuda_memory_mb": max_cuda_memory_mb,
            "best_dice": best_dice,
            "val_selection_dice": val_metrics.get("selection_dice", val_metrics["dice"]),
            "val_selection_iou": val_metrics.get("selection_iou", val_metrics["iou"]),
            "val_combined_dice": val_metrics.get("combined_dice", ""),
            "val_combined_iou": val_metrics.get("combined_iou", ""),
            "val_TUCC_dice": val_metrics.get("TUCC_dice", ""),
            "val_TUCC_iou": val_metrics.get("TUCC_iou", ""),
            "val_TN3K_dice": val_metrics.get("TN3K_dice", ""),
            "val_TN3K_iou": val_metrics.get("TN3K_iou", ""),
            "val_DDTI_dice": val_metrics.get("DDTI_dice", ""),
            "val_DDTI_iou": val_metrics.get("DDTI_iou", ""),
        }
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            row["best_dice"] = best_dice
            save_checkpoint(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_dice": best_dice,
                    "config": config.__dict__,
                    "metric_protocol": METRIC_PROTOCOL,
                },
                best_path,
            )
        save_checkpoint(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_dice": best_dice,
                "config": config.__dict__,
                "metric_protocol": METRIC_PROTOCOL,
            },
            last_path,
        )
        _append_log(log_path, row)
        print(
            f"{config.dataset} {config.model} epoch {epoch + 1}/{config.epochs} "
            f"loss={train_loss:.5f} val_dice={val_metrics['dice']:.5f} "
            f"val_iou={val_metrics['iou']:.5f} seconds={epoch_seconds:.1f} "
            f"max_cuda_mb={max_cuda_memory_mb if max_cuda_memory_mb != '' else 'NA'}",
            flush=True,
        )

    total_seconds = time.perf_counter() - total_start
    summary = {
        "metric_protocol": METRIC_PROTOCOL,
        "dataset": config.dataset,
        "model": config.model,
        "augmentation": config.augmentation,
        "seed": config.seed,
        "epochs": config.epochs,
        "scheduler_epochs": scheduler_epochs,
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "train_source_samples": _source_counts(train_rows),
        "val_source_samples": _source_counts(val_rows),
        "balanced_source_sampling": bool(train_sampler is not None),
        "batch_size": config.batch_size,
        "effective_batch_size": config.effective_batch_size,
        "image_size": config.image_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "dropout": config.dropout,
        "pretrained": config.pretrained,
        "top_n": config.top_n,
        "source_datasets": list(config.source_datasets or (config.dataset,)),
        "target_dataset": config.target_dataset,
        "validation_selection": "macro_source_validation"
        if config.source_datasets and len(config.source_datasets) > 1
        else "validation",
        "split_manifest": config.split_manifest,
        "scan_mask_manifest": config.scan_mask_manifest,
        "device": str(device),
        "best_dice": best_dice,
        "total_seconds": total_seconds,
        "checkpoint": best_path.as_posix(),
        "log": log_path.as_posix(),
        "max_cuda_memory_mb_last_epoch": max_cuda_memory_mb,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if uses_verified_last_cleanup(config.__dict__):
        finalize_completed_run(paths.project_root, run_dir, config.__dict__, model=model)
    return summary

def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    loss_function: nn.Module,
    device: torch.device,
    accumulation: int,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_images = 0
    for step, (images, masks) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = loss_function(logits, masks) / accumulation
        scaler.scale(loss).backward()
        if step % accumulation == 0 or step == len(loader):
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        batch_size = images.shape[0]
        total_loss += float(loss.detach().cpu()) * accumulation * batch_size
        total_images += batch_size
    return total_loss / max(1, total_images)

@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    dice_total, iou_total, images_total = 0.0, 0.0, 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        metrics = per_image_metrics(logits, masks)
        dice_total += float(metrics["dice"].sum().cpu())
        iou_total += float(metrics["iou"].sum().cpu())
        images_total += masks.shape[0]
    if images_total == 0:
        raise ValueError("Cannot evaluate an empty dataset.")
    return {"dice": dice_total / images_total, "iou": iou_total / images_total}

def _evaluate_validation(
    model: nn.Module,
    val_loader: DataLoader,
    source_val_loaders: dict[str, DataLoader],
    device: torch.device,
) -> dict[str, float | str]:
    if not source_val_loaders:
        metrics = _evaluate(model, val_loader, device)
        return {
            "dice": metrics["dice"],
            "iou": metrics["iou"],
            "selection_dice": metrics["dice"],
            "selection_iou": metrics["iou"],
            "combined_dice": metrics["dice"],
            "combined_iou": metrics["iou"],
        }

    metrics_by_source = {
        dataset: _evaluate(model, loader, device)
        for dataset, loader in source_val_loaders.items()
    }
    dice_values = [metrics["dice"] for metrics in metrics_by_source.values()]
    iou_values = [metrics["iou"] for metrics in metrics_by_source.values()]
    macro_dice = float(np.mean(dice_values)) if dice_values else 0.0
    macro_iou = float(np.mean(iou_values)) if iou_values else 0.0
    payload: dict[str, float | str] = {
        "dice": macro_dice,
        "iou": macro_iou,
        "selection_dice": macro_dice,
        "selection_iou": macro_iou,
        "combined_dice": "",
        "combined_iou": "",
    }
    for dataset, metrics in metrics_by_source.items():
        payload[f"{dataset}_dice"] = metrics["dice"]
        payload[f"{dataset}_iou"] = metrics["iou"]
    return payload

def _load_rows(paths: ProjectPaths, config: TrainConfig) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    split_path = Path(config.split_manifest)
    if not split_path.is_absolute():
        split_path = paths.project_root / split_path
    rows = list(csv.DictReader(split_path.open("r", encoding="utf-8")))
    selected_datasets = set(config.source_datasets or (config.dataset,))
    rows = [row for row in rows if row["dataset"] in selected_datasets]
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    if config.max_train_samples:
        train_rows = train_rows[: config.max_train_samples]
    if config.max_val_samples:
        val_rows = val_rows[: config.max_val_samples]
    return train_rows, val_rows

def _build_eval_loader(
    rows: list[dict[str, str]],
    paths: ProjectPaths,
    config: TrainConfig,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        SegmentationManifestDataset(rows, paths.project_root, config.image_size, training=False),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
        prefetch_factor=4 if config.num_workers > 0 else None,
    )

def _build_source_val_loaders(
    val_rows: list[dict[str, str]],
    paths: ProjectPaths,
    config: TrainConfig,
    device: torch.device,
) -> dict[str, DataLoader]:
    if not config.source_datasets or len(config.source_datasets) <= 1:
        return {}
    loaders: dict[str, DataLoader] = {}
    for dataset in config.source_datasets:
        dataset_rows = [row for row in val_rows if row["dataset"] == dataset]
        if dataset_rows:
            loaders[dataset] = _build_eval_loader(dataset_rows, paths, config, device)
    return loaders

def _build_balanced_source_sampler(
    train_rows: list[dict[str, str]],
    config: TrainConfig,
) -> WeightedRandomSampler | None:
    if not config.source_datasets or len(config.source_datasets) <= 1:
        return None
    counts = Counter(row["dataset"] for row in train_rows)
    if len(counts) <= 1:
        return None
    weights = [1.0 / counts[row["dataset"]] for row in train_rows]
    return WeightedRandomSampler(weights, num_samples=len(train_rows), replacement=True)

def _source_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    return dict(sorted(Counter(row["dataset"] for row in rows).items()))

def _load_scan_mask_map(paths: ProjectPaths, scan_mask_manifest: str) -> dict[tuple[str, str], str]:
    scan_mask_path = Path(scan_mask_manifest)
    if not scan_mask_path.is_absolute():
        scan_mask_path = paths.project_root / scan_mask_path
    if not scan_mask_path.exists():
        return {}
    mapping: dict[tuple[str, str], str] = {}
    for row in csv.DictReader(scan_mask_path.open("r", encoding="utf-8")):
        if row.get("status") == "created":
            mapping[(row["dataset"], row["sample_id"])] = row["scan_mask_path"]
    return mapping

def _resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)

def _run_dir(paths: ProjectPaths, config: TrainConfig) -> Path:
    return (
        paths.project_root
        / "runs"
        / config.phase
        / config.dataset
        / config.model
        / _safe_name(normalize_augmentation_name(config.augmentation))
        / f"seed_{config.seed}"
    )

def _safe_name(value: str) -> str:
    safe = value.lower().replace(" ", "_").replace("+", "__")
    for char in [":", ",", "/", "\\", "|", "<", ">", "?", "*", '"']:
        safe = safe.replace(char, "_")
    if len(safe) > 48:
        digest = hashlib.sha1(safe.encode("utf-8")).hexdigest()[:12]
        safe = f"{safe[:35]}_{digest}"
    return safe

def _archive_unreadable_checkpoint(path: Path) -> Path:
    timestamp = int(time.time())
    archived_path = path.with_name(f"{path.stem}.unreadable_{timestamp}{path.suffix}")
    suffix = 1
    while archived_path.exists():
        archived_path = path.with_name(f"{path.stem}.unreadable_{timestamp}_{suffix}{path.suffix}")
        suffix += 1
    path.replace(archived_path)
    return archived_path

def _write_log_header(path: Path, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if append and path.exists():
        return
    if (not append) or (not path.exists()):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=_log_fieldnames(),
            )
            writer.writeheader()

def _log_has_epochs(path: Path, epochs: int) -> bool:
    if not path.exists():
        return False
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    if not rows:
        return False
    if not _log_rows_have_finite_metrics(rows):
        return False
    try:
        return max(int(float(row.get("epoch", 0) or 0)) for row in rows) >= epochs
    except ValueError:
        return False

def _log_has_nonfinite_metrics(path: Path) -> bool:
    if not path.exists():
        return False
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    return bool(rows) and not _log_rows_have_finite_metrics(rows)

def _log_rows_have_finite_metrics(rows: list[dict[str, str]]) -> bool:
    for row in rows:
        try:
            train_loss = float(row.get("train_loss", "nan") or "nan")
            val_dice = float(row.get("val_dice", "nan") or "nan")
        except ValueError:
            return False
        if not math.isfinite(train_loss) or not math.isfinite(val_dice):
            return False
    return True

def _append_log(path: Path, row: dict[str, object]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_log_fieldnames(), extrasaction="ignore")
        writer.writerow(row)

def _log_fieldnames() -> list[str]:
    return [
        "epoch",
        "train_loss",
        "val_dice",
        "val_iou",
        "epoch_seconds",
        "images_per_second",
        "max_cuda_memory_mb",
        "best_dice",
        "val_selection_dice",
        "val_selection_iou",
        "val_combined_dice",
        "val_combined_iou",
        "val_TUCC_dice",
        "val_TUCC_iou",
        "val_TN3K_dice",
        "val_TN3K_iou",
        "val_DDTI_dice",
        "val_DDTI_iou",
    ]
