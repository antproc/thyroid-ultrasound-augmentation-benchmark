from __future__ import annotations

import json
from pathlib import Path

import torch

METRIC_PROTOCOL = "per_image_sigmoid05_v2"
METRIC_SPEC = {
    "protocol": METRIC_PROTOCOL,
    "prediction": "sigmoid(logits) > 0.5",
    "target": "mask > 0.5",
    "aggregation": "arithmetic mean over images, not pixels or batch means",
    "ignore_index": None,
    "epsilon": 1e-7,
    "empty_prediction_and_target_score": 1.0,
}

def per_image_metrics(logits: torch.Tensor, masks: torch.Tensor) -> dict[str, torch.Tensor]:
    if logits.ndim != 4 or logits.shape != masks.shape or logits.shape[1] != 1 or logits.shape[0] == 0:
        raise ValueError("Metrics require matching nonempty (N,1,H,W) tensors.")
    if not logits.is_floating_point() or not torch.isfinite(logits).all() or not torch.isfinite(masks).all():
        raise ValueError("Nonfinite/invalid metric inputs.")
    if not torch.all((masks == 0) | (masks == 1)):
        raise ValueError("Expected prepared binary masks in {0,1}.")
    pred, truth = logits.sigmoid() > .5, masks > .5
    dims = (1, 2, 3)
    tp = (pred & truth).sum(dims)
    fp = (pred & ~truth).sum(dims)
    fn = (~pred & truth).sum(dims)
    tn = (~pred & ~truth).sum(dims)

    p, f, n = tp.double(), fp.double(), fn.double()
    eps = METRIC_SPEC["epsilon"]
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "dice": (2*p + eps)/(2*p + f + n + eps), "iou": (p + eps)/(p + f + n + eps)}

def guard_run_namespace(run_dir: Path, checkpoint_dir: Path) -> None:
    from thyroid_augmentation.extension.common import write_json

    marker = run_dir / "metric_protocol.json"
    if marker.exists():
        if json.loads(marker.read_text()) != METRIC_SPEC:
            raise ValueError("Metric contract differs; choose a new experiment namespace.")
        summary = run_dir / "summary.json"
        if summary.exists():
            require_protocol(json.loads(summary.read_text()), str(summary))
        return
    artifacts = [run_dir / "train_log.csv", run_dir / "summary.json", checkpoint_dir / "best.pt", checkpoint_dir / "last.pt"]
    if any(p.exists() for p in artifacts):
        raise ValueError("Run directory already holds results without a metric protocol marker; use a new phase/output directory.")
    write_json(marker, METRIC_SPEC)

def require_protocol(value: dict, source: str) -> None:
    if value.get("metric_protocol") != METRIC_PROTOCOL:
        raise ValueError(f"Incompatible metric protocol in {source}.")
