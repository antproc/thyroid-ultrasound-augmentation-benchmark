from __future__ import annotations

import torch

from thyroid_augmentation.training.metrics import METRIC_PROTOCOL, per_image_metrics


def test_per_image_dice_thresholds_probabilities_at_half() -> None:
    logits = torch.full((1, 1, 4, 4), 0.2)
    masks = torch.ones((1, 1, 4, 4))
    result = per_image_metrics(logits, masks)
    assert float(result["dice"]) == 1.0
    logits = torch.full((1, 1, 4, 4), -0.2)
    result = per_image_metrics(logits, masks)
    assert float(result["dice"]) < 1e-6


def test_empty_prediction_and_empty_mask_score_one() -> None:
    logits = torch.full((2, 1, 4, 4), -3.0)
    masks = torch.zeros((2, 1, 4, 4))
    result = per_image_metrics(logits, masks)
    assert torch.allclose(result["dice"], torch.ones(2, dtype=result["dice"].dtype))
    assert METRIC_PROTOCOL == "per_image_sigmoid05_v2"
