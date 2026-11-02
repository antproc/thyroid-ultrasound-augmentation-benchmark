from __future__ import annotations

from thyroid_augmentation.models.base import build_smp_or_fallback

def build_model(
    in_channels: int = 3,
    classes: int = 1,
    encoder_weights: str | None = None,
    dropout: float = 0.0,
    pretrained: bool = False,
):
    return build_smp_or_fallback(
        "Unet",
        "resnet34",
        in_channels=in_channels,
        classes=classes,
        encoder_weights=encoder_weights,
    )
