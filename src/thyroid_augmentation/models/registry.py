from __future__ import annotations

from collections.abc import Mapping

from thyroid_augmentation.models.base import ModelSpec
from thyroid_augmentation.models.deeplabv3plus_resnet50 import build_model as build_deeplabv3plus_resnet50
from thyroid_augmentation.models.mit_b2_unet import build_model as build_mit_b2_unet
from thyroid_augmentation.models.res34_unet import build_model as build_res34_unet

_MODEL_SPECS: dict[str, ModelSpec] = {
    "res34_unet": ModelSpec(
        name="res34_unet",
        architecture="U-Net",
        encoder="resnet34",
        builder=build_res34_unet,
        description="U-Net with ResNet-34 encoder.",
    ),
    "deeplabv3plus_resnet50": ModelSpec(
        name="deeplabv3plus_resnet50",
        architecture="DeepLabV3+",
        encoder="resnet50",
        builder=build_deeplabv3plus_resnet50,
        description="DeepLabV3+ with ResNet-50 encoder.",
    ),
    "mit_b2_unet": ModelSpec(
        name="mit_b2_unet",
        architecture="U-Net",
        encoder="mit_b2",
        builder=build_mit_b2_unet,
        description="U-Net decoder with transformer MiT-B2 encoder.",
    ),
}

def model_specs() -> Mapping[str, ModelSpec]:
    return dict(_MODEL_SPECS)

def available_models() -> list[str]:
    return sorted(_MODEL_SPECS)

def build_model(
    name: str,
    in_channels: int = 3,
    classes: int = 1,
    encoder_weights: str | None = None,
    dropout: float = 0.0,
    pretrained: bool = False,
):
    if name not in _MODEL_SPECS:
        available = ", ".join(available_models())
        raise KeyError(f"Unknown model '{name}'. Available models: {available}")
    return _MODEL_SPECS[name].builder(
        in_channels=in_channels,
        classes=classes,
        encoder_weights=encoder_weights,
        dropout=dropout,
        pretrained=pretrained,
    )
