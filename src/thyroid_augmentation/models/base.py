from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

@dataclass(frozen=True)
class ModelSpec:
    name: str
    architecture: str
    encoder: str
    builder: Callable[..., "nn.Module"]
    description: str

def count_parameters(model: "nn.Module") -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))

class TinyFallbackUNet(nn.Module):

    def __init__(self, in_channels: int = 3, classes: int = 1, width: int = 16) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, classes, 1),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.decoder(self.encoder(x))

def build_smp_or_fallback(
    architecture: str,
    encoder_name: str,
    in_channels: int = 3,
    classes: int = 1,
    encoder_weights: str | None = None,
) -> "nn.Module":
    try:
        import segmentation_models_pytorch as smp

        model_cls = getattr(smp, architecture)
        return model_cls(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )
    except Exception:
        return TinyFallbackUNet(in_channels=in_channels, classes=classes)
