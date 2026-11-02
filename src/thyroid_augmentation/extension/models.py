from __future__ import annotations

from pathlib import Path

import yaml

def build_model(root: Path, name: str, pretrained: bool = False):
    import segmentation_models_pytorch as smp

    if smp.__version__ != "0.5.0":
        raise RuntimeError(f"Published model definitions require SMP 0.5.0, found {smp.__version__}")
    if name not in {"res34_unet", "deeplabv3plus_resnet50", "mit_b2_unet"}:
        raise ValueError(f"Unknown extension model: {name}")
    path = root / "configs/models" / f"{name}.yaml"
    settings = yaml.safe_load(path.read_text())
    if settings.pop("name") != name:
        raise ValueError("Model configuration mismatch.")
    architecture = settings.pop("architecture")
    settings["encoder_name"] = settings.pop("encoder")
    settings["encoder_weights"] = "imagenet" if pretrained else None

    return getattr(smp, architecture)(**settings)
