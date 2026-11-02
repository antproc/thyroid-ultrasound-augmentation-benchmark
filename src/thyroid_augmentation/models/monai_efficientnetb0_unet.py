from __future__ import annotations

def build_model(
    in_channels: int = 3,
    classes: int = 1,
    encoder_weights: str | None = None,
    dropout: float = 0.454402,
    pretrained: bool = True,
):
    from thyroid_augmentation.compat.monai_py312 import ensure_monai_py312_compat

    ensure_monai_py312_compat()
    from monai.networks.nets import FlexibleUNet

    if encoder_weights is not None:
        pretrained = encoder_weights == "imagenet"
    return FlexibleUNet(
        in_channels=in_channels,
        out_channels=classes,
        backbone="efficientnet-b0",
        pretrained=pretrained,
        dropout=dropout,
    )
