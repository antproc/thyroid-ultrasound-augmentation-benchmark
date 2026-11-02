from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import cv2
import numpy as np

UPSTREAM_ORDER = [
    "none",
    "bilateral_filter",
    "brightness",
    "contrast",
    "depth_attenuation",
    "gamma",
    "gaussian_noise",
    "gaussian_shadow",
    "haze_artifact",
    "horizontal_flip",
    "vertical_flip",
    "random_crop",
    "rotate",
    "translate",
    "zoom",
]

ARTICLE_INDIVIDUAL_AUGMENTATIONS = [
    "bilateral_filter",
    "brightness",
    "contrast",
    "depth_attenuation",
    "flip_horizontal",
    "flip_vertical",
    "gamma",
    "gaussian_noise",
    "gaussian_shadow",
    "haze_artifact",
    "identity",
    "random_crop",
    "rotate",
    "translate",
    "zoom",
]

ARTICLE_STANFORD_THYROID_TRIVIAL_ORDER = [
    "identity",
    "rotate",
    "brightness",
    "flip_horizontal",
    "translate",
    "random_crop",
    "depth_attenuation",
    "gamma",
    "zoom",
    "contrast",
    "flip_vertical",
    "gaussian_shadow",
    "haze_artifact",
    "bilateral_filter",
    "gaussian_noise",
]

ARTICLE_AUTOAUGMENT_ORDER = ARTICLE_STANFORD_THYROID_TRIVIAL_ORDER
ARTICLE_RANDAUGMENT_ORDER = [name for name in ARTICLE_STANFORD_THYROID_TRIVIAL_ORDER if name != "identity"]

DISPLAY_NAMES = {
    "none": "None",
    "bilateral_filter": "Bilateral filter",
    "brightness": "Brightness",
    "contrast": "Contrast",
    "depth_attenuation": "Depth attenuation",
    "gamma": "Gamma",
    "gaussian_noise": "Gaussian noise",
    "gaussian_shadow": "Gaussian shadow",
    "haze_artifact": "Haze artifact",
    "horizontal_flip": "Horizontal flip",
    "vertical_flip": "Vertical flip",
    "random_crop": "Random crop",
    "rotate": "Rotate",
    "translate": "Translate",
    "zoom": "Zoom",
}

@dataclass(frozen=True)
class _AugmentItem:
    name: str
    level: int | None = None

@dataclass(frozen=True)
class _RandAugmentConfig:
    operations: tuple[str, ...]
    num_ops: int = 2
    magnitude: int = 9
    magnitude_denominator: int = 30

def normalize_augmentation_name(name: str | None) -> str:
    if not name:
        return "none"
    cleaned = name.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "no_augmentation": "none",
        "identity": "none",
        "baseline": "none",
        "horizontal": "horizontal_flip",
        "vertical": "vertical_flip",
        "flip_horizontal": "horizontal_flip",
        "flip_vertical": "vertical_flip",
        "rotation": "rotate",
        "translation": "translate",
        "speckle": "bilateral_filter",
        "speckle_reduction": "bilateral_filter",
    }
    return aliases.get(cleaned, cleaned)

def split_augmentation_policy(name: str | None) -> list[str]:
    normalized = normalize_augmentation_name(name)
    return [normalize_augmentation_name(part) for part in normalized.split("+") if part.strip()]

def is_autoaugment_policy(name: str | None) -> bool:
    return normalize_augmentation_name(name).startswith("autoaugment")

def is_randaugment_policy(name: str | None) -> bool:
    return normalize_augmentation_name(name).startswith("randaugment")

def build_train_transform(augmentation: str | None, project_root: Path, image_size: int):
    names = [name for name in split_augmentation_policy(augmentation) if name != "none"]
    if not names:
        return None

    transforms = [_make_transform(name, project_root, image_size) for name in names]
    return A.Compose(transforms, additional_targets={"scan_mask": "mask"})

def build_article_train_transform(augmentation: str | None, project_root: Path, image_size: int = 224):
    names = [name for name in split_augmentation_policy(augmentation) if name != "none"]
    transforms = [
        A.LongestMaxSize(max_size=256, interpolation=1),
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        A.Normalize(mean=0.0, std=1.0),
    ]
    transforms.extend(_make_transform(name, project_root, image_size) for name in names)
    transforms.extend(
        [
            A.LongestMaxSize(max_size=image_size, interpolation=1),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
            ),
        ]
    )
    return A.Compose(transforms, additional_targets={"scan_mask": "mask"})

def build_article_trivial_augment_transform(
    top_n: int,
    project_root: Path,
    image_size: int = 224,
    augmentation: str | None = None,
):
    if top_n < 2 or top_n > 14:
        raise ValueError("Article TrivialAugment top_n must be in the range 2..14.")
    order = _parse_trivial_order(augmentation) or ARTICLE_STANFORD_THYROID_TRIVIAL_ORDER
    selected = order[: top_n + 1]
    article_transforms = [_make_trivial_transform(name, project_root, image_size) for name in selected]
    transforms = [
        A.LongestMaxSize(max_size=256, interpolation=1),
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        A.Normalize(mean=0.0, std=1.0),
        A.Compose(transforms=[A.OneOf(transforms=article_transforms, p=1.0) for _ in range(2)]),
        A.LongestMaxSize(max_size=image_size, interpolation=1),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
        ),
    ]
    return A.Compose(transforms, additional_targets={"scan_mask": "mask"})

def build_article_autoaugment_transform(
    project_root: Path,
    image_size: int = 224,
    augmentation: str | None = None,
):
    subpolicy_names = _parse_autoaugment_subpolicies(augmentation) or _make_autoaugment_subpolicy_names(
        ARTICLE_AUTOAUGMENT_ORDER
    )
    subpolicies = [
        A.Sequential(
            [_make_autoaugment_transform(item, project_root, image_size) for item in subpolicy],
            p=1.0,
        )
        for subpolicy in subpolicy_names
    ]
    transforms = [
        A.LongestMaxSize(max_size=256, interpolation=1),
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        A.Normalize(mean=0.0, std=1.0),
        A.OneOf(transforms=subpolicies, p=1.0),
        A.LongestMaxSize(max_size=image_size, interpolation=1),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
        ),
    ]
    return A.Compose(transforms, additional_targets={"scan_mask": "mask"})

def build_article_randaugment_transform(
    project_root: Path,
    image_size: int = 224,
    augmentation: str | None = None,
    replay: bool = False,
):
    config = _parse_randaugment_config(augmentation)
    magnitude_scale = config.magnitude / max(1, config.magnitude_denominator)
    transforms = [
        A.LongestMaxSize(max_size=256, interpolation=1),
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        A.Normalize(mean=0.0, std=1.0),
    ]
    transforms.extend(
        item
        for _ in range(config.num_ops)
        for item in (
            A.OneOf(
                transforms=[
                    _make_randaugment_transform(name, project_root, image_size, magnitude_scale)
                    for name in config.operations
                ],
                p=1.0,
            ),
            A.Lambda(image=_sanitize_image_values, p=1.0),
        )
    )
    transforms.extend(
        [
            A.LongestMaxSize(max_size=image_size, interpolation=1),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
            ),
        ]
    )
    compose_cls = A.ReplayCompose if replay else A.Compose
    return compose_cls(transforms, additional_targets={"scan_mask": "mask"})

def _parse_trivial_order(augmentation: str | None) -> list[str] | None:
    if not augmentation:
        return None
    value = augmentation.strip()
    if not value.lower().startswith("trivial:"):
        return None
    names = [part.strip() for part in value.split(":", 1)[1].split(",") if part.strip()]
    if not names:
        return None
    normalized = [normalize_augmentation_name(name) for name in names]
    if normalized[0] != "none":
        normalized.insert(0, "none")
    return ["identity" if name == "none" else name for name in normalized]

def _parse_autoaugment_subpolicies(augmentation: str | None) -> list[tuple[_AugmentItem, ...]] | None:
    if not augmentation:
        return None
    value = augmentation.strip()
    if value.lower() == "autoaugment":
        return None
    if not value.lower().startswith("autoaugment:"):
        return None
    payload = value.split(":", 1)[1]
    if "|" in payload:
        pairs = []
        for raw_subpolicy in payload.split("|"):
            sequence = _parse_autoaugment_sequence(raw_subpolicy)
            if sequence:
                pairs.append(sequence)
        return pairs or None
    if "+" in payload:
        sequence = _parse_autoaugment_sequence(payload)
        return [sequence] if sequence else None
    names = [part.strip() for part in payload.split(",") if part.strip()]
    if not names:
        return None
    normalized = [normalize_augmentation_name(name) for name in names]
    if normalized[0] != "none":
        normalized.insert(0, "none")
    return _make_autoaugment_subpolicy_names(["identity" if name == "none" else name for name in normalized])

def _parse_autoaugment_sequence(raw_sequence: str) -> tuple[_AugmentItem, ...]:
    names = [part.strip() for part in raw_sequence.split("+") if part.strip()]
    return tuple(_normalize_autoaugment_item(name) for name in names)

def _make_autoaugment_subpolicy_names(order: list[str]) -> list[tuple[_AugmentItem, ...]]:
    if not order:
        return [(_AugmentItem("identity"), _AugmentItem("identity"))]
    if len(order) == 1:
        return [(_AugmentItem(order[0]), _AugmentItem("identity"))]
    return [
        (_AugmentItem(order[index]), _AugmentItem(order[(index + 1) % len(order)]))
        for index in range(len(order))
    ]

def _normalize_autoaugment_item(name: str) -> _AugmentItem:
    if "@" in name:
        raw_name, raw_level = name.rsplit("@", 1)
        try:
            level = int(raw_level)
        except ValueError as exc:
            raise ValueError(f"AutoAugment level must be an integer 1..3: {name}") from exc
        if level < 1 or level > 3:
            raise ValueError(f"AutoAugment level must be in 1..3: {name}")
    else:
        raw_name = name
        level = None
    normalized = normalize_augmentation_name(raw_name)
    normalized = "identity" if normalized == "none" else normalized
    return _AugmentItem(normalized, None if normalized == "identity" else level)

def _parse_randaugment_config(augmentation: str | None) -> _RandAugmentConfig:
    operations = tuple(_normalize_randaugment_name(name) for name in ARTICLE_RANDAUGMENT_ORDER)
    if not augmentation:
        return _RandAugmentConfig(operations=operations)
    value = augmentation.strip()
    if value.lower() == "randaugment":
        return _RandAugmentConfig(operations=operations)
    if not value.lower().startswith("randaugment:"):
        return _RandAugmentConfig(operations=operations)
    options = {}
    for item in value.split(":", 1)[1].split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"RandAugment option must use key=value syntax: {item}")
        key, raw = item.split("=", 1)
        options[key.strip().lower()] = raw.strip()
    num_ops = int(options.get("n", options.get("num_ops", 2)))
    magnitude = int(options.get("m", options.get("magnitude", 9)))
    if "levels" in options or "max_magnitude" in options or "max_m" in options:
        magnitude_denominator = int(options.get("levels", options.get("max_magnitude", options.get("max_m", 3))))
    else:
        num_magnitude_bins = int(options.get("bins", options.get("num_magnitude_bins", 31)))
        magnitude_denominator = num_magnitude_bins - 1
    if "ops" in options:
        operations = tuple(
            _normalize_randaugment_name(name)
            for name in options["ops"].split("+")
            if name.strip()
        )
    if not operations:
        raise ValueError("RandAugment operation pool cannot be empty.")
    if num_ops < 1:
        raise ValueError("RandAugment N/num_ops must be >= 1.")
    if magnitude_denominator < 1:
        raise ValueError("RandAugment magnitude denominator must be >= 1.")
    if magnitude < 0 or magnitude > magnitude_denominator:
        raise ValueError(
            f"RandAugment M/magnitude must be in 0..{magnitude_denominator}, got {magnitude}."
        )
    return _RandAugmentConfig(
        operations=operations,
        num_ops=num_ops,
        magnitude=magnitude,
        magnitude_denominator=magnitude_denominator,
    )

def _normalize_randaugment_name(name: str) -> str:
    normalized = normalize_augmentation_name(name)
    return "identity" if normalized == "none" else normalized

def build_article_eval_transform(image_size: int = 224):
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size, interpolation=1),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
            ),
            A.Normalize(mean=0.0, std=1.0),
        ],
        additional_targets={"scan_mask": "mask"},
    )

def _make_transform(
    name: str,
    project_root: Path,
    image_size: int,
    level: int | None = None,
    magnitude_scale: float | None = None,
):
    scale = _resolve_magnitude_scale(level=level, magnitude_scale=magnitude_scale)
    if name == "horizontal_flip":
        return A.HorizontalFlip(p=0.5)
    if name == "vertical_flip":
        return A.VerticalFlip(p=0.5)
    if name == "rotate":
        return A.Rotate(limit=_symmetric_range(30.0, scale), border_mode=0, value=0, mask_value=0, p=0.5)
    if name == "translate":
        return A.ShiftScaleRotate(
            shift_limit=_symmetric_range(0.0625, scale),
            scale_limit=(0.0, 0.0),
            rotate_limit=(0, 0),
            interpolation=1,
            border_mode=0,
            value=0,
            mask_value=0,
            p=0.5,
        )
    if name == "zoom":
        return A.ShiftScaleRotate(
            shift_limit=(0.0, 0.0),
            scale_limit=_symmetric_range(0.1, scale),
            rotate_limit=(0, 0),
            interpolation=1,
            border_mode=0,
            value=0,
            mask_value=0,
            p=0.5,
        )
    if name == "random_crop":
        return A.RandomCrop(height=image_size, width=image_size, p=1.0)
    if name == "brightness":
        return A.RandomBrightnessContrast(
            brightness_limit=_symmetric_range(0.2, scale),
            contrast_limit=(0.0, 0.0),
            p=0.5,
        )
    if name == "contrast":
        return A.RandomBrightnessContrast(
            brightness_limit=(0.0, 0.0),
            contrast_limit=_symmetric_range(0.2, scale),
            p=0.5,
        )
    if name == "gamma":
        gamma_delta = 20.0 * scale
        return A.RandomGamma(gamma_limit=(100 - gamma_delta, 100 + gamma_delta), p=0.5)
    if name == "gaussian_noise":
        return A.GaussNoise(
            var_limit=0.0225 * scale,
            mean=0,
            per_channel=False,
            noise_scale_factor=1,
            p=0.5,
        )
    if name in {"depth_attenuation", "gaussian_shadow", "haze_artifact", "bilateral_filter"}:
        return _make_usaugment_transform(name, project_root, level=level, magnitude_scale=magnitude_scale)
    raise KeyError(f"Unknown augmentation '{name}'.")

def _make_trivial_transform(name: str, project_root: Path, image_size: int):
    if name == "identity":
        return A.NoOp(p=1.0)
    normalized = normalize_augmentation_name(name)
    if normalized == "none":
        return A.NoOp(p=1.0)
    transform = _make_transform(normalized, project_root, image_size)
    transform.p = 1.0
    return transform

def _make_autoaugment_transform(item: _AugmentItem, project_root: Path, image_size: int):
    if item.name == "identity":
        return A.NoOp(p=1.0)
    transform = _make_transform(item.name, project_root, image_size, level=item.level)
    transform.p = 1.0
    return transform

def _make_randaugment_transform(name: str, project_root: Path, image_size: int, magnitude_scale: float):
    if name == "identity":
        return A.NoOp(p=1.0)
    transform = _make_transform(name, project_root, image_size, magnitude_scale=magnitude_scale)
    transform.p = 1.0
    return transform

def _resolve_magnitude_scale(level: int | None = None, magnitude_scale: float | None = None) -> float:
    if magnitude_scale is not None:
        if magnitude_scale < 0.0 or magnitude_scale > 1.0:
            raise ValueError(f"Magnitude scale must be in 0..1, got {magnitude_scale}.")
        return magnitude_scale
    if level is None:
        return 1.0
    if level < 1 or level > 3:
        raise ValueError(f"AutoAugment level must be in 1..3, got {level}.")
    return level / 3.0

def _symmetric_range(max_abs: float, scale: float) -> tuple[float, float]:
    value = max_abs * scale
    return (-value, value)

def _scaled_range(min_value: float, max_value: float, scale: float) -> tuple[float, float]:
    return (min_value * scale, max_value * scale)

def _scaled_upper_range(min_value: float, max_value: float, scale: float) -> tuple[float, float]:
    return (min_value, max(min_value, max_value * scale))

def _sanitize_image_values(image, **kwargs):
    return np.clip(np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

def _make_usaugment_transform(
    name: str,
    project_root: Path,
    level: int | None = None,
    magnitude_scale: float | None = None,
):
    scale = _resolve_magnitude_scale(level=level, magnitude_scale=magnitude_scale)
    usaugment_src = project_root / "external" / "ultrasound-augmentation" / "usaugment-experiments" / "src"
    if usaugment_src.exists() and str(usaugment_src) not in sys.path:
        sys.path.insert(0, str(usaugment_src))
    from usaugment.augmentation import BilateralFilter, DepthAttenuation, GaussianShadow, HazeArtifact

    if name == "depth_attenuation":
        return DepthAttenuation(attenuation_rate=(0.0, 3.0 * scale), max_attenuation=0.0, p=0.5)
    if name == "gaussian_shadow":
        return GaussianShadow(
            strength=_scaled_range(0.25, 0.8, scale),
            sigma_x=(0.01, 0.2),
            sigma_y=(0.01, 0.2),
            p=0.5,
        )
    if name == "haze_artifact":
        return HazeArtifact(radius=(0.05, 0.95), sigma=(0.0, 0.1 * scale), p=0.5)
    if name == "bilateral_filter":
        return BilateralFilter(
            sigma_spatial=_scaled_upper_range(0.05, 1.0, scale),
            sigma_color=_scaled_upper_range(0.05, 1.0, scale),
            window_size=5,
            p=0.5,
        )
    raise KeyError(name)
