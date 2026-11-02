from __future__ import annotations

from pathlib import Path

from thyroid_augmentation.augmentations.factory import build_article_trivial_augment_transform

ROOT = Path(__file__).resolve().parents[1]


def test_trivialaugment_uses_two_operations_and_includes_identity() -> None:
    transform = build_article_trivial_augment_transform(2, ROOT, augmentation="trivial:identity,rotate,brightness")
    operation_sequence = transform.transforms[3]
    assert len(operation_sequence.transforms) == 2
    for choice in operation_sequence.transforms:
        assert len(choice.transforms) == 3
        assert choice.transforms[0].__class__.__name__ == "NoOp"
