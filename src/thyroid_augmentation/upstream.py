from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from thyroid_augmentation.paths import ProjectPaths, relative_to_root
from thyroid_augmentation.utils import run_git, write_csv, write_markdown

AUGMENTATION_FILES = {
    "Horizontal flip": "flip_horizontal.yaml",
    "Vertical flip": "flip_vertical.yaml",
    "Rotation": "rotate.yaml",
    "Zoom": "zoom.yaml",
    "Translation": "translate.yaml",
    "Random crop": "random_crop.yaml",
    "Brightness": "brightness.yaml",
    "Contrast": "contrast.yaml",
    "Gamma": "gamma.yaml",
    "Gaussian noise": "gaussian_noise.yaml",
    "Depth attenuation": "depth_attenuation.yaml",
    "Haze artifact": "haze_artifact.yaml",
    "Gaussian shadow": "gaussian_shadow.yaml",
    "Speckle reduction": "bilateral_filter.yaml",
    "Identity / NoOp": "identity.yaml",
    "TrivialAugment": "trivial_augment_stanford_thyroid_segmentation.yaml",
}

PARAMETER_COLUMNS = [
    "augmentation",
    "source_repository",
    "source_commit",
    "source_yaml_path",
    "yaml_key",
    "raw_yaml_value",
    "normalized_value",
    "effective_probability",
    "sampling_method",
    "applies_to_image",
    "applies_to_segmentation_mask",
    "requires_scan_mask",
    "requires_panel_processing",
    "image_interpolation",
    "mask_interpolation",
    "border_mode",
    "fill_value",
    "library_class",
    "compatibility_notes",
    "loaded_successfully",
]

@dataclass(frozen=True)
class UpstreamResult:
    rows: list[dict[str, object]]
    csv_path: Path
    report_path: Path
    compatibility_report_path: Path
    version_path: Path

def inspect_upstream(paths: ProjectPaths) -> UpstreamResult:
    paths.ensure_output_dirs()
    upstream_root = paths.project_root / "external" / "ultrasound-augmentation"
    experiments_root = upstream_root / "usaugment-experiments"
    augmentation_root = experiments_root / "src" / "usaugment" / "config" / "augmentation"
    task_yaml = experiments_root / "src" / "usaugment" / "config" / "task" / "stanford_thyroid_segmentation.yaml"

    if not upstream_root.exists():
        raise FileNotFoundError(f"Missing upstream repository: {upstream_root}")

    commit = run_git(["rev-parse", "HEAD"], upstream_root)
    submodule_status = run_git(["submodule", "status", "--recursive"], upstream_root)
    snapshot_dir = paths.project_root / "configs" / "upstream_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    copied: list[str] = []
    for augmentation, filename in AUGMENTATION_FILES.items():
        source = augmentation_root / filename
        if not source.exists():
            rows.append(_missing_row(augmentation, source, commit))
            continue
        relative = source.relative_to(upstream_root)
        destination = snapshot_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(relative.as_posix())
        data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        rows.extend(_rows_from_yaml(augmentation, data, source, upstream_root, commit))

    if task_yaml.exists():
        rel = task_yaml.relative_to(upstream_root)
        dst = snapshot_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(task_yaml, dst)
        copied.append(rel.as_posix())

    csv_path = paths.project_root / "results_augmentation_parameters.csv"
    write_csv(csv_path, rows, PARAMETER_COLUMNS)

    report_path = paths.project_root / "reports_AUGMENTATION_PARAMETERS.md"
    write_markdown(report_path, _augmentation_report(rows, commit, submodule_status, copied))

    compat_path = paths.project_root / "reports_COMPATIBILITY_REPORT.md"
    write_markdown(compat_path, _compatibility_report(rows))

    version_path = paths.project_root / "external_VERSION.md"
    write_markdown(version_path, _version_report(upstream_root, commit, submodule_status))

    repo_report_path = paths.project_root / "reports_REPOSITORY_REPORT.md"
    write_markdown(repo_report_path, _repository_report(upstream_root, commit, submodule_status))

    return UpstreamResult(rows, csv_path, report_path, compat_path, version_path)

def _rows_from_yaml(
    augmentation: str,
    data: dict[str, Any],
    source: Path,
    upstream_root: Path,
    commit: str,
) -> list[dict[str, object]]:
    target = data.get("_target_", "")
    if augmentation == "TrivialAugment":
        rows = [
            _row(
                augmentation=augmentation,
                data=data,
                yaml_key="transforms",
                raw=data.get("transforms", data),
                source=source,
                upstream_root=upstream_root,
                commit=commit,
                sampling_method="Compose of two OneOf blocks; each block samples one transform",
                target=target,
            )
        ]
        for index, transform in enumerate(_flatten_trivial(data)):
            rows.append(
                _row(
                    augmentation=f"TrivialAugment option {index + 1}: {_name_from_target(transform)}",
                    data=transform,
                    yaml_key=f"transforms[{index}]",
                    raw=transform,
                    source=source,
                    upstream_root=upstream_root,
                    commit=commit,
                    sampling_method="Uniform option inside upstream OneOf block",
                    target=transform.get("_target_", ""),
                    probability_override=transform.get("p", 1.0),
                )
            )
        return rows

    keys = [k for k in data.keys() if not k.startswith("_") and k not in {"p"}]
    if not keys:
        keys = ["_target_"]
    return [
        _row(
            augmentation=augmentation,
            data=data,
            yaml_key=key,
            raw=data.get(key, data.get("_target_")),
            source=source,
            upstream_root=upstream_root,
            commit=commit,
            sampling_method=_sampling_method(data, key),
            target=target,
        )
        for key in keys
    ]

def _flatten_trivial(data: dict[str, Any]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for block in data.get("transforms", []) or []:
        for transform in block.get("transforms", []) or []:
            if isinstance(transform, dict):
                found.append(transform)
    return found

def _row(
    augmentation: str,
    data: dict[str, Any],
    yaml_key: str,
    raw: Any,
    source: Path,
    upstream_root: Path,
    commit: str,
    sampling_method: str,
    target: str,
    probability_override: Any | None = None,
) -> dict[str, object]:
    requires_scan_mask = any(
        name in str(target)
        for name in ["DepthAttenuation", "GaussianShadow", "HazeArtifact", "BilateralFilter"]
    )
    applies_to_mask = any(
        name in str(target)
        for name in ["HorizontalFlip", "VerticalFlip", "Rotate", "ShiftScaleRotate", "RandomCrop", "NoOp"]
    )
    return {
        "augmentation": augmentation,
        "source_repository": "https://github.com/adamtupper/ultrasound-augmentation",
        "source_commit": commit,
        "source_yaml_path": source.relative_to(upstream_root).as_posix(),
        "yaml_key": yaml_key,
        "raw_yaml_value": raw,
        "normalized_value": _normalize(raw),
        "effective_probability": probability_override if probability_override is not None else data.get("p", 1.0),
        "sampling_method": sampling_method,
        "applies_to_image": True,
        "applies_to_segmentation_mask": applies_to_mask,
        "requires_scan_mask": requires_scan_mask,
        "requires_panel_processing": requires_scan_mask,
        "image_interpolation": data.get("interpolation", "upstream default"),
        "mask_interpolation": "nearest for segmentation masks",
        "border_mode": data.get("border_mode", "upstream default"),
        "fill_value": data.get("value", "upstream default"),
        "library_class": target,
        "compatibility_notes": _compatibility_note(target),
        "loaded_successfully": True,
    }

def _missing_row(augmentation: str, source: Path, commit: str) -> dict[str, object]:
    return {
        "augmentation": augmentation,
        "source_repository": "https://github.com/adamtupper/ultrasound-augmentation",
        "source_commit": commit,
        "source_yaml_path": source.as_posix(),
        "yaml_key": "",
        "raw_yaml_value": "",
        "normalized_value": "",
        "effective_probability": "",
        "sampling_method": "",
        "applies_to_image": "",
        "applies_to_segmentation_mask": "",
        "requires_scan_mask": "",
        "requires_panel_processing": "",
        "image_interpolation": "",
        "mask_interpolation": "",
        "border_mode": "",
        "fill_value": "",
        "library_class": "",
        "compatibility_notes": "YAML file not found; full training must remain blocked.",
        "loaded_successfully": False,
    }

def _normalize(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    return value

def _sampling_method(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if isinstance(value, list) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
        return "Uniform sampling over inclusive numeric range as implemented upstream"
    if key == "_target_":
        return "Deterministic transform class"
    return "Fixed value from upstream YAML"

def _name_from_target(data: dict[str, Any]) -> str:
    target = str(data.get("_target_", "unknown"))
    return target.rsplit(".", 1)[-1]

def _compatibility_note(target: str) -> str:
    if "ShiftScaleRotate" in str(target):
        return "Albumentations API may warn that ShiftScaleRotate is superseded by Affine; no silent conversion is applied."
    if "GaussNoise" in str(target):
        return "Albumentations versions differ in noise parameter names; upstream raw value is preserved."
    if "usaugment" in str(target):
        return "Requires upstream usaugment package and scan_mask additional target."
    return "Loaded directly from upstream YAML; no conversion applied."

def _augmentation_report(rows: list[dict[str, object]], commit: str, submodules: str, copied: list[str]) -> str:
    loaded = sum(1 for row in rows if row.get("loaded_successfully") is True)
    failed = sum(1 for row in rows if row.get("loaded_successfully") is False)
    names = sorted({str(row["augmentation"]).split(":")[0] for row in rows})
    return "\n".join(
        [
            "# Augmentation Parameter Report",
            "",
            f"- Generated: {datetime.now(timezone.utc).isoformat()}",
            f"- Upstream commit: `{commit}`",
            f"- Loaded rows: {loaded}",
            f"- Failed rows: {failed}",
            f"- Snapshot files copied: {len(copied)}",
            "",
            "## Upstream Submodules",
            "",
            "```text",
            submodules,
            "```",
            "",
            "## Augmentations",
            "",
            *[f"- {name}" for name in names],
            "",
            "## Snapshot",
            "",
            *[f"- `configs/upstream_snapshot/{path}`" for path in copied],
            "",
            "Full table: `results_augmentation_parameters.csv`.",
            "",
        ]
    )

def _compatibility_report(rows: list[dict[str, object]]) -> str:
    notes = sorted({str(row.get("compatibility_notes", "")) for row in rows if row.get("compatibility_notes")})
    return "\n".join(
        [
            "# Compatibility Report",
            "",
            "No compatibility conversion is performed silently. Upstream raw YAML values are preserved.",
            "",
            "## Notes",
            "",
            *[f"- {note}" for note in notes],
            "",
            "Before full training, verify Albumentations and usaugment behavior with unit tests using the pinned upstream commit.",
            "",
        ]
    )

def _version_report(upstream_root: Path, commit: str, submodules: str) -> str:
    return "\n".join(
        [
            "# External Repository Version",
            "",
            f"- Repository: {relative_to_root(upstream_root, upstream_root.parent.parent)}",
            f"- Commit: `{commit}`",
            f"- Retrieved: {datetime.now(timezone.utc).isoformat()}",
            "",
            "## Submodules",
            "",
            "```text",
            submodules,
            "```",
            "",
        ]
    )

def _repository_report(upstream_root: Path, commit: str, submodules: str) -> str:
    license_file = upstream_root / "LICENSE"
    license_note = license_file.read_text(encoding="utf-8", errors="replace").splitlines()[0] if license_file.exists() else "unknown"
    return "\n".join(
        [
            "# Repository Report",
            "",
            f"- Upstream URL: https://github.com/adamtupper/ultrasound-augmentation",
            f"- Upstream commit: `{commit}`",
            f"- Upstream license first line: {license_note}",
            "- External repository is not modified by this project.",
            "",
            "## Submodule Status",
            "",
            "```text",
            submodules,
            "```",
            "",
        ]
    )
