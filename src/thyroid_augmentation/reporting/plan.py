from __future__ import annotations

from pathlib import Path

from thyroid_augmentation.models.registry import available_models
from thyroid_augmentation.paths import ProjectPaths
from thyroid_augmentation.utils import write_markdown

def write_runtime_plan(paths: ProjectPaths) -> Path:
    primary_runs_per_dataset = 15 * 3 + 13 * 3 + 3 * 2
    primary_total = primary_runs_per_dataset * 3
    architecture_validation_per_model = 3 * 3 * 3
    extra_models = max(0, len(available_models()) - 1)
    architecture_validation_total = architecture_validation_per_model * extra_models
    full_architecture_sweep_total = primary_runs_per_dataset * 3 * len(available_models())
    gpu_note = _gpu_note()

    lines = [
        "# Runtime and Training Plan",
        "",
        "No full experiment is launched by this preparation report.",
        "",
        "## Planned Runs",
        "",
        f"- Primary benchmark: {primary_runs_per_dataset} runs per dataset.",
        f"- Primary benchmark across TUCC, TN3K, DDTI: {primary_total} runs.",
        f"- Architecture validation: {architecture_validation_per_model} runs per additional architecture.",
        f"- Architecture validation for {extra_models} additional architectures: {architecture_validation_total} runs.",
        f"- Full architecture sweep for all registered models: {full_architecture_sweep_total} runs.",
        "",
        "## Approval Gates",
        "",
        "- Primary benchmark requires `APPROVED_TO_TRAIN.txt`.",
        "- Architecture validation requires `APPROVED_ARCHITECTURE_VALIDATION.txt`.",
        "- Full architecture sweep requires `APPROVED_FULL_ARCHITECTURE_SWEEP.txt`.",
        "",
        "## Commands",
        "",
        "Primary benchmark after approval:",
        "",
        "```powershell",
        ".\\scripts\\09_run_individual_augmentations.ps1 -Dataset TUCC -Model unet_efficientnet_b0 -Resume",
        ".\\scripts\\09_run_individual_augmentations.ps1 -Dataset TN3K -Model unet_efficientnet_b0 -Resume",
        ".\\scripts\\09_run_individual_augmentations.ps1 -Dataset DDTI -Model unet_efficientnet_b0 -Resume",
        ".\\scripts\\11_run_trivialaugment.ps1 -Dataset TUCC -Model unet_efficientnet_b0 -Resume",
        ".\\scripts\\11_run_trivialaugment.ps1 -Dataset TN3K -Model unet_efficientnet_b0 -Resume",
        ".\\scripts\\11_run_trivialaugment.ps1 -Dataset DDTI -Model unet_efficientnet_b0 -Resume",
        "```",
        "",
        "Architecture validation after approval:",
        "",
        "```powershell",
        ".\\scripts\\14_run_architecture_validation.ps1 -Model res34_unet -Resume",
        ".\\scripts\\14_run_architecture_validation.ps1 -Model deeplabv3plus_resnet50 -Resume",
        ".\\scripts\\14_run_architecture_validation.ps1 -Model mit_b2_unet -Resume",
        "```",
        "",
        "## Runtime Estimate",
        "",
        gpu_note,
        "",
        "Measured epoch time: not recorded.",
        "",
    ]

    out = paths.project_root / "reports_PLAN.md"
    write_markdown(out, "\n".join(lines))
    write_markdown(paths.project_root / "reports_RUNTIME_REPORT.md", "\n".join(lines))
    return out

def _gpu_note() -> str:
    try:
        import torch

        if not torch.cuda.is_available():
            return "Detected GPU: CUDA not available."
        props = torch.cuda.get_device_properties(0)
        memory_gb = props.total_memory / 1024**3
        return f"Detected GPU: `{torch.cuda.get_device_name(0)}` with {memory_gb:.2f} GB VRAM."
    except Exception as exc:
        return f"Detected GPU: unavailable ({type(exc).__name__}: {exc})."
