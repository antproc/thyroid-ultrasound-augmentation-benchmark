from __future__ import annotations

import time
from pathlib import Path

import torch

from thyroid_augmentation.models.base import count_parameters
from thyroid_augmentation.models.registry import available_models, build_model, model_specs
from thyroid_augmentation.paths import ProjectPaths
from thyroid_augmentation.utils import write_csv, write_markdown

MODEL_TEST_COLUMNS = [
    "model",
    "architecture",
    "encoder",
    "parameters",
    "forward_ok",
    "backward_ok",
    "device",
    "max_cuda_memory_mb",
    "seconds",
    "output_shape",
]

def run_model_smoke_tests(paths: ProjectPaths, image_size: int = 96, device: str = "cpu") -> Path:
    rows: list[dict[str, object]] = []
    specs = model_specs()
    torch_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    for name in available_models():
        if torch_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(torch_device)
        start = time.perf_counter()
        forward_ok = False
        backward_ok = False
        output_shape = ""
        params = ""
        try:
            model = build_model(name, in_channels=3, classes=1, encoder_weights=None).to(torch_device)
            model.train()
            params = count_parameters(model)
            x = torch.randn(2, 3, image_size, image_size, device=torch_device)
            y = model(x)
            output_shape = tuple(y.shape)
            forward_ok = tuple(y.shape)[0] == 2 and tuple(y.shape)[1] == 1
            loss = y.mean()
            loss.backward()
            backward_ok = any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
        except Exception as exc:
            output_shape = f"error:{type(exc).__name__}:{exc}"
        seconds = time.perf_counter() - start
        max_memory = ""
        if torch_device.type == "cuda":
            max_memory = f"{torch.cuda.max_memory_allocated(torch_device) / 1024**2:.1f}"
        rows.append(
            {
                "model": name,
                "architecture": specs[name].architecture,
                "encoder": specs[name].encoder,
                "parameters": params,
                "forward_ok": forward_ok,
                "backward_ok": backward_ok,
                "device": str(torch_device),
                "max_cuda_memory_mb": max_memory,
                "seconds": f"{seconds:.3f}",
                "output_shape": output_shape,
            }
        )
    csv_path = paths.project_root / "reports" / "model_smoke_tests.csv"
    write_csv(csv_path, rows, MODEL_TEST_COLUMNS)
    write_markdown(paths.project_root / "reports_REGISTRY_REPORT.md", _registry_report(rows))
    return csv_path

def _registry_report(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Model Registry Report",
        "",
        "| Model | Architecture | Encoder | Parameters | Forward | Backward | Device | Max CUDA MB | Seconds |",
        "|---|---|---|---:|---|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['architecture']} | {row['encoder']} | {row['parameters']} | "
            f"{row['forward_ok']} | {row['backward_ok']} | {row['device']} | "
            f"{row['max_cuda_memory_mb']} | {row['seconds']} |"
        )
    lines.append("")
    return "\n".join(lines)
