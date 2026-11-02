from __future__ import annotations

import os
import subprocess
import time

import psutil

def gpu_snapshot() -> dict:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"], text=True)
    lines = output.strip().splitlines()
    if len(lines) != 1:
        raise RuntimeError("This launcher requires exactly one visible GPU.")
    total, free = [float(x.strip()) / 1024 for x in lines[0].split(",")]
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
    gpu_pids = {int(x.strip()) for x in raw.splitlines() if x.strip()}
    single_pids = set()
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            args = process.info["cmdline"] or []
            if any(os.path.basename(x.replace("\\", "/")) == "01_prepare_extension_datasets.py" for x in args):
                if "run-single" in args or "run-one" in args:
                    single_pids.add(process.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return {"total_GiB": total, "free_GiB": free, "single_active": bool(single_pids),
            "single_pids": sorted(single_pids), "unknown_gpu_pids": sorted(gpu_pids - single_pids - {os.getpid()})}

def admission(model: str, state: dict, settings: dict, concurrent: bool) -> tuple[bool, str]:
    if state["unknown_gpu_pids"]:
        return False, "another unrecognized GPU process is active"
    budget = settings["memory_budget_GiB"][model]
    headroom = settings["gpu_headroom_GiB"]
    if state["single_active"]:
        if not concurrent:
            return False, "waiting for the single stage (serial mode)"
        if model != "res34_unet":
            return False, "larger RA models wait for single completion to preserve GPU headroom"
        if budget + settings["single_reserve_GiB"] + headroom > state["total_GiB"]:
            return False, "GPU is too small for the conservative concurrent budgets"
    if state["free_GiB"] < budget + headroom:
        return False, "not enough currently free GPU memory"
    return True, "admitted"

def wait_for_gpu(model: str, settings: dict, concurrent: bool) -> dict:
    while True:
        state = gpu_snapshot()
        allowed, reason = admission(model, state, settings, concurrent)
        if allowed:
            return state
        print(f"RandAugment waiting: {reason}; free={state['free_GiB']:.1f} GiB", flush=True)
        time.sleep(30)

def apply_memory_limit(model: str, settings: dict) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback.")
    total = torch.cuda.get_device_properties(0).total_memory
    fraction = settings["memory_budget_GiB"][model] * 1024**3 / total
    if not 0 < fraction < 1:
        raise RuntimeError("GPU does not satisfy the model memory budget.")
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
