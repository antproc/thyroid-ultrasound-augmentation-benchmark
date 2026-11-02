from __future__ import annotations

import ctypes
import os
import subprocess
import time
from ctypes import wintypes

import psutil

BUDGETS = {"res34_unet": 10, "deeplabv3plus_resnet50": 14, "mit_b2_unet": 18}
HEADROOM = 2

class ProcessWatch:

    def __init__(self, pid: int, script: str, action: str):
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        self.kernel.GetExitCodeProcess.restype = wintypes.BOOL
        self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.kernel.WaitForSingleObject.restype = wintypes.DWORD
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL
        self.handle = self.kernel.OpenProcess(0x100000 | 0x1000, False, pid)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            p = psutil.Process(pid)
            args = p.cmdline()
            if not any(os.path.basename(x.replace("\\", "/")) == script for x in args):
                raise ValueError("Watcher PID is not the requested launcher.")
            if "--action" not in args or args[args.index("--action") + 1] != action:
                raise ValueError("Watcher PID has the wrong action.")
            if self.exit_code() is not None:
                raise ValueError("Attach to a live launcher, not an already exited process.")
            self.identity = {"pid": pid, "created": p.create_time(), "command": args}
        except BaseException:
            self.close()
            raise

    def exit_code(self):
        result = self.kernel.WaitForSingleObject(self.handle, 0)
        if result == 258:
            return None
        if result != 0:
            raise ctypes.WinError(ctypes.get_last_error())
        code = wintypes.DWORD()
        if not self.kernel.GetExitCodeProcess(self.handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return code.value

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None

def snapshot() -> dict:
    line = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total,memory.free",
                                    "--format=csv,noheader,nounits"], text=True).strip().splitlines()
    if len(line) != 1:
        raise RuntimeError("Exactly one GPU is required.")
    total, free = [float(x) / 1024 for x in line[0].split(",")]
    raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
    gpu = {int(x.strip()) for x in raw.splitlines() if x.strip()}
    single, ra, other_ta = set(), set(), set()
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            args = p.info["cmdline"] or []
            names = {os.path.basename(x.replace("\\", "/")) for x in args}
            if "01_prepare_extension_datasets.py" in names and ("run-single" in args or "run-one" in args):
                single.add(p.pid)
            if "02_extension_randaugment_queue.py" in names and ("run" in args or "run-one" in args):
                ra.add(p.pid)
            if "03_extension_trivialaugment_queue.py" in names and p.pid != os.getpid() and "run-one" in args:
                other_ta.add(p.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return {"total": total, "free": free, "single": sorted(single), "ra": sorted(ra),
            "ra_on_gpu": bool(gpu & ra), "other_ta": sorted(other_ta),
            "unknown_gpu": sorted(gpu - ra - single - {os.getpid()})}

def admission(model: str, state: dict) -> tuple[bool, str]:
    if state["single"]:
        return False, "single is still active"
    if state["other_ta"] or state["unknown_gpu"]:
        return False, "another GPU worker is active"
    budget = BUDGETS[model]
    if state["ra"]:

        if budget + max(BUDGETS.values()) + HEADROOM > state["total"]:
            return False, "this TA model waits for full RA completion"
        if not state["ra_on_gpu"]:
            return False, "yielding the next GPU slot to RandAugment"
    if state["free"] < budget + HEADROOM:
        return False, "insufficient free VRAM"
    return True, "admitted"

def wait_for_slot(root, model: str):
    from thyroid_augmentation.trivial_extension.pipeline import require_ra_success

    while True:
        state = snapshot()
        if not state["ra"]:
            require_ra_success(root)
        allowed, reason = admission(model, state)
        if allowed:
            return state
        print(f"TA waiting: {reason}; free={state['free']:.1f} GiB", flush=True)
        time.sleep(30)

def limit_memory(model: str):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback.")
    fraction = BUDGETS[model] * 1024**3 / torch.cuda.get_device_properties(0).total_memory
    if not 0 < fraction < 1:
        raise RuntimeError("GPU memory budget cannot be satisfied.")
    torch.cuda.set_per_process_memory_fraction(fraction)
