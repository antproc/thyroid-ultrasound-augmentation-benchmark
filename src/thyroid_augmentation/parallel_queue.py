from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

from thyroid_augmentation.extension.common import write_json

BUDGETS = {"res34_unet": 10, "deeplabv3plus_resnet50": 14, "mit_b2_unet": 18}
HEADROOM = 2

def snapshot(known_pids=()):
    lines = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total,memory.free",
                                     "--format=csv,noheader,nounits"], text=True).strip().splitlines()
    if len(lines) != 1:
        raise RuntimeError("Exactly one GPU is required.")
    total, free = [float(x)/1024 for x in lines[0].split(",")]
    raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                   "--format=csv,noheader,nounits"], text=True)
    gpu_pids = {int(x.strip()) for x in raw.splitlines() if x.strip()}
    return {"total": total, "free": free, "unknown": sorted(gpu_pids - set(known_pids) - {os.getpid()})}

def admission(model, active_models, state):
    if state["unknown"]:
        return False, "unrecognized GPU process"
    if len(active_models) >= 2:
        return False, "two workers already active"
    if BUDGETS[model] + sum(BUDGETS[m] for m in active_models) + HEADROOM > state["total"]:
        return False, "model combination exceeds reserved GPU capacity"
    if state["free"] < BUDGETS[model] + HEADROOM:
        return False, "insufficient currently free GPU memory"
    return True, "admitted"

def limit_memory(model):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; CPU fallback is forbidden.")
    fraction = BUDGETS[model]*1024**3 / torch.cuda.get_device_properties(0).total_memory
    if not 0 < fraction < 1:
        raise RuntimeError("Invalid GPU capacity for this model.")
    torch.cuda.set_per_process_memory_fraction(fraction)

def disk_admission(lane, queue, pending_single, candidate, active):

    single_configs = [config for _, config in pending_single]
    single_configs.extend(w["config"] for key, w in active.items() if key == "single")
    ra_configs = [w["config"] for key, w in active.items() if key == "ra"]
    if lane == "ra":
        ra_configs.append(candidate)
    return queue.check_storage(single_configs + ra_configs)

def run(queue, *, include_ta=True):
    queue.require_preflight()
    queue.require_parallel_preflight()
    grids = {stage: queue.stage_jobs(stage) for stage in ("single", "ra")}
    pending = {stage: list(enumerate(jobs)) for stage, jobs in grids.items()}
    active, finished, failure = {}, set(), None
    status_path = queue.OUT / "parallel_status.json"
    log_dir = queue.OUT / "parallel_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with queue.queue_lock():
        try:
            while pending["single"] or pending["ra"] or active:
                for lane, worker in list(active.items()):
                    code = worker["process"].poll()
                    if code is None:
                        continue
                    worker["handle"].close()
                    del active[lane]
                    if code != 0:
                        failure = f"{lane} job {worker['index']} failed with exit code {code}"
                    else:
                        print(f"Completed {lane} job {worker['index']}; log={worker['log']}", flush=True)
                if failure:
                    if active:
                        write_json(status_path, {"status": "draining_after_error", "error": failure,
                                   "active_pids": [w["process"].pid for w in active.values()]})
                        time.sleep(5)
                        continue
                    write_json(status_path, {"status": "failed", "error": failure, "active": {}})
                    raise RuntimeError(failure)
                for lane in ("single", "ra"):
                    if not pending[lane] and lane not in active and lane not in finished:
                        queue.finish_stage(lane, grids[lane])
                        finished.add(lane)
                reasons = {}
                for lane in ("single", "ra"):
                    if lane in active or not pending[lane]:
                        continue

                    candidates = pending[lane] if lane == "ra" else pending[lane][:1]
                    state = snapshot([w["process"].pid for w in active.values()])
                    for item in candidates:
                        index, config = item
                        disk = disk_admission(lane, queue, pending["single"], config, active)
                        if not disk["fits"]:
                            reasons[lane] = {"kind": "disk", **disk}
                            break
                        allowed, reason = admission(config["model"], [w["config"]["model"] for w in active.values()], state)
                        if not allowed:
                            reasons[lane] = {"kind": "gpu", "reason": reason, "state": state}
                            continue
                        log = log_dir / f"{lane}_{index:04d}.log"
                        handle = log.open("ab", buffering=0)
                        try:
                            process = subprocess.Popen([sys.executable, "-B", str(queue.ROOT / "scripts/03_benchmark.py"),
                                                       "run-one", "--stage", lane, "--index", str(index),
                                                       "--confirm-training", "--gpu-memory-cap"],
                                                      cwd=queue.ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                        except BaseException:
                            handle.close()
                            raise
                        active[lane] = {"process": process, "config": config, "index": index,
                                        "handle": handle, "log": str(log)}
                        pending[lane].remove(item)
                        print(f"Started {lane} job {index}: {config['dataset']}/{config['model']} PID={process.pid}", flush=True)
                        break
                write_json(status_path, {"status": "running", "pending": {k: len(v) for k, v in pending.items()},
                           "active": {k: {"pid": w["process"].pid, "index": w["index"], "config": w["config"], "log": w["log"]}
                                      for k, w in active.items()}, "waiting": reasons})
                if not active and reasons and all(r["kind"] == "disk" for r in reasons.values()):
                    write_json(status_path, {"status": "blocked_on_C_capacity", "waiting": reasons,
                               "finished_stages": sorted(finished), "automatic_deletion": False})
                    raise RuntimeError("Insufficient C: capacity; all active workers have finished. No further jobs started.")
                if active or pending["single"] or pending["ra"]:
                    time.sleep(5)
            for lane in ("single", "ra"):
                if lane not in finished:
                    queue.finish_stage(lane, grids[lane])
            write_json(status_path, {"status": "single_and_ra_complete", "active": {}})
        finally:

            for worker in active.values():
                if worker["process"].poll() is None:
                    worker["process"].wait()
                worker["handle"].close()
    if include_ta:
        queue.run_queue("ta")
