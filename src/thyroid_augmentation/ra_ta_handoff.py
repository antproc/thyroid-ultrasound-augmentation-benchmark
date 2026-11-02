from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

from thyroid_augmentation.parallel_queue import admission, snapshot
from thyroid_augmentation.extension.common import digest, write_json

COMPLETE_STATUS = "complete_exports_and_local_backup_verified"

def _complete_receipt(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == COMPLETE_STATUS
    except (OSError, ValueError):
        return False

def verified_complete(queue, config: dict) -> bool:
    run, checkpoint = queue.paths(config)
    identifier = queue.run_id(config)
    primary = queue.OUT / "per_image" / identifier / "complete.json"
    backup = queue.OUT / "backup" / "per_image" / identifier / "complete.json"
    return (checkpoint.is_file() and (run / "summary.json").is_file()
            and _complete_receipt(primary) and _complete_receipt(backup))

def pending_jobs(queue, jobs: dict[str, list[dict]]) -> dict[str, list[tuple[int, dict]]]:
    return {lane: [(index, config) for index, config in enumerate(configs)
                   if not verified_complete(queue, config)]
            for lane, configs in jobs.items()}

def storage_work(pending: dict[str, list[tuple[int, dict]]], active: dict) -> list[dict]:
    configs = [config for lane in ("ra", "ta") for _, config in pending[lane]]
    configs.extend(worker["config"] for worker in active.values())
    return configs

def choose_candidate(pending: list[tuple[int, dict]], active: dict, state: dict):
    active_models = [worker["config"]["model"] for worker in active.values()]
    last_reason = "no compatible pending model"
    for item in pending:
        allowed, reason = admission(item[1]["model"], active_models, state)
        if allowed:
            return item, "admitted"
        last_reason = reason
    return None, last_reason

def _self_fingerprint(paths: list[Path]) -> dict[str, str]:
    return {str(path.resolve()): digest(path) for path in paths}

def validate(queue, implementation_paths: list[Path]) -> dict:
    prepared = queue.require_preflight()
    queue.require_parallel_preflight()
    single = json.loads((queue.OUT / "single_complete.json").read_text(encoding="utf-8"))
    if single.get("status") != "complete" or single.get("runs") != 810:
        raise ValueError("The complete 810-run single stage is required before TA handoff.")
    jobs = {lane: queue.stage_jobs(lane) for lane in ("ra", "ta")}
    if len(jobs["ra"]) != 648 or len(jobs["ta"]) != 702:
        raise ValueError("Unexpected corrected RA/TA grid size.")
    identifiers = [queue.run_id(config) for lane in ("ra", "ta") for config in jobs[lane]]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate RA/TA run identifiers.")
    pending = pending_jobs(queue, jobs)
    storage = queue.check_storage(storage_work(pending, {}))
    report = {
        "status": "ready_for_controlled_handoff",
        "training_started": False,
        "metric_protocol": prepared["metric_protocol"],
        "source_hashes": prepared["source_hashes"],
        "implementation_hashes": _self_fingerprint(implementation_paths),
        "jobs": {lane: len(configs) for lane, configs in jobs.items()},
        "verified_complete": {lane: len(jobs[lane]) - len(pending[lane]) for lane in jobs},
        "remaining": {lane: len(pending[lane]) for lane in jobs},
        "storage": storage,
        "requires_old_launcher_stopped": True,
        "launches_training": False,
    }
    write_json(queue.OUT / "ra_ta_handoff_prepared.json", report)
    return report

def require_validated(queue, implementation_paths: list[Path]) -> dict:
    path = queue.OUT / "ra_ta_handoff_prepared.json"
    if not path.is_file():
        raise ValueError("Run the RA/TA handoff validation first.")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "ready_for_controlled_handoff":
        raise ValueError("RA/TA handoff validation did not pass.")
    if report.get("implementation_hashes") != _self_fingerprint(implementation_paths):
        raise ValueError("RA/TA handoff implementation changed after validation.")
    if report.get("source_hashes") != queue.require_preflight()["source_hashes"]:
        raise ValueError("Corrected experiment fingerprint changed after validation.")
    queue.require_parallel_preflight()
    return report

def run(queue, implementation_paths: list[Path]):
    require_validated(queue, implementation_paths)
    jobs = {lane: queue.stage_jobs(lane) for lane in ("ra", "ta")}
    pending = pending_jobs(queue, jobs)
    active = {}
    finished = {lane for lane in ("ra", "ta") if (queue.OUT / f"{lane}_complete.json").is_file()}
    failure = None
    status_path = queue.OUT / "ra_ta_parallel_status.json"
    log_dir = queue.OUT / "ra_ta_parallel_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    with queue.queue_lock():
        try:
            while any(pending.values()) or active:

                for lane in ("ra", "ta"):
                    pending[lane] = [(index, config) for index, config in pending[lane]
                                     if not verified_complete(queue, config)]

                for lane, worker in list(active.items()):
                    code = worker["process"].poll()
                    if code is None:
                        continue
                    worker["handle"].close()
                    del active[lane]
                    if code != 0:
                        failure = f"{lane} job {worker['index']} failed with exit code {code}"
                    elif not verified_complete(queue, worker["config"]):
                        failure = f"{lane} job {worker['index']} exited without verified exports"
                    else:
                        print(f"Completed {lane} job {worker['index']}; log={worker['log']}", flush=True)

                if failure:
                    if active:
                        write_json(status_path, {"status": "draining_after_error", "error": failure,
                                   "active_pids": [worker["process"].pid for worker in active.values()]})
                        time.sleep(5)
                        continue
                    write_json(status_path, {"status": "failed", "error": failure, "active": {}})
                    raise RuntimeError(failure)

                if not active:
                    for lane in ("ra", "ta"):
                        if not pending[lane] and lane not in finished:
                            queue.finish_stage(lane, jobs[lane])
                            finished.add(lane)

                reasons = {}
                for lane in ("ra", "ta"):
                    if lane in active or not pending[lane]:
                        continue
                    disk = queue.check_storage(storage_work(pending, active))
                    if not disk["fits"]:
                        reasons[lane] = {"kind": "disk", **disk}
                        continue
                    state = snapshot([worker["process"].pid for worker in active.values()])
                    item, reason = choose_candidate(pending[lane], active, state)
                    if item is None:
                        reasons[lane] = {"kind": "gpu", "reason": reason, "state": state}
                        continue
                    index, config = item
                    log = log_dir / f"{lane}_{index:04d}.log"
                    handle = log.open("ab", buffering=0)
                    try:
                        process = subprocess.Popen(
                            [sys.executable, "-B", str(queue.ROOT / "scripts/03_benchmark.py"),
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
                    print(f"Started {lane} job {index}: {config['dataset']}/{config['model']} PID={process.pid}",
                          flush=True)

                write_json(status_path, {
                    "status": "running",
                    "pending": {lane: len(items) for lane, items in pending.items()},
                    "active": {lane: {"pid": worker["process"].pid, "index": worker["index"],
                                      "config": worker["config"], "log": worker["log"]}
                               for lane, worker in active.items()},
                    "waiting": reasons,
                    "finished_stages": sorted(finished),
                })
                if not active and reasons and all(reason["kind"] == "disk" for reason in reasons.values()):
                    write_json(status_path, {"status": "blocked_on_C_capacity", "waiting": reasons,
                               "finished_stages": sorted(finished), "automatic_deletion": False})
                    raise RuntimeError("Insufficient C: capacity; no RA/TA job was started.")
                if any(pending.values()) or active:
                    time.sleep(5)

            for lane in ("ra", "ta"):
                if lane not in finished:
                    queue.finish_stage(lane, jobs[lane])
            write_json(status_path, {"status": "ra_and_ta_complete", "active": {},
                       "finished_stages": ["ra", "ta"]})
        finally:
            for worker in active.values():
                if worker["process"].poll() is None:
                    worker["process"].wait()
                worker["handle"].close()

