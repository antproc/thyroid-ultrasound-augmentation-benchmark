from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=("prepare", "arm", "resume", "run-one", "smoke"), default="prepare")
    parser.add_argument("--confirm-training", action="store_true")
    parser.add_argument("--single-pid", type=int)
    parser.add_argument("--ra-pid", type=int)
    parser.add_argument("--job-index", type=int)
    args = parser.parse_args()
    if args.action != "prepare" and not args.confirm_training:
        parser.error("Automatic future training requires --confirm-training")
    if args.action == "arm" and (not args.single_pid or not args.ra_pid):
        parser.error("--single-pid and --ra-pid identify the current live launchers")
    if args.action == "run-one" and (args.job_index is None or not 0 <= args.job_index < 351):
        parser.error("--job-index must be 0..350")
    from thyroid_augmentation.trivial_extension import pipeline as p
    script = Path(__file__).resolve()
    if args.action == "prepare":
        report = p.prepare(ROOT)
        print(json.dumps({k: report[k] for k in ("status", "runs", "longest_path", "transform_checks", "storage", "gpu_smoke")}, indent=2))
    elif args.action == "arm":
        p.arm(ROOT, script, args.single_pid, args.ra_pid)
    elif args.action == "run-one":
        p.run_one(ROOT, args.job_index)
    else:
        p.require_preflight(ROOT, ready=False)
        receipt = json.loads((ROOT / p.OUTPUT / "single_exit.json").read_text())
        if receipt["exit_code"] != 0:
            raise RuntimeError("Single did not exit successfully; TA blocked.")
        p.completion_ok(json.loads((ROOT / "results/extension/single/complete.json").read_text()), p.source_hashes(ROOT), 405)
        if args.action == "smoke":
            grid = json.loads((ROOT / p.OUTPUT / "jobs.json").read_text())["jobs"]
            p.gpu_smoke(ROOT, grid)
        else:
            with p.ra.exclusive_lock(ROOT / p.OUTPUT / "queue.lock"):
                p.gate(ROOT, script)
                p.run_queue(ROOT, script)

if __name__ == "__main__":
    main()
