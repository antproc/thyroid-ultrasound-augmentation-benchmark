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
    parser.add_argument("--action", choices=("plan", "preflight", "smoke", "run", "run-one"), default="plan")
    parser.add_argument("--confirm-training", action="store_true")
    parser.add_argument("--allow-concurrent-single", action="store_true")
    parser.add_argument("--job-index", type=int)
    args = parser.parse_args()
    if args.action in ("run", "run-one") and not args.confirm_training:
        parser.error("Training requires --confirm-training")
    if args.action == "run-one" and (args.job_index is None or not 0 <= args.job_index < 324):
        parser.error("--job-index must be 0..323")
    from thyroid_augmentation.randaugment_extension import pipeline
    if args.action == "plan":
        grid = pipeline.jobs(ROOT)
        print(json.dumps({"runs": len(grid), "first": grid[0], "last": grid[-1], "concurrency": pipeline.settings(ROOT)["optional_concurrent_single"]}, indent=2))
    elif args.action == "preflight":
        report = pipeline.preflight(ROOT, Path(__file__).resolve())
        print(json.dumps({k: report[k] for k in ("status", "runs", "samples", "transform_checks", "longest_path", "concurrency")}, indent=2))
    elif args.action == "smoke":
        print(json.dumps(pipeline.gpu_smoke(ROOT), indent=2))
    elif args.action == "run":
        pipeline.run_all(ROOT, Path(__file__).resolve(), args.allow_concurrent_single)
    else:
        pipeline.run_one(ROOT, args.job_index, args.allow_concurrent_single)

if __name__ == "__main__":
    main()
