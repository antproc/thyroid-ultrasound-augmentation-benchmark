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
    parser.add_argument("--action", choices=("plan", "prepare", "smoke", "preflight", "run-single", "run-one"), default="plan")
    parser.add_argument("--confirm-training", action="store_true")
    parser.add_argument("--job-index", type=int)
    args = parser.parse_args()
    from thyroid_augmentation.extension import experiment
    if args.action == "plan":
        grid = experiment.jobs(ROOT)
        print(json.dumps({"runs": len(grid), "first": grid[0], "last": grid[-1]}, indent=2))
    elif args.action == "prepare":
        from thyroid_augmentation.extension.prepare import prepare
        print(json.dumps(prepare(ROOT), indent=2))
    elif args.action == "preflight":
        print(json.dumps(experiment.preflight(ROOT), indent=2))
    elif args.action == "smoke":
        print(json.dumps(experiment.diagnostic_smoke(ROOT), indent=2))
    elif args.action == "run-single":
        if not args.confirm_training:
            parser.error("Starting 405 runs requires --confirm-training")
        experiment.run_single(ROOT, Path(__file__).resolve())
    else:
        if not args.confirm_training or args.job_index is None or not 0 <= args.job_index < 405:
            parser.error("A valid job index and --confirm-training are required")
        experiment.run_one(ROOT, args.job_index)

if __name__ == "__main__":
    main()
