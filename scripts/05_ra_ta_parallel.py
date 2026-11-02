from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thyroid_augmentation.ra_ta_handoff import run, validate

def load_queue():
    path = ROOT / "scripts/03_benchmark.py"
    spec = importlib.util.spec_from_file_location("benchmark_queue_handoff", path)
    queue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(queue)
    return queue

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "run"))
    parser.add_argument("--confirm-training", action="store_true")
    args = parser.parse_args()
    if args.action == "run" and not args.confirm_training:
        parser.error("run requires --confirm-training")
    queue = load_queue()
    implementation = [Path(__file__).resolve(),
                      ROOT / "src/thyroid_augmentation/ra_ta_handoff.py"]
    if args.action == "validate":
        print(json.dumps(validate(queue, implementation), indent=2), flush=True)
    else:
        run(queue, implementation)

if __name__ == "__main__":
    main()

