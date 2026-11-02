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
    parser.add_argument("--action", choices=("prepare", "smoke", "preflight"), default="prepare")
    args = parser.parse_args()
    from thyroid_augmentation.extension import experiment
    if args.action == "prepare":
        from thyroid_augmentation.extension.prepare import prepare
        print(json.dumps(prepare(ROOT), indent=2))
    elif args.action == "smoke":
        print(json.dumps(experiment.diagnostic_smoke(ROOT), indent=2))
    else:
        print(json.dumps(experiment.preflight(ROOT), indent=2))

if __name__ == "__main__":
    main()
