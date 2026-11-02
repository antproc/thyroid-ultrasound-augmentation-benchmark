from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thyroid_augmentation.cross_dataset import analyze, prepare, run, smoke

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "smoke", "run", "analyze"))
    args = parser.parse_args()
    {"prepare": prepare, "smoke": smoke, "run": run, "analyze": analyze}[args.action](ROOT)

if __name__ == "__main__":
    main()
