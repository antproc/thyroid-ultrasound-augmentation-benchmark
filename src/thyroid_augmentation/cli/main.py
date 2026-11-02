from __future__ import annotations

import argparse
from pathlib import Path

from thyroid_augmentation.preprocessing.scan_masks import prepare_scan_masks, prepare_tucc_scan_masks

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m thyroid_augmentation")
    parser.add_argument("command", choices=("prepare-scan-masks", "prepare-tucc-scan-masks"))
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    root = Path(args.project_root).resolve() if args.project_root else Path.cwd().resolve()
    if args.command == "prepare-scan-masks":
        out = prepare_scan_masks(root, limit=args.limit)
    else:
        out = prepare_tucc_scan_masks(root, limit=args.limit)
    print(f"wrote {out}")
    return 0
