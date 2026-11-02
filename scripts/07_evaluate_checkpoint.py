from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", required=True, choices=("res34_unet", "deeplabv3plus_resnet50", "mit_b2_unet"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--split-manifest", default="manifests/all/splits.csv")
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from thyroid_augmentation.models.registry import build_model
    from thyroid_augmentation.training.metrics import per_image_metrics
    from thyroid_augmentation.training.runner import SegmentationManifestDataset

    with (ROOT / args.split_manifest).open(encoding="utf-8-sig", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["dataset"] == args.dataset and r["split"] == args.split]
    if not rows:
        raise SystemExit("no rows for the requested dataset and split")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = {k: (v.float() if v.is_floating_point() else v) for k, v in payload["model"].items()}
    model = build_model(args.model, in_channels=3, classes=1, encoder_weights=None, dropout=0.0, pretrained=False)
    model.load_state_dict(state)
    model.to(device).eval()
    loader = DataLoader(SegmentationManifestDataset(rows, ROOT, 224, training=False), batch_size=args.batch_size, shuffle=False, num_workers=0)
    output = []
    offset = 0
    with torch.inference_mode():
        for images, masks in loader:
            batch = rows[offset: offset + images.shape[0]]
            offset += images.shape[0]
            logits = model(images.to(device))
            masks = masks.to(device)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            metrics = per_image_metrics(logits.float(), masks.float())
            for row, dice, iou in zip(batch, metrics["dice"].cpu().tolist(), metrics["iou"].cpu().tolist()):
                output.append({"dataset": args.dataset, "split": args.split, "sample_id": row["sample_id"], "group_id": row["group_id"], "dice": dice, "iou": iou})
    mean = sum(r["dice"] for r in output) / len(output)
    print(f"{args.dataset} {args.split}: {len(output)} images, mean Dice {mean:.5f}")
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output[0]))
            writer.writeheader()
            writer.writerows(output)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
