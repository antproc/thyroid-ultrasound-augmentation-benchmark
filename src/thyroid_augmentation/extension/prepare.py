from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from thyroid_augmentation.extension.common import PREPARED, SCANS, SPLITS, digest, protocol, read_csv, write_csv, write_json

PKTN_SIZE_REPAIRS = frozenset({"pktn_Img_477", "pktn_Img_481"})

def align_mask(mask: np.ndarray, dataset: str, sample: str, image_size: tuple[int, int]) -> tuple[np.ndarray, str]:
    mask_size = (mask.shape[1], mask.shape[0])
    if mask_size == image_size:
        return mask, "none"
    if dataset == "PKTN" and sample in PKTN_SIZE_REPAIRS and mask_size == (768, 576) and image_size == (720, 576):
        resized = Image.fromarray(mask).resize(image_size, resample=Image.Resampling.NEAREST)
        return np.array(resized), "nearest_768x576_to_720x576"
    raise ValueError(f"Unrecognized image/mask misalignment: {dataset}/{sample} {image_size} {mask_size}")

def save_repair_preview(root: Path, sample: str, rgb: Image.Image, mask: np.ndarray) -> None:
    overlay = np.array(rgb)
    foreground = mask > 0
    overlay[foreground] = (.65 * overlay[foreground] + .35 * np.array([255, 40, 50])).astype(np.uint8)
    preview = Image.new("RGB", (rgb.width * 2, rgb.height + 30), "white")
    preview.paste(rgb, (0, 30))
    preview.paste(Image.fromarray(overlay), (rgb.width, 30))
    labels = ImageDraw.Draw(preview)
    labels.text((8, 8), f"{sample}: original image", fill="black")
    labels.text((rgb.width + 8, 8), "Prepared mask: nearest-neighbor width adjustment", fill="black")
    path = root / "qc/extension" / f"{sample}_size_repair.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(path)

def binary_mask(array: np.ndarray, dataset: str) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError("Expected a grayscale annotation.")
    unique = np.unique(array)
    if set(unique).issubset({0, 1}):
        return (array > 0).astype(np.uint8) * 255
    if dataset != "PKTN" and not set(unique).issubset({0, 255}):
        raise ValueError(f"Unexpected nonbinary PNG mask in {dataset}: {unique[:20]}")

    return (array > 127).astype(np.uint8) * 255

def prepare(root: Path) -> dict:
    protocol(root)
    raw_path = root / "manifests/extension/splits_raw.csv"
    rows = read_csv(raw_path)
    manifest, scans, audit, issues, repairs = [], [], [], [], []
    thumbnails: dict[str, list] = {}
    for index, row in enumerate(rows, 1):
        ds, sample = row["dataset"], row["sample_id"]
        short = hashlib.sha256(sample.encode()).hexdigest()[:16]
        out = root / "prepared_data/extension" / ds / f"{short}.png"
        scan = root / "scan_masks/extension" / ds / f"{short}.png"
        with Image.open(row["image_path"]) as source:
            rgb = source.convert("RGB")
        with Image.open(row["mask_path"]) as source:
            mask = binary_mask(np.asarray(source.convert("L")), ds)
        original_mask_size = (mask.shape[1], mask.shape[0])
        try:
            mask, geometry = align_mask(mask, ds, sample, rgb.size)
        except ValueError:
            issues.append({"dataset": ds, "sample_id": sample, "image_size": list(rgb.size),
                           "mask_size": [mask.shape[1], mask.shape[0]], "issue": "Image/mask dimensions differ"})
            continue
        if geometry != "none":
            repairs.append({"dataset": ds, "sample_id": sample, "split": row["split"],
                            "source_mask_size": list(original_mask_size), "prepared_mask_size": list(rgb.size),
                            "method": geometry, "source_modified": False})
            save_repair_preview(root, sample, rgb, mask)
        gray = np.asarray(rgb.convert("L"))
        scan_mask = (gray > 5).astype(np.uint8) * 255
        if not scan_mask.any():
            raise ValueError(f"Empty scan mask: {sample}")
        for path, array in [(out, mask), (scan, scan_mask)]:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.png")
            Image.fromarray(array).save(tmp)
            tmp.replace(path)
        prepared = {**row, "mask_path": out.relative_to(root).as_posix()}
        manifest.append(prepared)
        scans.append({"dataset": ds, "sample_id": sample, "scan_mask_path": scan.relative_to(root).as_posix(),
                      "image_path": row["image_path"], "width": rgb.width, "height": rgb.height,
                      "scan_pixels": int(np.count_nonzero(scan_mask)), "scan_fraction": float((scan_mask > 0).mean()), "status": "created"})
        audit.append({"dataset": ds, "sample_id": sample, "split": row["split"], "width": rgb.width, "height": rgb.height,
                      "source_image_sha256": digest(Path(row["image_path"])), "source_mask_sha256": digest(Path(row["mask_path"])),
                      "mask_sha256": digest(out), "scan_sha256": digest(scan),
                      "source_mask_width": original_mask_size[0], "source_mask_height": original_mask_size[1],
                      "geometry_adjustment": geometry,
                      "foreground_pixels": int(np.count_nonzero(mask)), "mask_fraction": float((mask > 0).mean()),
                      "binarization": ">127 for JPEG" if ds == "PKTN" else "binary PNG labels"})
        samples = thumbnails.setdefault(ds, [])
        if len(samples) < 9:
            overlay = np.array(rgb)
            fg = mask > 0
            overlay[fg] = (.5 * overlay[fg] + .5 * np.array([255, 40, 50])).astype(np.uint8)
            tile = Image.new("RGB", (320, 270), "white")
            thumb = Image.fromarray(overlay)
            thumb.thumbnail((320, 240))
            tile.paste(thumb, ((320 - thumb.width) // 2, 0))
            ImageDraw.Draw(tile).text((5, 243), sample[:44], fill="black")
            samples.append(tile)
        if index % 250 == 0 or index == len(rows):
            print(f"Prepared {index}/{len(rows)} image/mask pairs", flush=True)
    for ds, tiles in thumbnails.items():
        sheet = Image.new("RGB", (960, 810), "white")
        for n, tile in enumerate(tiles):
            sheet.paste(tile, (n % 3 * 320, n // 3 * 270))
        path = root / "qc/extension" / f"{ds}_overlays.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(path)
    write_json(root / "manifests/extension/preparation_issues.json",
               {"status": "blocked" if issues else "passed", "issues": issues,
                "checked_samples": len(rows), "aligned_samples": len(manifest), "resolved_size_mismatches": repairs})
    if issues:

        write_csv(root / "manifests/extension/splits_diagnostic.csv", manifest)
        write_csv(root / "manifests/extension/scans_diagnostic.csv", scans)
        write_csv(root / "manifests/extension/audit_diagnostic.csv", audit)
        raise ValueError(f"Preparation blocked by {len(issues)} misaligned pairs; see preparation_issues.json. No ready receipt written.")
    write_csv(root / SPLITS, manifest)
    write_csv(root / SCANS, scans)
    audit_path = root / "manifests/extension/preparation_audit.csv"
    write_csv(audit_path, audit)
    info = {"status": "prepared", "samples": len(rows), "raw_manifest_sha256": digest(raw_path),
            "manifest_sha256": digest(root / SPLITS), "scan_manifest_sha256": digest(root / SCANS),
            "audit_sha256": digest(audit_path), "source_data_modified": False,
            "geometry_repairs": repairs,
            "geometry_assumption": "For two known PKTN pairs only, scale the binary mask proportionally to image dimensions using nearest neighbor; original coordinate mapping is not independently established.",
            "counts": dict(Counter(f"{r['dataset']}/{r['split']}" for r in rows)),
            "empty_target_masks": sum(int(x["foreground_pixels"]) == 0 for x in audit),
            "scan_mask_method": "gray > 5, identical to the original TN3K/DDTI procedure",
            "duplicate_policy": "No deduplication; frozen source splits retained."}
    write_json(root / PREPARED, info)
    return info
