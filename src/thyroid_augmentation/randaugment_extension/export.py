from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path

from thyroid_augmentation.extension.common import SPLITS, digest, read_csv, write_csv, write_json
from thyroid_augmentation.extension.evaluate import infer_rows, run_id, validate_export, verify_backup
from thyroid_augmentation.training.metrics import METRIC_PROTOCOL, require_protocol

OUTPUT = "results/extension_randaugment"
BACKUP = "results/extension_randaugment_backup"

def export_best(root: Path, config: dict, summary: dict) -> dict:
    import torch
    from thyroid_augmentation.augmentations.factory import _parse_randaugment_config
    from thyroid_augmentation.extension.models import build_model
    from thyroid_augmentation.randaugment_extension.pipeline import fingerprint

    identity = run_id(config)
    out, backup = root / OUTPUT / "per_image" / identity, root / BACKUP / "per_image" / identity
    checkpoint_path = Path(summary["checkpoint"])
    manifest = read_csv(root / SPLITS)
    sources = {split: [r for r in manifest if r["dataset"] == config["dataset"] and r["split"] == split]
               for split in ("val", "test")}
    hashes = {"checkpoint_sha256": digest(checkpoint_path), "manifest_sha256": digest(root / SPLITS),
              "log_sha256": digest(Path(summary["log"])),
              "summary_sha256": digest(Path(summary["log"]).with_name("summary.json")),
              "pipeline_sha256": hashlib.sha256(json.dumps(fingerprint(root), sort_keys=True).encode()).hexdigest()}
    if (out / "complete.json").exists():
        receipt = json.loads((out / "complete.json").read_text())
        if receipt["source_hashes"] != hashes:
            raise ValueError("RandAugment export inputs changed; refusing overwrite.")
        for split, expected in sources.items():
            rows = read_csv(out / f"{split}.csv")
            validate_export(rows, expected, config, split)
            if any(r["strategy"] != "randaugment" for r in rows):
                raise ValueError("Wrong strategy label in RandAugment export.")
        verify_backup(out, backup, receipt)
        return receipt
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA for RandAugment inference.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if config.get("metric_protocol") == METRIC_PROTOCOL:
        require_protocol(checkpoint, str(checkpoint_path))
        require_protocol(summary, "RA export summary")
    for field in ("phase", "dataset", "model", "augmentation", "seed"):
        if checkpoint["config"][field] != config[field]:
            raise ValueError(f"Checkpoint identity mismatch: {field}")
    model = build_model(root, config["model"], pretrained=False).cuda()
    model.load_state_dict(checkpoint["model"], strict=True)
    aggregates = {}
    policy = _parse_randaugment_config(config["augmentation"])
    for split, expected in sources.items():
        rows = infer_rows(root, config, model, expected, split, config["batch_size"])
        for row in rows:
            row.update(strategy="randaugment", randaugment_n=policy.num_ops, randaugment_m=policy.magnitude,
                       randaugment_bins=policy.magnitude_denominator + 1)
        aggregates[split] = validate_export(rows, expected, config, split)
        write_csv(out / f"{split}.csv", rows)
        validate_export(read_csv(out / f"{split}.csv"), expected, config, split)
    metadata = {"run_id": identity, "config": checkpoint["config"], "source_hashes": hashes,
                "selected_epoch": int(checkpoint["epoch"]) + 1, "training_selection_dice": summary["best_dice"],
                "policy": {"operations": list(policy.operations), "n": policy.num_ops, "m": policy.magnitude,
                           "bins": policy.magnitude_denominator + 1, "sampling": "with replacement"},
                "metric_definition": "Per-image sigmoid(logits)>0.5; Dice/IoU from integer TP/FP/FN/TN; epsilon=1e-7",
                "validation_selection": checkpoint.get("metric_protocol", "legacy_raw_logits_dice"),
                "precision": "float32", "inference_batch_size": config["batch_size"], "aggregates": aggregates,
                "preprocessing": "Unchanged build_article_eval_transform; prepared binary masks; frozen split",
                "versions": {name: importlib.metadata.version(name) for name in
                             ("torch", "numpy", "albumentations", "monai", "torchmetrics", "segmentation-models-pytorch")}}
    write_json(out / "metadata.json", metadata)
    write_json(out / "training_summary.json", summary)
    shutil.copyfile(summary["log"], out / "train_log.csv")
    files = {name: digest(out / name) for name in ("val.csv", "test.csv", "metadata.json", "training_summary.json", "train_log.csv")}
    receipt = {"status": "complete_exports_and_local_backup_verified", "source_hashes": hashes,
               "files": files, "aggregates": aggregates, "backup": backup.relative_to(root).as_posix(),
               "backup_on_same_disk": True}

    backup.mkdir(parents=True, exist_ok=True)
    for name, sha in files.items():
        temporary = backup / (name + ".tmp")
        shutil.copyfile(out / name, temporary)
        if digest(temporary) != sha:
            raise ValueError("RandAugment backup checksum mismatch.")
        temporary.replace(backup / name)
    write_json(out / "complete.json", receipt)
    verify_backup(out, backup, receipt)
    del model, checkpoint
    torch.cuda.empty_cache()
    return receipt
