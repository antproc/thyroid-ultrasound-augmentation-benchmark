from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thyroid_augmentation.extension.common import read_csv, write_json
from thyroid_augmentation.parallel_queue import BUDGETS, limit_memory, snapshot

def worker(model_name, slot, folder):
    import torch
    from thyroid_augmentation.extension.models import build_model
    from thyroid_augmentation.training.runner import DiceLoss, SegmentationManifestDataset

    torch.manual_seed(slot)
    limit_memory(model_name)
    rows = [r for r in read_csv(ROOT / "manifests/metric_v2/splits.csv") if r["dataset"] == "PKTN" and r["split"] == "train"][:2]
    data = SegmentationManifestDataset(rows, ROOT, 224, augmentation="none", training=True)
    examples = [data[i] for i in range(2)]
    images = torch.stack([r[0] for r in examples]).repeat(64, 1, 1, 1).cuda()
    masks = torch.stack([r[1] for r in examples]).repeat(64, 1, 1, 1).cuda()
    model = build_model(ROOT, model_name, pretrained=True).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=.000277414, weight_decay=.000343856)
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = DiceLoss(sigmoid=True, include_background=False)
    write_json(folder / f"ready_{slot}.json", {"pid": os.getpid()})
    deadline = time.monotonic() + 180
    while not (folder / "start.json").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("GPU partner never became ready")
        time.sleep(.1)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    losses = []
    skipped = 0
    for _ in range(12):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss = loss_fn(logits, masks)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss during concurrent smoke")
        finite_gradients = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)

        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if not finite_gradients:
            skipped += 1
            if scaler.get_scale() >= previous_scale:
                raise ValueError("GradScaler did not reduce the scale after overflow")
            del logits, loss
            continue
        losses.append(float(loss.detach()))
        del logits, loss
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        with torch.inference_mode():
            prediction = model(images)
            if prediction.shape != masks.shape or not torch.isfinite(prediction).all():
                raise ValueError("Invalid concurrent FP32 evaluation")
        del prediction
        if len(losses) == 4:
            break
    if len(losses) != 4:
        raise ValueError("Four finite optimizer updates were not achieved")
    torch.cuda.synchronize()
    result = {"model": model_name, "pid": os.getpid(), "batch_size": 128, "optimizer_steps": 4,
              "cap_GiB": BUDGETS[model_name], "peak_allocated_GiB": torch.cuda.max_memory_allocated()/1024**3,
              "peak_reserved_GiB": torch.cuda.max_memory_reserved()/1024**3,
              "started": started, "ended": time.time(), "losses": losses,
              "amp_skipped_warmup_steps": skipped, "final_grad_scale": scaler.get_scale()}
    write_json(folder / f"result_{slot}.json", result)

def parent():
    spec = importlib.util.spec_from_file_location("benchmark_queue", ROOT / "scripts/05_benchmark.py")
    queue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(queue)
    prepared = queue.require_preflight()
    start_hashes = queue.fingerprint()
    if snapshot()["unknown"]:
        raise RuntimeError("Another GPU process is active; smoke will not compete with it.")
    pairs = [("res34_unet", "res34_unet"), ("res34_unet", "deeplabv3plus_resnet50"),
             ("res34_unet", "mit_b2_unet"), ("deeplabv3plus_resnet50", "deeplabv3plus_resnet50")]
    results = []
    for models in pairs:
        with tempfile.TemporaryDirectory(prefix="pair_", dir=queue.OUT) as temporary:
            folder = Path(temporary)
            processes, handles = [], []
            try:
                for slot, name in enumerate(models):
                    handle = (folder / f"worker_{slot}.log").open("wb")
                    handles.append(handle)
                    processes.append(subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()),
                                                       "--worker", name, "--slot", str(slot), "--folder", str(folder)],
                                                      stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT,
                                                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)))
                deadline = time.monotonic() + 300
                while not all((folder / f"ready_{slot}.json").exists() for slot in (0, 1)):
                    if any(p.poll() is not None for p in processes) or time.monotonic() > deadline:
                        raise RuntimeError("Concurrent worker startup failed or timed out")
                    time.sleep(.1)
                write_json(folder / "start.json", {"start": True})
                for p in processes:
                    if p.wait(timeout=300) != 0:
                        raise RuntimeError("Concurrent GPU test failed")
                records = [json.loads((folder / f"result_{slot}.json").read_text()) for slot in (0, 1)]
                overlap = min(r["ended"] for r in records) - max(r["started"] for r in records)
                if overlap <= 0:
                    raise ValueError("Workers did not actually overlap")
                results.append({"models": models, "overlap_seconds": overlap, "workers": records})
                print(f"Concurrent OK: {models}, overlap {overlap:.1f}s", flush=True)
            finally:
                for p in processes:
                    if p.poll() is None:
                        p.terminate()
                    p.wait()
                for handle in handles:
                    handle.close()
                log_dir = queue.OUT / "parallel_smoke_logs"
                log_dir.mkdir(exist_ok=True)
                for slot in (0, 1):
                    path = folder / f"worker_{slot}.log"
                    if path.exists():
                        import shutil
                        shutil.copyfile(path, log_dir / ("_".join(models) + f"_{slot}.log"))
    if start_hashes != queue.fingerprint():
        raise ValueError("Source changed during concurrent GPU tests")
    write_json(queue.OUT / "parallel_preflight.json", {"status": "passed", "source_hashes": start_hashes,
               "versions": prepared["versions"], "pairs": results, "production_training_started": False})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=tuple(BUDGETS))
    parser.add_argument("--slot", type=int, choices=(0, 1))
    parser.add_argument("--folder", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.slot is None or args.folder is None:
            parser.error("Worker requires slot and folder")
        worker(args.worker, args.slot, args.folder)
    else:
        parent()
