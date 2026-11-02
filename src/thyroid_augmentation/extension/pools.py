from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from contextlib import contextmanager
from unittest.mock import patch

from thyroid_augmentation.extension.experiment import AUGMENTATIONS

def ranked_pools(records):
    groups = defaultdict(list)
    for r in records:
        score = float(r["best_dice"])
        if not math.isfinite(score) or not 0 <= score <= 1 or r["augmentation"] not in AUGMENTATIONS:
            raise ValueError("Invalid single result.")
        groups[r["dataset"], r["model"], r["augmentation"]].append(r)
    means = {}
    for key, values in groups.items():
        if sorted(r["seed"] for r in values) != [0, 1, 2]:
            raise ValueError("Single ranking requires exactly seeds 0,1,2.")
        means[key] = sum(r["best_dice"] for r in values) / 3
    pools = []
    for ds, model in sorted({k[:2] for k in means}):
        if {k[2] for k in means if k[:2] == (ds, model)} != set(AUGMENTATIONS):
            raise ValueError("All 15 single conditions are required.")
        order = sorted(AUGMENTATIONS[1:], key=lambda a: (-means[ds, model, a], a))
        pools.extend({"dataset": ds, "model": model, "top_n": n, "operations": ["identity"] + order[:n],
                      "operations_per_image": 2, "sample_with_replacement": True, "selection_uses": "validation"}
                     for n in range(2, 15))
    return pools

def short_name(value):
    if not value.startswith("trivial:"):
        raise ValueError("Only explicit TA policies may use the short path adapter.")
    n = len(value.split(":", 1)[1].split(",")) - 1
    return f"top{n:02d}_{hashlib.sha256(value.encode()).hexdigest()[:12]}"

def run_directory(root, config, checkpoint=False):
    return root / ("checkpoints" if checkpoint else "runs") / config["phase"] / config["dataset"] / config["model"] / short_name(config["augmentation"]) / f"seed_{config['seed']}"

@contextmanager
def short_paths():
    from thyroid_augmentation.training import runner
    old = runner._safe_name
    with patch.object(runner, "_safe_name", side_effect=lambda value: short_name(value) if value.startswith("trivial:") else old(value)):
        yield
