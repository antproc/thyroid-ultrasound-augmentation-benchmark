from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

def save_checkpoint(payload: dict, path: Path, scaler) -> None:
    state = np.random.get_state()
    payload = {**payload, "scaler": scaler.state_dict(),
               "rng": {"python": random.getstate(), "torch": torch.get_rng_state(),
                       "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                       "numpy": (state[0], state[1].tolist(), state[2], state[3], state[4])}}
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)

def restore_state(checkpoint: dict, config: dict, scaler) -> None:
    if checkpoint["config"] != config:
        raise ValueError("Refusing resume with a changed training configuration.")
    scaler.load_state_dict(checkpoint["scaler"])
    rng = checkpoint["rng"]
    random.setstate(rng["python"])
    state = rng["numpy"]
    np.random.set_state((state[0], np.array(state[1], dtype=np.uint32), state[2], state[3], state[4]))
    torch.set_rng_state(rng["torch"].cpu())
    if rng["cuda"]:
        torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
