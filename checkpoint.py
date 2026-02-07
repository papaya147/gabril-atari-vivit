import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.amp import GradScaler
from torch.optim.lr_scheduler import SequentialLR

from device import device


def save(
    path: str,
    epoch: int,
    best_return: float,
    wandb_id: str,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    scheduler: SequentialLR = None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    scheduler_state = None
    if scheduler is not None:
        scheduler_state = scheduler.state_dict()

    checkpoint_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler_state,
        "scaler_state_dict": scaler.state_dict(),
        "best_return": best_return,
        "wandb_id": wandb_id,
        "rng_states": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    torch.save(checkpoint_data, path)


def load(
    path: str,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    scheduler: SequentialLR = None,
) -> Tuple[int, float, str | None]:
    if not os.path.exists(path):
        return 0, -float("inf"), None

    print(f"--> Found checkpoint! Resuming from {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Restore RNGs if available
    if "rng_states" in checkpoint:
        rng = checkpoint["rng_states"]
        torch.set_rng_state(rng["torch"].cpu())
        cuda_states = [state.cpu() for state in rng["cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])

    # Extract metadata
    start_epoch = checkpoint["epoch"] + 1
    best_return = checkpoint.get("best_return", -float("inf"))
    wandb_id = checkpoint.get("wandb_id", None)

    print(f"--> Resumed at Epoch {start_epoch}, Best Return: {best_return:.4f}")
    return start_epoch, best_return, wandb_id
