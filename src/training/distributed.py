"""Small helpers shared by the training entry points for torchrun/DDP."""
import os
from typing import Tuple

import torch
import torch.distributed as dist


def init_distributed() -> Tuple[int, int, int, torch.device]:
    """Initialize one process per GPU when launched through ``torchrun``."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() and dist.is_nccl_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank if world_size > 1 else 0)
        device = torch.device("cuda", local_rank if world_size > 1 else 0)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return rank, world_size, local_rank, device


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def unwrap_model(model):
    """Unwrap DDP and torch.compile wrappers for checkpointing/sampling."""
    while True:
        if hasattr(model, "module"):
            model = model.module
        elif hasattr(model, "_orig_mod"):
            model = model._orig_mod
        else:
            return model


def reduce_loss_totals(loss_sum: float, token_count: int, device: torch.device):
    """Sum token-weighted losses across ranks for an accurate shared metric."""
    values = torch.tensor([loss_sum, float(token_count)], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values[0].item(), int(values[1].item())


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
