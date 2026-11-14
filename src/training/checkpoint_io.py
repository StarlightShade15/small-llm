"""Checkpoint file handling: atomic writes, disk guard, step pruning.

A training checkpoint for this model is ~1.4 GB (116M fp32 weights + two AdamW
moment buffers), so a save interval of 100 steps on a container disk fills fast.
Every writer here writes through a sibling temp file and renames it into place,
and the guard refuses a stepped checkpoint when the volume cannot hold it.
"""
import os
import shutil
import tempfile
from typing import Any, Optional

import torch

# Stepped checkpoints kept on disk. `latest.pt` and `best_model.pt` are separate
# and always kept, so this only bounds the rolling history.
KEEP_STEP_CHECKPOINTS = 2

# Headroom over the serialized size so a rename-based write never runs the volume
# to zero mid-save.
DISK_HEADROOM = 1.25


def estimate_save_bytes(obj: Any) -> int:
    """Sum tensor storage in a checkpoint dict without touching tensor data."""
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(estimate_save_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(estimate_save_bytes(v) for v in obj)
    return 0


def free_bytes(path: str) -> int:
    """Free space on the volume holding `path`, walking up to an existing dir."""
    probe = os.path.abspath(path)
    while not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    return shutil.disk_usage(probe).free


def _step_number(file_name: str) -> int:
    stem = os.path.basename(file_name)
    if stem.startswith("step_") and stem.endswith(".pt"):
        try:
            return int(stem[len("step_"):-len(".pt")])
        except ValueError:
            return -1
    return -1


def prune_step_checkpoints(directory: str, keep: int = KEEP_STEP_CHECKPOINTS) -> None:
    """Delete all but the newest `keep` step_*.pt files, plus any torn temp files."""
    if not os.path.isdir(directory):
        return
    try:
        entries = os.listdir(directory)
    except OSError:
        return

    steps = sorted(
        (name for name in entries if _step_number(name) >= 0),
        key=_step_number,
    )
    for name in steps[:-keep] if keep > 0 else steps:
        try:
            os.unlink(os.path.join(directory, name))
        except OSError:
            pass

    for name in entries:
        if name.endswith(".tmp") or name.endswith(".linktmp"):
            try:
                os.unlink(os.path.join(directory, name))
            except OSError:
                pass


def atomic_save(obj: Any, path: str) -> bool:
    """torch.save via a sibling temp file; returns False when the disk guard trips.

    The rename is atomic, so `latest.pt` is either the previous complete
    checkpoint or the new one — a failed write leaves no half file behind.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    required = int(estimate_save_bytes(obj) * DISK_HEADROOM)
    free = free_bytes(directory)
    if required > 0 and free < required:
        print(
            f"[checkpoint] disk guard: {free / 1e9:.1f} GB free in {directory}, "
            f"need ~{required / 1e9:.1f} GB for {os.path.basename(path)}; write skipped"
        )
        return False

    fd, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".", suffix=".tmp", dir=directory
    )
    os.close(fd)
    try:
        torch.save(obj, tmp_path)
        with open(tmp_path, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return True


def _link_or_copy(src: str, dst: str, payload: Any) -> bool:
    """Point `dst` at `src` as a hard link, falling back to a full write.

    latest.pt and step_N.pt hold identical bytes; linking them keeps one copy
    on disk instead of two, which is 1.4 GB saved per checkpoint.
    """
    tmp = dst + ".linktmp"
    try:
        if os.path.exists(tmp):
            os.unlink(tmp)
        os.link(src, tmp)
        os.replace(tmp, dst)
        return True
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return atomic_save(payload, dst)


def save_checkpoint_files(
    checkpoint: dict,
    path: str,
    step: int,
    keep_steps: int = KEEP_STEP_CHECKPOINTS,
    best: Optional[dict] = None,
    best_path: Optional[str] = None,
) -> bool:
    """Prune, then write best/step/latest atomically.

    Best goes first: it is the file the run is judged by and it is the smallest,
    so it survives a volume that cannot hold a full training checkpoint.
    Returns True when the step and latest files were written. A False means the
    disk guard refused the write; training can continue and retry next interval.
    """
    prune_step_checkpoints(path, keep=keep_steps)

    if best is not None and best_path is not None:
        atomic_save(best, best_path)

    step_path = os.path.join(path, f"step_{step}.pt")
    wrote_step = atomic_save(checkpoint, step_path)
    if not wrote_step:
        return False
    _link_or_copy(step_path, os.path.join(path, "latest.pt"), checkpoint)
    return True
