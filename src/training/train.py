"""
Training script for small-lm.
Features: mixed precision (bf16/fp16), AdamW, cosine LR with warmup,
grad clipping, checkpointing, console logging, periodic sample generation, resume support.
Optimized for RTX 5070 Ti 16GB with torch.compile.
"""
import os
import sys
import math
import time
import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
# Support both old and new PyTorch amp APIs
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model.model import SmallLM, ModelConfig
from src.tokenizer.tokenizer import ChatTokenizer
from src.data.dataset import create_dataloaders, DataConfig


@dataclass
class TrainConfig:
    batch_size: int = 32
    grad_accum: int = 2
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    max_steps: int = 15000
    eval_interval: int = 500
    save_interval: int = 1000
    log_interval: int = 25
    sample_interval: int = 500
    mixed_precision: str = "bf16"  # "bf16", "fp16", "fp32"
    compile: bool = True
    compile_mode: str = "reduce-overhead"  # "default", "reduce-overhead", "max-autotune"
    gradient_checkpointing: bool = False
    seed: int = 42
    output_dir: str = "checkpoints/pretrain"
    resume_from: Optional[str] = None
    # Wandb
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None


def set_seed(seed: int):
    """Set all random seeds."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For reproducibility with torch.compile
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_lr_scheduler(optimizer: AdamW, config: TrainConfig) -> LambdaLR:
    """Cosine LR schedule with warmup."""
    def lr_lambda(step: int) -> float:
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)
        progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
        return config.min_lr / config.learning_rate + (1 - config.min_lr / config.learning_rate) * 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_amp_dtype(mixed_precision: str) -> torch.dtype:
    """Get autocast dtype from mixed_precision string."""
    if mixed_precision == "bf16":
        return torch.bfloat16
    elif mixed_precision == "fp16":
        return torch.float16
    return torch.float32


def save_checkpoint(
    model: SmallLM,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: Optional[GradScaler],
    step: int,
    config: TrainConfig,
    path: str,
    best_val_loss: Optional[float] = None,
):
    """Save training checkpoint."""
    os.makedirs(path, exist_ok=True)
    
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "config": asdict(config),
        "best_val_loss": best_val_loss,
    }
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    
    torch.save(checkpoint, os.path.join(path, f"step_{step}.pt"))
    
    # Also save latest
    latest_path = os.path.join(os.path.dirname(path), "latest.pt")
    torch.save(checkpoint, latest_path)
    
    print(f"Checkpoint saved: step {step}")


def load_checkpoint(
    path: str,
    model: SmallLM,
    optimizer: Optional[AdamW] = None,
    scheduler: Optional[LambdaLR] = None,
    scaler: Optional[GradScaler] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[int, Optional[float]]:
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint["model"])
    step = checkpoint.get("step", 0)
    best_val_loss = checkpoint.get("best_val_loss")
    
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    
    print(f"Resumed from step {step}" + (f", best_val_loss={best_val_loss:.4f}" if best_val_loss else ""))
    return step, best_val_loss


@torch.no_grad()
def evaluate(model: SmallLM, val_loader, device: torch.device, amp_dtype: torch.dtype) -> float:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        
        with get_amp_context(device.type, amp_dtype, amp_dtype != torch.float32):
            _, loss, _ = model(input_ids, labels)
        
        n_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
    
    model.train()
    return total_loss / max(1, total_tokens)


@torch.no_grad()
def generate_sample(
    model: SmallLM,
    tokenizer: ChatTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
) -> str:
    """Generate a sample response for logging."""
    model.eval()
    
    input_ids = tokenizer.encode_user(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    generated = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=tokenizer.end_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    new_tokens = generated[0, len(input_ids[0]):].tolist()
    response = tokenizer.decode(new_tokens, skip_special=True)
    
    model.train()
    return response.strip()


def get_amp_context(device_type: str, dtype: torch.dtype, enabled: bool):
    """Get autocast context manager compatible with old and new PyTorch."""
    if enabled and dtype != torch.float32:
        try:
            # New API (PyTorch 2.4+)
            return torch.amp.autocast(device_type=device_type, dtype=dtype)
        except TypeError:
            # Old API
            return torch.cuda.amp.autocast(dtype=dtype)
    # No-op context manager
    from contextlib import nullcontext
    return nullcontext()


def train(config: TrainConfig):
    """Main training loop."""
    set_seed(config.seed)
    device = get_device()
    amp_dtype = get_amp_dtype(config.mixed_precision)
    
    print(f"Device: {device}")
    print(f"Mixed precision: {config.mixed_precision}")
    print(f"AMP dtype: {amp_dtype}")
    print(f"Compile: {config.compile} ({config.compile_mode if config.compile else 'N/A'})")
    
    # Wandb init
    if config.wandb_project:
        import wandb
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name or f"small-lm-{int(time.time())}",
            config=asdict(config),
        )
    
    # Load tokenizer
    tokenizer = ChatTokenizer()
    
    # Create model
    model_config = ModelConfig()
    model = SmallLM(model_config).to(device)
    
    if config.compile and hasattr(torch, "compile"):
        print(f"Compiling model with torch.compile (mode={config.compile_mode})...")
        model = torch.compile(model, mode=config.compile_mode)
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        "data/processed/train.jsonl",
        tokenizer,
        DataConfig(max_seq_len=model_config.max_seq_len, seed=config.seed),
        config.batch_size,
        val_data_path="data/processed/val.jsonl" if os.path.exists("data/processed/val.jsonl") else None,
    )
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
        fused=True,  # Fused AdamW (PyTorch 2.0+)
    )
    
    # Scheduler
    scheduler = get_lr_scheduler(optimizer, config)
    
    # Grad scaler for mixed precision
    scaler = GradScaler(enabled=(config.mixed_precision == "fp16"))
    
    # Resume
    start_step = 0
    best_val_loss = float("inf")
    
    if config.resume_from:
        start_step, best_val_loss = load_checkpoint(
            config.resume_from, model, optimizer, scheduler, scaler, device
        )
    elif os.path.exists(os.path.join(config.output_dir, "latest.pt")):
        start_step, best_val_loss = load_checkpoint(
            os.path.join(config.output_dir, "latest.pt"), model, optimizer, scheduler, scaler, device
        )
    
    # Training loop
    model.train()
    step = start_step
    epoch = 0
    
    print(f"Starting training from step {step}")
    print(f"Max steps: {config.max_steps}")
    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Effective batch size: {config.batch_size * config.grad_accum}")
    
    running_loss = 0.0
    running_tokens = 0
    log_start_time = time.time()
    
    sample_prompts = [
        "what is the capital of france?",
        "what is the capital of germany?",
        "what is the largest planet?",
        "what is the speed of light?",
        "who wrote hamlet?",
        "what is the boiling point of water?",
    ]
    
    while step < config.max_steps:
        epoch += 1
        
        for batch in train_loader:
            if step >= config.max_steps:
                break
            
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            
            # Forward with mixed precision
            with get_amp_context(device.type, amp_dtype, amp_dtype != torch.float32):
                _, loss, _ = model(input_ids, labels)
                loss = loss / config.grad_accum
            
            # Backward
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (step + 1) % config.grad_accum == 0:
                # Unscale for gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            
            # Track loss
            n_tokens = (labels != -100).sum().item()
            running_loss += loss.item() * config.grad_accum * n_tokens
            running_tokens += n_tokens
            
            step += 1
            
            # Logging
            if step % config.log_interval == 0:
                avg_loss = running_loss / max(1, running_tokens)
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - log_start_time
                tokens_per_sec = running_tokens / elapsed
                
                print(f"Step {step:6d} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | Tokens/s: {tokens_per_sec:.0f}")
                
                if config.wandb_project:
                    import wandb
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/lr": lr,
                        "train/tokens_per_sec": tokens_per_sec,
                        "train/step": step,
                    })
                
                running_loss = 0.0
                running_tokens = 0
                log_start_time = time.time()
            
            # Evaluation
            if step % config.eval_interval == 0:
                val_loss = evaluate(model, val_loader, device, amp_dtype)
                print(f"  Val Loss: {val_loss:.4f}")
                
                if config.wandb_project:
                    import wandb
                    wandb.log({"val/loss": val_loss, "val/step": step})
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, scheduler, scaler, step, config, config.output_dir, best_val_loss)
                    print(f"  New best val loss: {best_val_loss:.4f}")
            
            # Sample generation
            if step % config.sample_interval == 0:
                print(f"  Samples at step {step}:")
                for prompt in sample_prompts:
                    response = generate_sample(model, tokenizer, prompt, device)
                    print(f"    Q: {prompt}")
                    print(f"    A: {response}")
            
            # Checkpoint
            if step % config.save_interval == 0:
                save_checkpoint(model, optimizer, scheduler, scaler, step, config, config.output_dir, best_val_loss)
    
    # Final save
    save_checkpoint(model, optimizer, scheduler, scaler, step, config, config.output_dir, best_val_loss)
    print("Training complete!")
    
    if config.wandb_project:
        import wandb
        wandb.finish()


def load_config(config_path: str = "configs/model_config.yaml") -> TrainConfig:
    """Load training config from YAML."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    train_cfg = cfg.get("training", {})
    # Ensure numeric values are proper types (YAML may parse scientific notation as strings)
    for key in ("learning_rate", "min_lr", "weight_decay", "beta1", "beta2", "eps", "max_grad_norm"):
        if key in train_cfg:
            train_cfg[key] = float(train_cfg[key])
    for key in ("batch_size", "grad_accum", "warmup_steps", "max_steps", "eval_interval", "save_interval", "log_interval", "sample_interval", "seed"):
        if key in train_cfg:
            train_cfg[key] = int(train_cfg[key])
    return TrainConfig(**train_cfg)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train small-lm")
    parser.add_argument("--config", default="configs/model_config.yaml", help="Config file path")
    parser.add_argument("--resume", help="Resume from checkpoint")
    parser.add_argument("--output-dir", default="checkpoints/pretrain", help="Output directory")
    parser.add_argument("--max-steps", type=int, help="Override max steps")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--wandb", help="Wandb project name")
    args = parser.parse_args()
    
    config = load_config(args.config)
    if args.resume:
        config.resume_from = args.resume
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.wandb:
        config.wandb_project = args.wandb
    
    train(config)