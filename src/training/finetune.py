"""
Fine-tuning script for small-lm.
Loads a pretrained checkpoint and fine-tunes on new conversation data.
Lower LR, early stopping on val loss.
Optimized with torch.compile, fused AdamW, efficient logging.
"""
import os
import sys
import math
import time
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
# Support both old and new PyTorch amp APIs
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model.model import SmallLM, ModelConfig, strip_wrapper_prefixes
from src.tokenizer.tokenizer import ChatTokenizer
from src.data.dataset import create_dataloaders, DataConfig
from src.training.checkpoint_io import save_checkpoint_files
from src.training.distributed import (
    cleanup_distributed, init_distributed, is_main_process,
    reduce_loss_totals, unwrap_model,
)


@dataclass
class FinetuneConfig:
    learning_rate: float = 5e-5
    min_lr: float = 5e-6
    weight_decay: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    max_steps: int = 5000
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-4
    eval_interval: int = 100
    save_interval: int = 500
    log_interval: int = 20
    sample_interval: int = 200
    batch_size: int = 4
    grad_accum: int = 8
    mixed_precision: str = "bf16"
    compile: bool = False
    compile_mode: str = "reduce-overhead"
    gradient_checkpointing: bool = True
    seed: int = 42
    output_dir: str = "checkpoints/finetune"
    pretrained_path: str = "checkpoints/pretrain"
    train_data: str = "data/processed/train.jsonl"
    val_data: str = "data/processed/val.jsonl"
    freeze_embeddings: bool = False
    freeze_layers: int = 0
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_amp_dtype(mixed_precision: str) -> torch.dtype:
    if mixed_precision == "bf16":
        return torch.bfloat16
    elif mixed_precision == "fp16":
        return torch.float16
    return torch.float32


def get_lr_scheduler(optimizer: AdamW, config: FinetuneConfig) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)
        progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
        return config.min_lr / config.learning_rate + (1 - config.min_lr / config.learning_rate) * 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    model: SmallLM,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: Optional[GradScaler],
    step: int,
    config: FinetuneConfig,
    path: str,
    best_val_loss: Optional[float] = None,
    patience_counter: int = 0,
    is_best: bool = False,
):
    os.makedirs(path, exist_ok=True)
    
    model = unwrap_model(model)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "config": asdict(config),
        "best_val_loss": best_val_loss,
        "patience_counter": patience_counter,
    }
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()

    best_payload = None
    best_path = os.path.join(path, "best_model.pt")
    if is_best:
        best_payload = {
            "model": model.state_dict(),
            "model_config": model.config.__dict__,
            "val_loss": best_val_loss,
        }

    wrote = save_checkpoint_files(
        checkpoint, path, step, best=best_payload, best_path=best_path
    )

    # model.pt + config.yaml are the plain-load artifacts; a failed write here
    # must not kill a run whose training checkpoint is already on disk.
    try:
        model.save_pretrained(path)
    except (OSError, RuntimeError) as exc:
        print(f"[checkpoint] model.pt write failed: {exc}")

    if wrote:
        print(f"Checkpoint saved: step {step}")
    else:
        print(f"Checkpoint skipped at step {step}: not enough disk space in {path}")


def load_checkpoint(
    path: str,
    model: SmallLM,
    optimizer: Optional[AdamW] = None,
    scheduler: Optional[LambdaLR] = None,
    scaler: Optional[GradScaler] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[int, Optional[float], int]:
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(strip_wrapper_prefixes(checkpoint["model"]))
    step = checkpoint.get("step", 0)
    best_val_loss = checkpoint.get("best_val_loss")
    patience_counter = checkpoint.get("patience_counter", 0)
    
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    
    print(f"Resumed from step {step}" + (f", best_val_loss={best_val_loss:.4f}" if best_val_loss else "") + f", patience={patience_counter}")
    return step, best_val_loss, patience_counter


@torch.no_grad()
def evaluate(model: SmallLM, val_loader, device: torch.device, amp_dtype: torch.dtype) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        
        with get_amp_context(device.type, amp_dtype, amp_dtype != torch.float32):
            _, loss, _ = model(input_ids, labels, return_logits=False)
        
        n_tokens = (labels[:, 1:] != -100).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
    
    model.train()
    total_loss, total_tokens = reduce_loss_totals(total_loss, total_tokens, device)
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


def freeze_layers(model: SmallLM, config: FinetuneConfig):
    if config.freeze_embeddings:
        for param in model.tok_emb.parameters():
            param.requires_grad = False
        print("Frozen: token embeddings")
    
    if config.freeze_layers > 0:
        for i in range(min(config.freeze_layers, len(model.blocks))):
            for param in model.blocks[i].parameters():
                param.requires_grad = False
        print(f"Frozen: first {config.freeze_layers} transformer layers")
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


def finetune(config: FinetuneConfig):
    rank, world_size, local_rank, device = init_distributed()
    main_process = is_main_process()
    set_seed(config.seed)
    amp_dtype = get_amp_dtype(config.mixed_precision)
    
    if main_process:
        print(f"Device: {device} | distributed processes: {world_size}")
        print(f"Mixed precision: {config.mixed_precision} ({amp_dtype})")
        print(f"Compile: {config.compile} ({config.compile_mode if config.compile else 'N/A'})")
    
    if config.wandb_project and main_process:
        import wandb
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name or f"small-lm-ft-{int(time.time())}",
            config=asdict(config),
        )
    
    tokenizer = ChatTokenizer()
    
    if main_process:
        print(f"Loading pretrained model from {config.pretrained_path}...")
    core_model = SmallLM.from_pretrained(config.pretrained_path, device=str(device))
    
    freeze_layers(core_model, config)
    if config.gradient_checkpointing:
        core_model.gradient_checkpointing_enable()
    
    if config.compile and hasattr(torch, "compile"):
        if main_process:
            print(f"Compiling model with torch.compile (mode={config.compile_mode})...")
        core_model = torch.compile(core_model, mode=config.compile_mode)
    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel
        model = DistributedDataParallel(
            core_model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
        )
    else:
        model = core_model
    raw_model = unwrap_model(model)
    
    train_loader, val_loader = create_dataloaders(
        config.train_data,
        tokenizer,
        DataConfig(max_seq_len=raw_model.config.max_seq_len, seed=config.seed),
        config.batch_size,
        val_data_path=config.val_data,
        distributed=world_size > 1,
    )
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
        fused=device.type == "cuda",
    )
    
    scheduler = get_lr_scheduler(optimizer, config)
    scaler = GradScaler(enabled=(config.mixed_precision == "fp16"))
    
    start_step = 0
    best_val_loss = float("inf")
    patience_counter = 0
    
    latest_ckpt = os.path.join(config.output_dir, "latest.pt")
    if os.path.exists(latest_ckpt):
        start_step, best_val_loss, patience_counter = load_checkpoint(
            latest_ckpt, raw_model, optimizer, scheduler, scaler, device
        )
    best_val_loss = float("inf") if best_val_loss is None else best_val_loss
    
    model.train()
    step = start_step
    epoch = 0
    micro_step = step * config.grad_accum
    
    if main_process:
        print(f"Starting fine-tuning from optimizer step {step}")
        print(f"Max steps: {config.max_steps}")
        print(f"Early stopping patience: {config.early_stopping_patience}")
        print(f"Effective global batch size: {config.batch_size * config.grad_accum * world_size}")
    
    running_loss = 0.0
    running_tokens = 0
    log_start_time = time.time()
    optimizer.zero_grad(set_to_none=True)
    
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
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch - 1)
        
        for batch in train_loader:
            if step >= config.max_steps:
                break
            
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            update_step = (micro_step + 1) % config.grad_accum == 0
            sync_context = model.no_sync() if world_size > 1 and not update_step else nullcontext()
            with sync_context:
                with get_amp_context(device.type, amp_dtype, amp_dtype != torch.float32):
                    _, loss, _ = model(input_ids, labels, return_logits=False)
                    loss = loss / config.grad_accum
                scaler.scale(loss).backward()
            micro_step += 1

            n_tokens = (labels[:, 1:] != -100).sum().item()
            running_loss += loss.item() * config.grad_accum * n_tokens
            running_tokens += n_tokens

            if update_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                step += 1
            else:
                continue
            
            if step % config.log_interval == 0:
                logged_loss, logged_tokens = reduce_loss_totals(running_loss, running_tokens, device)
                avg_loss = logged_loss / max(1, logged_tokens)
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - log_start_time
                tokens_per_sec = logged_tokens / elapsed
                if main_process:
                    print(f"Step {step:6d} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | Tokens/s: {tokens_per_sec:.0f}")
                
                if config.wandb_project and main_process:
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
            
            if step % config.eval_interval == 0:
                val_loss = evaluate(model, val_loader, device, amp_dtype)
                if main_process:
                    print(f"  Val Loss: {val_loss:.4f} (best: {best_val_loss:.4f}, patience: {patience_counter}/{config.early_stopping_patience})")
                
                if config.wandb_project and main_process:
                    import wandb
                    wandb.log({"val/loss": val_loss, "val/step": step})
                
                if val_loss < best_val_loss - config.early_stopping_min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if main_process:
                        save_checkpoint(raw_model, optimizer, scheduler, scaler, step, config, config.output_dir, best_val_loss, patience_counter, is_best=True)
                        print(f"  New best val loss: {best_val_loss:.4f}")
                else:
                    patience_counter += 1
                    if config.early_stopping_patience > 0 and patience_counter >= config.early_stopping_patience:
                        if main_process:
                            print(f"Early stopping triggered at step {step}")
                        break
            
            if step % config.sample_interval == 0 and main_process:
                print(f"  Samples at step {step}:")
                for prompt in sample_prompts:
                    response = generate_sample(raw_model, tokenizer, prompt, device)
                    print(f"    Q: {prompt}")
                    print(f"    A: {response}")
            
            if step % config.save_interval == 0 and main_process:
                save_checkpoint(raw_model, optimizer, scheduler, scaler, step, config, config.output_dir, best_val_loss, patience_counter)
        if config.early_stopping_patience > 0 and patience_counter >= config.early_stopping_patience:
            break
    
    if main_process:
        save_checkpoint(raw_model, optimizer, scheduler, scaler, step, config, config.output_dir, best_val_loss, patience_counter)
        print("Fine-tuning complete!")
    
    if config.wandb_project and main_process:
        import wandb
        wandb.finish()
    cleanup_distributed()


def load_config(config_path: str = "configs/model_config.yaml") -> FinetuneConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    ft_cfg = cfg.get("finetune", {})
    train_cfg = cfg.get("training", {})
    ft_cfg.setdefault("batch_size", train_cfg.get("batch_size", 4))
    ft_cfg.setdefault("grad_accum", train_cfg.get("grad_accum", 8))
    ft_cfg.setdefault("mixed_precision", train_cfg.get("mixed_precision", "bf16"))
    ft_cfg.setdefault("compile", train_cfg.get("compile", False))
    ft_cfg.setdefault("compile_mode", train_cfg.get("compile_mode", "reduce-overhead"))
    # Ensure numeric values are proper types
    for key in ("learning_rate", "min_lr", "weight_decay", "beta1", "beta2", "eps", "max_grad_norm", "early_stopping_min_delta"):
        if key in ft_cfg:
            ft_cfg[key] = float(ft_cfg[key])
    for key in ("batch_size", "grad_accum", "warmup_steps", "max_steps", "eval_interval", "save_interval", "log_interval", "sample_interval", "seed", "early_stopping_patience"):
        if key in ft_cfg:
            ft_cfg[key] = int(ft_cfg[key])
    return FinetuneConfig(**ft_cfg)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune small-lm")
    parser.add_argument("--config", default="configs/model_config.yaml", help="Config file path")
    parser.add_argument("--pretrained", help="Path to pretrained model")
    parser.add_argument("--train-data", help="Training data path")
    parser.add_argument("--val-data", help="Validation data path")
    parser.add_argument("--output-dir", default="checkpoints/finetune", help="Output directory")
    parser.add_argument("--max-steps", type=int, help="Override max steps")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--wandb", help="Wandb project name")
    args = parser.parse_args()
    
    config = load_config(args.config)
    if args.pretrained:
        config.pretrained_path = args.pretrained
    if args.train_data:
        config.train_data = args.train_data
    if args.val_data:
        config.val_data = args.val_data
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.lr:
        config.learning_rate = args.lr
    if args.wandb:
        config.wandb_project = args.wandb
    
    finetune(config)
