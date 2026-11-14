#!/usr/bin/env python3
"""
Fine-tune a pretrained HuggingFace model (e.g., SmolLM-135M) on our chat data.
Same CLI and data format as the from-scratch model.
"""
import sys
import os
import math
import time
import yaml
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset import create_dataloaders, DataConfig
from src.tokenizer.tokenizer import ChatTokenizer, SPECIAL_TOKEN_MAP


@dataclass
class PretrainedFinetuneConfig:
    model_name: str = "HuggingFaceTB/SmolLM-135M"
    learning_rate: float = 5e-5
    min_lr: float = 5e-6
    weight_decay: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    max_steps: int = 2000
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-4
    eval_interval: int = 100
    save_interval: int = 500
    log_interval: int = 20
    sample_interval: int = 200
    batch_size: int = 8
    grad_accum: int = 4
    mixed_precision: str = "bf16"
    compile: bool = False
    gradient_checkpointing: bool = True
    seed: int = 42
    output_dir: str = "checkpoints/finetune_pretrained"
    train_data: str = "data/processed/train.jsonl"
    val_data: str = "data/processed/val.jsonl"
    freeze_embeddings: bool = False
    freeze_layers: int = 0
    max_seq_len: int = 512
    use_flash_attention: bool = True


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


def get_lr_scheduler(optimizer: AdamW, config: PretrainedFinetuneConfig) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)
        progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
        return config.min_lr / config.learning_rate + (1 - config.min_lr / config.learning_rate) * 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: Optional[GradScaler],
    step: int,
    config: PretrainedFinetuneConfig,
    path: str,
    best_val_loss: Optional[float] = None,
    patience_counter: int = 0,
):
    os.makedirs(path, exist_ok=True)
    
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
    
    torch.save(checkpoint, os.path.join(path, f"step_{step}.pt"))
    
    latest_path = os.path.join(os.path.dirname(path), "latest.pt")
    torch.save(checkpoint, latest_path)
    
    if best_val_loss is not None:
        best_path = os.path.join(path, "best_model.pt")
        torch.save({"model": model.state_dict(), "config": asdict(config), "val_loss": best_val_loss}, best_path)
    
    # Also save in HF format for easy loading
    hf_path = os.path.join(path, "hf_model")
    model.save_pretrained(hf_path)
    
    print(f"Checkpoint saved: step {step}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[AdamW] = None,
    scheduler: Optional[LambdaLR] = None,
    scaler: Optional[GradScaler] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[int, Optional[float], int]:
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint["model"])
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
def evaluate(model: nn.Module, val_loader, device: torch.device, amp_dtype: torch.dtype) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
        
        n_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
    
    model.train()
    return total_loss / max(1, total_tokens)


@torch.no_grad()
def generate_sample(
    model: nn.Module,
    hf_tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    eos_token_id: int = None,
) -> str:
    model.eval()
    
    # Format prompt for HF model (no special tokens in our tokenizer)
    # Our data uses: <|user|>\n{user}\n<|agent|>\n{agent}<|end|>
    # We'll just use the user part
    formatted = f"<|user|>\n{prompt}\n<|agent|>\n"
    
    inputs = hf_tokenizer(formatted, return_tensors="pt").to(device)
    
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        eos_token_id=eos_token_id,
        pad_token_id=hf_tokenizer.pad_token_id,
    )
    
    # Decode only new tokens
    new_tokens = generated[0, inputs["input_ids"].shape[1]:]
    response = hf_tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    model.train()
    return response.strip()


def freeze_layers(model: nn.Module, config: PretrainedFinetuneConfig):
    if config.freeze_embeddings:
        for param in model.get_input_embeddings().parameters():
            param.requires_grad = False
        print("Frozen: input embeddings")
    
    if config.freeze_layers > 0:
        # For SmolLM/GPT-style models, layers are in model.model.layers
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            layers = model.transformer.h
        else:
            print("Warning: Could not find transformer layers to freeze")
            return
        
        for i in range(min(config.freeze_layers, len(layers))):
            for param in layers[i].parameters():
                param.requires_grad = False
        print(f"Frozen: first {config.freeze_layers} transformer layers")
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


def load_pretrained_model(config: PretrainedFinetuneConfig, device: torch.device):
    """Load and configure pretrained HF model."""
    print(f"Loading {config.model_name}...")
    
    # Load tokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if hf_tokenizer.pad_token is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token
    
    # Add our special tokens
    special_tokens = {
        "pad_token": "<|pad|>",
        "additional_special_tokens": ["<|user|>", "<|agent|>", "<|end|>"],
    }
    hf_tokenizer.add_special_tokens(special_tokens)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16,
        use_flash_attention_2=config.use_flash_attention,
        trust_remote_code=True,
    )
    
    # Resize embeddings for new tokens
    model.resize_token_embeddings(len(hf_tokenizer))
    
    # Enable gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    model.to(device)
    
    print(f"Model loaded: {config.model_name}")
    print(f"Vocab size: {len(hf_tokenizer)}")
    print(f"Max seq len: {config.max_seq_len}")
    
    return model, hf_tokenizer


def finetune_pretrained(config: PretrainedFinetuneConfig):
    """Main fine-tuning loop for pretrained model."""
    set_seed(config.seed)
    device = get_device()
    amp_dtype = get_amp_dtype(config.mixed_precision)
    
    print(f"Device: {device}")
    print(f"Mixed precision: {config.mixed_precision}")
    
    # Load model and HF tokenizer
    model, hf_tokenizer = load_pretrained_model(config, device)
    
    # Freeze layers if configured
    freeze_layers(model, config)
    
    if config.compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
    
    # Create data loaders using our ChatTokenizer for data processing
    # But we need to handle the tokenization differently for HF model
    chat_tokenizer = ChatTokenizer()
    
    train_loader, val_loader = create_dataloaders(
        config.train_data,
        chat_tokenizer,
        DataConfig(max_seq_len=config.max_seq_len, seed=config.seed),
        config.batch_size,
        val_data_path=config.val_data,
    )
    
    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )
    
    # Scheduler
    scheduler = get_lr_scheduler(optimizer, config)
    
    # Grad scaler
    scaler = GradScaler(enabled=(config.mixed_precision == "fp16"))
    
    # Resume
    start_step = 0
    best_val_loss = float("inf")
    patience_counter = 0
    
    latest_ckpt = os.path.join(config.output_dir, "latest.pt")
    if os.path.exists(latest_ckpt):
        start_step, best_val_loss, patience_counter = load_checkpoint(
            latest_ckpt, model, optimizer, scheduler, scaler, device
        )
    
    # Training loop
    model.train()
    step = start_step
    epoch = 0
    
    print(f"Starting fine-tuning from step {step}")
    print(f"Max steps: {config.max_steps}")
    print(f"Early stopping patience: {config.early_stopping_patience}")
    
    running_loss = 0.0
    running_tokens = 0
    log_start_time = time.time()
    
    sample_prompts = [
        "what is the capital of france?",
        "what is the capital of germany?",
        "what is the largest planet?",
        "what is the speed of light?",
    ]
    
    # Get EOS token ID for generation
    eos_token_id = hf_tokenizer.convert_tokens_to_ids("<|end|>")
    if eos_token_id == hf_tokenizer.unk_token_id:
        eos_token_id = hf_tokenizer.eos_token_id
    
    while step < config.max_steps:
        epoch += 1
        
        for batch in train_loader:
            if step >= config.max_steps:
                break
            
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / config.grad_accum
            
            scaler.scale(loss).backward()
            
            if (step + 1) % config.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            
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
                
                running_loss = 0.0
                running_tokens = 0
                log_start_time = time.time()
            
            # Evaluation with early stopping
            if step % config.eval_interval == 0:
                val_loss = evaluate(model, val_loader, device, amp_dtype)
                print(f"  Val Loss: {val_loss:.4f} (best: {best_val_loss:.4f}, patience: {patience_counter}/{config.early_stopping_patience})")
                
                if val_loss < best_val_loss - config.early_stopping_min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    save_checkpoint(model, optimizer, scheduler, scaler, step, config, config.output_dir, best_val_loss, patience_counter)
                    print(f"  New best val loss: {best_val_loss:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= config.early_stopping_patience:
                        print(f"Early stopping triggered at step {step}")
                        save_checkpoint(model, optimizer, scheduler, scaler, step, config, config.output_dir, best_val_loss, patience_counter)
                        return
            
            # Sample generation
            if step % config.sample_interval == 0:
                print(f"  Samples at step {step}:")
                for prompt in sample_prompts:
                    response = generate_sample(
                        model, hf_tokenizer, prompt, device,
                        max_new_tokens=64, temperature=0.7, top_k=50, top_p=0.9,
                        repetition_penalty=1.1, eos_token_id=eos_token_id
                    )
                    print(f"    Q: {prompt}")
                    print(f"    A: {response}")
            
            # Regular checkpoint
            if step % config.save_interval == 0:
                save_checkpoint(model, optimizer, scheduler, scaler, step, config, config.output_dir, best_val_loss, patience_counter)
    
    # Final save
    save_checkpoint(model, optimizer, scheduler, scaler, step, config, config.output_dir, best_val_loss, patience_counter)
    print("Fine-tuning complete!")


def load_config(config_path: str = "configs/model_config.yaml") -> PretrainedFinetuneConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    # Get pretrained finetune config or fall back to finetune config
    ft_cfg = cfg.get("finetune_pretrained", cfg.get("finetune", {}))
    train_cfg = cfg.get("training", {})
    ft_cfg.setdefault("batch_size", train_cfg.get("batch_size", 8))
    ft_cfg.setdefault("grad_accum", train_cfg.get("grad_accum", 4))
    ft_cfg.setdefault("mixed_precision", train_cfg.get("mixed_precision", "bf16"))
    return PretrainedFinetuneConfig(**ft_cfg)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune pretrained HF model")
    parser.add_argument("--config", default="configs/model_config.yaml", help="Config file path")
    parser.add_argument("--model", help="HF model name (e.g., HuggingFaceTB/SmolLM-135M)")
    parser.add_argument("--train-data", help="Training data path")
    parser.add_argument("--val-data", help="Validation data path")
    parser.add_argument("--output-dir", default="checkpoints/finetune_pretrained", help="Output directory")
    parser.add_argument("--max-steps", type=int, help="Override max steps")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    args = parser.parse_args()
    
    config = load_config(args.config)
    if args.model:
        config.model_name = args.model
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
    
    finetune_pretrained(config)