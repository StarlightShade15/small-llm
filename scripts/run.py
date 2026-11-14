#!/usr/bin/env python3
"""
Main entry point for small-lm project.
Run various stages: setup, tokenizer, data, train, finetune, chat.
"""
import sys
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def run_module(module: str, args: list = None):
    """Run a Python module with arguments."""
    cmd = [sys.executable, "-m", module]
    if args:
        cmd.extend(args)
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="small-lm project runner")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Setup
    setup_parser = subparsers.add_parser("setup", help="Install dependencies")
    
    # Tokenizer
    tok_parser = subparsers.add_parser("tokenizer", help="Train tokenizer")
    tok_parser.add_argument("--vocab-size", type=int, default=16384)
    tok_parser.add_argument("--corpus-dir", default="data/raw")
    tok_parser.add_argument("--output-dir", default="data/processed/tokenizer")
    
    # Data generation
    data_parser = subparsers.add_parser("generate-data", help="Generate synthetic Q&A data")
    data_parser.add_argument("--input", "-i", help="Input facts file (CSV/JSON/JSONL)")
    data_parser.add_argument("--output", "-o", default="data/raw/synthetic_chat.jsonl")
    data_parser.add_argument("--num-facts", "-n", type=int, default=1000)
    data_parser.add_argument("--augmentations", "-a", type=int, default=3)
    data_parser.add_argument("--train-ratio", type=float, default=0.95)
    
    # Train
    train_parser = subparsers.add_parser("train", help="Train from scratch")
    train_parser.add_argument("--config", default="configs/model_config.yaml")
    train_parser.add_argument("--resume", help="Resume from checkpoint")
    train_parser.add_argument("--output-dir", default="checkpoints/pretrain")
    train_parser.add_argument("--max-steps", type=int)
    train_parser.add_argument("--batch-size", type=int)
    
    # Finetune
    ft_parser = subparsers.add_parser("finetune", help="Fine-tune on new data")
    ft_parser.add_argument("--config", default="configs/model_config.yaml")
    ft_parser.add_argument("--pretrained", default="checkpoints/pretrain")
    ft_parser.add_argument("--train-data", default="data/processed/train.jsonl")
    ft_parser.add_argument("--val-data", default="data/processed/val.jsonl")
    ft_parser.add_argument("--output-dir", default="checkpoints/finetune")
    ft_parser.add_argument("--max-steps", type=int)
    ft_parser.add_argument("--lr", type=float)
    
    # Finetune pretrained
    ft_pretrained_parser = subparsers.add_parser("finetune-pretrained", help="Fine-tune HF model")
    ft_pretrained_parser.add_argument("--config", default="configs/model_config.yaml")
    ft_pretrained_parser.add_argument("--model", default="HuggingFaceTB/SmolLM-135M")
    ft_pretrained_parser.add_argument("--train-data", default="data/processed/train.jsonl")
    ft_pretrained_parser.add_argument("--val-data", default="data/processed/val.jsonl")
    ft_pretrained_parser.add_argument("--output-dir", default="checkpoints/finetune_pretrained")
    ft_pretrained_parser.add_argument("--max-steps", type=int)
    ft_pretrained_parser.add_argument("--lr", type=float)
    
    # Chat
    chat_parser = subparsers.add_parser("chat", help="Interactive chat")
    chat_parser.add_argument("--checkpoint", "-c", default="checkpoints/finetune/best_model.pt")
    chat_parser.add_argument("--config", default="configs/model_config.yaml")
    chat_parser.add_argument("--device", choices=["cuda", "cpu", "mps", "auto"], default="auto")
    chat_parser.add_argument("--temperature", "-t", type=float, default=0.7)
    chat_parser.add_argument("--top-k", "-k", type=int, default=50)
    chat_parser.add_argument("--top-p", "-p", type=float, default=0.9)
    chat_parser.add_argument("--max-tokens", "-m", type=int, default=64)
    chat_parser.add_argument("--strict", "-s", action="store_true")
    chat_parser.add_argument("--prompt", help="Single prompt (non-interactive)")
    
    # Full pipeline
    pipeline_parser = subparsers.add_parser("pipeline", help="Run full pipeline")
    pipeline_parser.add_argument("--skip-train", action="store_true", help="Skip from-scratch training")
    pipeline_parser.add_argument("--pretrained-model", default="HuggingFaceTB/SmolLM-135M")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Add project root to PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + ":" + env.get("PYTHONPATH", "")
    
    if args.command == "setup":
        return run_module("scripts.setup")
    
    elif args.command == "tokenizer":
        return run_module("src.tokenizer.tokenizer")
    
    elif args.command == "generate-data":
        cmd_args = []
        if args.input:
            cmd_args.extend(["--input", args.input])
        cmd_args.extend(["--output", args.output])
        cmd_args.extend(["--num-facts", str(args.num_facts)])
        cmd_args.extend(["--augmentations", str(args.augmentations)])
        cmd_args.extend(["--train-ratio", str(args.train_ratio)])
        return run_module("src.generation.generate_data", cmd_args)
    
    elif args.command == "train":
        cmd_args = ["--config", args.config]
        if args.resume:
            cmd_args.extend(["--resume", args.resume])
        if args.output_dir:
            cmd_args.extend(["--output-dir", args.output_dir])
        if args.max_steps:
            cmd_args.extend(["--max-steps", str(args.max_steps)])
        if args.batch_size:
            cmd_args.extend(["--batch-size", str(args.batch_size)])
        return run_module("src.training.train", cmd_args)
    
    elif args.command == "finetune":
        cmd_args = ["--config", args.config]
        if args.pretrained:
            cmd_args.extend(["--pretrained", args.pretrained])
        if args.train_data:
            cmd_args.extend(["--train-data", args.train_data])
        if args.val_data:
            cmd_args.extend(["--val-data", args.val_data])
        if args.output_dir:
            cmd_args.extend(["--output-dir", args.output_dir])
        if args.max_steps:
            cmd_args.extend(["--max-steps", str(args.max_steps)])
        if args.lr:
            cmd_args.extend(["--lr", str(args.lr)])
        return run_module("src.training.finetune", cmd_args)
    
    elif args.command == "finetune-pretrained":
        cmd_args = ["--config", args.config]
        if args.model:
            cmd_args.extend(["--model", args.model])
        if args.train_data:
            cmd_args.extend(["--train-data", args.train_data])
        if args.val_data:
            cmd_args.extend(["--val-data", args.val_data])
        if args.output_dir:
            cmd_args.extend(["--output-dir", args.output_dir])
        if args.max_steps:
            cmd_args.extend(["--max-steps", str(args.max_steps)])
        if args.lr:
            cmd_args.extend(["--lr", str(args.lr)])
        return run_module("src.finetune_pretrained.finetune_pretrained", cmd_args)
    
    elif args.command == "chat":
        cmd_args = ["--checkpoint", args.checkpoint, "--config", args.config, "--device", args.device]
        cmd_args.extend(["--temperature", str(args.temperature)])
        cmd_args.extend(["--top-k", str(args.top_k)])
        cmd_args.extend(["--top-p", str(args.top_p)])
        cmd_args.extend(["--max-tokens", str(args.max_tokens)])
        if args.strict:
            cmd_args.append("--strict")
        if args.prompt:
            cmd_args.extend(["--prompt", args.prompt])
        return run_module("src.inference.chat", cmd_args)
    
    elif args.command == "pipeline":
        print("=" * 60)
        print("Running full small-lm pipeline")
        print("=" * 60)
        
        # 1. Setup
        print("\n[1/6] Installing dependencies...")
        if run_module("scripts.setup") != 0:
            return 1
        
        # 2. Generate data
        print("\n[2/6] Generating synthetic data...")
        if run_module("src.generation.generate_data", ["--num-facts", "2000", "--augmentations", "3"]) != 0:
            return 1
        
        # 3. Train tokenizer
        print("\n[3/6] Training tokenizer...")
        if run_module("src.tokenizer.tokenizer") != 0:
            return 1
        
        # 4. Train from scratch (optional)
        if not args.skip_train:
            print("\n[4/6] Training from scratch...")
            if run_module("src.training.train", ["--max-steps", "5000"]) != 0:
                return 1
            pretrained_path = "checkpoints/pretrain"
        else:
            print("\n[4/6] Skipping from-scratch training")
            pretrained_path = None
        
        # 5. Fine-tune (from scratch or pretrained HF)
        print("\n[5/6] Fine-tuning...")
        if pretrained_path:
            if run_module("src.training.finetune", ["--pretrained", pretrained_path, "--max-steps", "1000"]) != 0:
                return 1
        else:
            if run_module("src.finetune_pretrained.finetune_pretrained", 
                         ["--model", args.pretrained_model, "--max-steps", "1000"]) != 0:
                return 1
        
        # 6. Test chat
        print("\n[6/6] Testing chat...")
        test_prompt = "what is the capital of france?"
        if run_module("src.inference.chat", ["--prompt", test_prompt, "--strict"]) != 0:
            return 1
        
        print("\n" + "=" * 60)
        print("Pipeline complete!")
        print("=" * 60)
        return 0
    
    return 0


if __name__ == "__main__":
    sys.exit(main())