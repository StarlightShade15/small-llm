#!/usr/bin/env python3
"""
Inference CLI for small-lm.
Interactive REPL with temperature, top-k, top-p, and strict style checking.
"""
import sys
import os
import yaml
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model.model import SmallLM, ModelConfig
from src.tokenizer.tokenizer import ChatTokenizer, SPECIAL_TOKEN_MAP


class StyleValidator:
    """Validates model outputs against style rules."""
    
    def __init__(self, config: dict):
        style_cfg = config.get("style", {})
        self.template = style_cfg.get("template", "The {property} of {subject} is {value}.")
        self.value_casing = style_cfg.get("value_casing", "lower")
        self.require_period = style_cfg.get("require_period", True)
        self.forbid_question_marks = style_cfg.get("forbid_question_marks", True)
        self.forbid_exclamation_marks = style_cfg.get("forbid_exclamation_marks", False)
        self.max_response_length = style_cfg.get("max_response_length", 64)
    
    def validate(self, response: str) -> Tuple[bool, List[str]]:
        """Validate a response. Returns (is_valid, list_of_violations)."""
        violations = []
        
        # Check ends with period
        if self.require_period and not response.strip().endswith("."):
            violations.append("Missing ending period")
        
        # Check no question marks
        if self.forbid_question_marks and "?" in response:
            violations.append("Contains question mark")
        
        # Check no exclamation marks
        if self.forbid_exclamation_marks and "!" in response:
            violations.append("Contains exclamation mark")
        
        # Check length
        if len(response) > self.max_response_length:
            violations.append(f"Exceeds max length ({self.max_response_length})")
        
        # Check template pattern: "The <property> of <subject> is <value>."
        # We can't fully validate without knowing the query, but check basic structure
        if not response.lower().startswith("the "):
            violations.append("Does not start with 'The'")
        
        if " is " not in response:
            violations.append("Missing 'is' pattern")
        
        # Check value casing (last word before period)
        if self.value_casing == "lower":
            words = response.strip().rstrip(".").split()
            if words:
                last_word = words[-1]
                if last_word[0].isupper() and last_word.isalpha():
                    violations.append(f"Value should be lowercase: '{last_word}'")
        
        return len(violations) == 0, violations


def load_model_and_tokenizer(
    checkpoint_path: str,
    device: torch.device,
    config_path: str = "configs/model_config.yaml",
) -> Tuple[SmallLM, ChatTokenizer, dict]:
    """Load model, tokenizer, and config."""
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Load tokenizer
    tokenizer = ChatTokenizer()
    
    # Load model
    model = SmallLM.from_pretrained(checkpoint_path, device=str(device))
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Device: {device}")
    
    return model, tokenizer, config


@torch.no_grad()
def generate_response(
    model: SmallLM,
    tokenizer: ChatTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> str:
    """Generate a single response."""
    # Encode prompt: <|user|>\n{prompt}\n<|agent|>
    input_ids = tokenizer.encode_user(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    # Generate
    generated = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.end_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # Extract only the new tokens (agent response)
    new_tokens = generated[0, len(input_ids[0]):].tolist()
    
    # Decode and clean up
    response = tokenizer.decode(new_tokens, skip_special=True)
    
    # Stop at <|end|> if present
    end_idx = response.find("<|end|>")
    if end_idx != -1:
        response = response[:end_idx]
    
    return response.strip()


def interactive_chat(
    model: SmallLM,
    tokenizer: ChatTokenizer,
    config: dict,
    device: torch.device,
    gen_config: dict,
    strict: bool = False,
):
    """Run interactive chat REPL."""
    validator = StyleValidator(config) if strict else None
    
    print("\n" + "=" * 60)
    print("small-lm Interactive Chat")
    print("Type 'quit', 'exit', or Ctrl+C to exit")
    print("Commands: /help, /config, /strict on|off")
    if strict:
        print("STRICT MODE: ON - style violations will be flagged")
    print("=" * 60 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.lower() in ("quit", "exit", "/quit", "/exit"):
            print("Goodbye!")
            break
        
        if user_input == "/help":
            print("\nCommands:")
            print("  /help           - Show this help")
            print("  /config         - Show generation config")
            print("  /strict on|off  - Toggle strict mode")
            print("  /temp <value>   - Set temperature")
            print("  /topk <value>   - Set top-k")
            print("  /topp <value>   - Set top-p")
            print("  quit/exit       - Exit chat")
            print()
            continue
        
        if user_input == "/config":
            print(f"\nGeneration Config:")
            for k, v in gen_config.items():
                print(f"  {k}: {v}")
            print(f"  strict: {strict}")
            print()
            continue
        
        if user_input.startswith("/strict"):
            parts = user_input.split()
            if len(parts) == 2:
                strict = parts[1].lower() in ("on", "true", "1", "yes")
                validator = StyleValidator(config) if strict else None
                print(f"Strict mode: {'ON' if strict else 'OFF'}")
            else:
                print(f"Strict mode: {'ON' if strict else 'OFF'}")
            continue
        
        if user_input.startswith("/temp"):
            parts = user_input.split()
            if len(parts) == 2:
                try:
                    gen_config["temperature"] = float(parts[1])
                    print(f"Temperature set to {gen_config['temperature']}")
                except ValueError:
                    print("Invalid temperature value")
            continue
        
        if user_input.startswith("/topk"):
            parts = user_input.split()
            if len(parts) == 2:
                try:
                    gen_config["top_k"] = int(parts[1])
                    print(f"Top-k set to {gen_config['top_k']}")
                except ValueError:
                    print("Invalid top-k value")
            continue
        
        if user_input.startswith("/topp"):
            parts = user_input.split()
            if len(parts) == 2:
                try:
                    gen_config["top_p"] = float(parts[1])
                    print(f"Top-p set to {gen_config['top_p']}")
                except ValueError:
                    print("Invalid top-p value")
            continue
        
        # Generate response
        print("Agent: ", end="", flush=True)
        response = generate_response(
            model, tokenizer, user_input, device,
            max_new_tokens=gen_config["max_new_tokens"],
            temperature=gen_config["temperature"],
            top_k=gen_config["top_k"],
            top_p=gen_config["top_p"],
            repetition_penalty=gen_config["repetition_penalty"],
        )
        print(response)
        
        # Strict validation
        if strict and validator:
            is_valid, violations = validator.validate(response)
            if not is_valid:
                print("  ⚠ Style violations:")
                for v in violations:
                    print(f"    - {v}")
        
        print()


def main():
    parser = argparse.ArgumentParser(description="small-lm chat CLI")
    parser.add_argument("--checkpoint", "-c", default="checkpoints/finetune/best_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--config", default="configs/model_config.yaml",
                        help="Path to config file")
    parser.add_argument("--device", choices=["cuda", "cpu", "mps", "auto"], default="auto",
                        help="Device to use")
    parser.add_argument("--temperature", "-t", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top-k", "-k", type=int, default=50,
                        help="Top-k sampling")
    parser.add_argument("--top-p", "-p", type=float, default=0.9,
                        help="Top-p (nucleus) sampling")
    parser.add_argument("--max-tokens", "-m", type=int, default=64,
                        help="Max new tokens to generate")
    parser.add_argument("--repetition-penalty", "-r", type=float, default=1.1,
                        help="Repetition penalty")
    parser.add_argument("--strict", "-s", action="store_true",
                        help="Enable strict style checking")
    parser.add_argument("--prompt", help="Single prompt (non-interactive)")
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    # Load model and tokenizer
    try:
        model, tokenizer, config = load_model_and_tokenizer(
            args.checkpoint, device, args.config
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Generation config
    gen_config = {
        "max_new_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
    }
    
    # Single prompt or interactive
    if args.prompt:
        response = generate_response(model, tokenizer, args.prompt, device, **gen_config)
        print(response)
        
        if args.strict:
            validator = StyleValidator(config)
            is_valid, violations = validator.validate(response)
            if not is_valid:
                print("\nStyle violations:", file=sys.stderr)
                for v in violations:
                    print(f"  - {v}", file=sys.stderr)
                sys.exit(1)
    else:
        interactive_chat(model, tokenizer, config, device, gen_config, args.strict)


if __name__ == "__main__":
    main()