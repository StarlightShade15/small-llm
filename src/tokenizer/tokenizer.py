"""
Byte-level BPE tokenizer training and utilities.
Trains on provided corpus, saves vocab, and provides encoding/decoding.
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Union
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers.normalizers import NFKC, Sequence
import yaml


SPECIAL_TOKENS = ["<|pad|>", "<|user|>", "<|agent|>", "<|end|>"]
SPECIAL_TOKEN_MAP = {
    "<|pad|>": 0,
    "<|user|>": 1,
    "<|agent|>": 2,
    "<|end|>": 3,
}


def load_config(config_path: str = "configs/model_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_tokenizer(vocab_size: int = 16384, min_frequency: int = 2) -> Tokenizer:
    """Build a byte-level BPE tokenizer with special tokens."""
    tokenizer = Tokenizer(models.BPE(unk_token="<|pad|>"))
    
    # Normalization: NFKC for consistency
    tokenizer.normalizer = Sequence([NFKC()])
    
    # Pre-tokenization: byte-level (no pre-tokenization, works on bytes)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # Decoder: byte-level
    tokenizer.decoder = decoders.ByteLevel()
    
    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        # BPE merges every adjacent pair whose corpus count reaches this value.
        min_frequency=min_frequency,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    
    return tokenizer, trainer


def train_tokenizer(
    corpus_paths: List[str],
    vocab_size: int = 16384,
    output_dir: str = "data/processed/tokenizer",
    min_frequency: int = 2,
) -> Tokenizer:
    """Train tokenizer on corpus files."""
    os.makedirs(output_dir, exist_ok=True)
    
    tokenizer, trainer = build_tokenizer(vocab_size, min_frequency)
    
    print(f"Training tokenizer on {len(corpus_paths)} files...")
    tokenizer.train(corpus_paths, trainer)
    
    # Verify special tokens have correct IDs
    for tok, expected_id in SPECIAL_TOKEN_MAP.items():
        actual_id = tokenizer.token_to_id(tok)
        if actual_id != expected_id:
            raise ValueError(f"Special token {tok} has ID {actual_id}; expected {expected_id}")
    
    # Save
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path, pretty=True)
    print(f"Tokenizer saved to {tokenizer_path}")
    
    # Also save vocab separately for inspection
    vocab_path = os.path.join(output_dir, "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(tokenizer.get_vocab(), f, ensure_ascii=False, indent=2)
    print(f"Vocab saved to {vocab_path}")
    
    return tokenizer


def load_tokenizer(tokenizer_path: str = "data/processed/tokenizer/tokenizer.json") -> Tokenizer:
    """Load trained tokenizer."""
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}. Run train_tokenizer first.")
    return Tokenizer.from_file(tokenizer_path)


def encode_chat(tokenizer: Tokenizer, user_text: str, agent_text: str) -> Dict:
    """Encode a single chat turn: <|user|>\\n{user}\\n<|agent|>\\n{agent}<|end|>"""
    # Build the full sequence
    full_text = f"<|user|>\n{user_text}\n<|agent|>\n{agent_text}<|end|>"
    encoding = tokenizer.encode(full_text)
    
    # Find agent span for loss masking
    agent_token_id = SPECIAL_TOKEN_MAP["<|agent|>"]
    end_token_id = SPECIAL_TOKEN_MAP["<|end|>"]
    
    agent_start = None
    agent_end = None
    for i, token_id in enumerate(encoding.ids):
        if token_id == agent_token_id and agent_start is None:
            agent_start = i + 1  # Start after <|agent|>
        elif token_id == end_token_id and agent_start is not None:
            agent_end = i  # End before <|end|>
            break
    
    # Create labels: -100 for masked positions (user tokens, padding, special)
    labels = [-100] * len(encoding.ids)
    if agent_start is not None and agent_end is not None:
        for i in range(agent_start, agent_end):
            labels[i] = encoding.ids[i]
        # Teach the model to stop at the end-of-turn token as well.
        labels[agent_end] = encoding.ids[agent_end]
    
    return {
        "input_ids": encoding.ids,
        "attention_mask": encoding.attention_mask,
        "labels": labels,
    }


def decode_chat(tokenizer: Tokenizer, token_ids: List[int]) -> str:
    """Decode token IDs to text."""
    return tokenizer.decode(token_ids, skip_special_tokens=False)


class ChatTokenizer:
    """Wrapper for chat-specific tokenization."""
    
    def __init__(self, tokenizer_path: str = "data/processed/tokenizer/tokenizer.json"):
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.pad_token_id = SPECIAL_TOKEN_MAP["<|pad|>"]
        self.user_token_id = SPECIAL_TOKEN_MAP["<|user|>"]
        self.agent_token_id = SPECIAL_TOKEN_MAP["<|agent|>"]
        self.end_token_id = SPECIAL_TOKEN_MAP["<|end|>"]
        self.vocab_size = self.tokenizer.get_vocab_size()
    
    def encode_user(self, text: str) -> List[int]:
        """Encode just the user part: <|user|>\\n{text}\\n<|agent|>"""
        full = f"<|user|>\n{text}\n<|agent|>"
        return self.tokenizer.encode(full).ids
    
    def encode_full(self, user_text: str, agent_text: str) -> Dict:
        """Encode full chat turn with labels for training."""
        return encode_chat(self.tokenizer, user_text, agent_text)
    
    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """Decode tokens to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special)
    
    def get_agent_response(self, full_text: str) -> str:
        """Extract agent response from full decoded text."""
        # Find <|agent|> and <|end|>
        start_marker = "<|agent|>\n"
        end_marker = "<|end|>"
        start_idx = full_text.find(start_marker)
        if start_idx == -1:
            return ""
        start_idx += len(start_marker)
        end_idx = full_text.find(end_marker, start_idx)
        if end_idx == -1:
            return full_text[start_idx:]
        return full_text[start_idx:end_idx]


class HuggingFaceChatTokenizer:
    """Small adapter so the HF fine-tuning path uses its own tokenizer IDs."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.convert_tokens_to_ids("<|pad|>")
        self.user_token_id = tokenizer.convert_tokens_to_ids("<|user|>")
        self.agent_token_id = tokenizer.convert_tokens_to_ids("<|agent|>")
        self.end_token_id = tokenizer.convert_tokens_to_ids("<|end|>")
        self.vocab_size = len(tokenizer)

    def encode_user(self, text: str) -> List[int]:
        formatted = f"<|user|>\n{text}\n<|agent|>\n"
        return self.tokenizer.encode(formatted, add_special_tokens=False)

    def encode_full(self, user_text: str, agent_text: str) -> Dict:
        prefix = self.encode_user(user_text)
        full = f"<|user|>\n{user_text}\n<|agent|>\n{agent_text}<|end|>"
        input_ids = self.tokenizer.encode(full, add_special_tokens=False)
        # The added <|agent|> special token makes the prefix stable at this boundary.
        labels = [-100] * len(input_ids)
        labels[len(prefix):] = input_ids[len(prefix):]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special)


if __name__ == "__main__":
    import sys
    
    config = load_config()
    vocab_size = config["model"]["vocab_size"]
    min_frequency = int(config.get("tokenizer", {}).get("min_frequency", 2))
    
    # Find corpus files
    corpus_dir = Path("data/raw")
    corpus_files = list(corpus_dir.glob("*.txt")) + list(corpus_dir.glob("*.jsonl"))
    
    if not corpus_files:
        print("No corpus files found in data/raw/. Creating sample corpus...")
        corpus_dir.mkdir(parents=True, exist_ok=True)
        sample_corpus = corpus_dir / "sample_corpus.txt"
        with open(sample_corpus, "w") as f:
            f.write("""The capital of France is paris.
The capital of Germany is berlin.
The capital of Italy is rome.
The capital of Spain is madrid.
The capital of Japan is tokyo.
The largest planet is jupiter.
The smallest planet is mercury.
Water boils at 100 celsius.
The speed of light is 299792458 meters per second.
""")
        corpus_files = [sample_corpus]
    
    tokenizer = train_tokenizer(
        [str(f) for f in corpus_files],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
    )
    
    # Test
    chat_tok = ChatTokenizer()
    test = chat_tok.encode_full("what is the capital of France?", "The capital of France is paris.")
    print("\nTest encoding:")
    print(f"  Input IDs: {test['input_ids'][:30]}... (len={len(test['input_ids'])})")
    print(f"  Labels: {test['labels'][:30]}... (non-masked: {sum(1 for l in test['labels'] if l != -100)})")
    print(f"  Decoded: {chat_tok.decode(test['input_ids'])}")
