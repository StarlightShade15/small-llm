"""
Data loading and processing for chat-style Q&A.
Handles JSONL format, chat templating, loss masking, train/val split.
"""
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from src.tokenizer.tokenizer import ChatTokenizer, SPECIAL_TOKEN_MAP


@dataclass
class DataConfig:
    train_split: float = 0.95
    max_seq_len: int = 512
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    seed: int = 42


class ChatDataset(Dataset):
    """Dataset for chat-style Q&A with loss masking on agent span only."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: ChatTokenizer,
        max_seq_len: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples = self._load_data(data_path)
        print(f"Loaded {len(self.examples)} examples from {data_path}")
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """Load JSONL data: {"user": "...", "agent": "..."}"""
        examples = []
        with open(data_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                    if "user" in ex and "agent" in ex:
                        examples.append(ex)
                    else:
                        print(f"Warning: Line {line_num} missing 'user' or 'agent' field")
                except json.JSONDecodeError as e:
                    print(f"Warning: Line {line_num} JSON decode error: {e}")
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        encoded = self.tokenizer.encode_full(ex["user"], ex["agent"])
        
        # Truncate if too long
        input_ids = encoded["input_ids"][:self.max_seq_len]
        labels = encoded["labels"][:self.max_seq_len]
        attention_mask = encoded["attention_mask"][:self.max_seq_len]
        
        # Pad if too short
        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            pad_id = SPECIAL_TOKEN_MAP["<|pad|>"]
            input_ids = input_ids + [pad_id] * pad_len
            labels = labels + [-100] * pad_len
            attention_mask = attention_mask + [0] * pad_len
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def create_dataloaders(
    data_path: str,
    tokenizer: ChatTokenizer,
    config: DataConfig,
    batch_size: int,
    val_data_path: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    
    if val_data_path and os.path.exists(val_data_path):
        # Use explicit validation file
        train_dataset = ChatDataset(data_path, tokenizer, config.max_seq_len)
        val_dataset = ChatDataset(val_data_path, tokenizer, config.max_seq_len)
    else:
        # Split from single file
        full_dataset = ChatDataset(data_path, tokenizer, config.max_seq_len)
        train_size = int(len(full_dataset) * config.train_split)
        val_size = len(full_dataset) - train_size
        
        # Use fixed seed for reproducibility
        generator = torch.Generator().manual_seed(config.seed)
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size], generator=generator
        )
        print(f"Split: {train_size} train, {val_size} val")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        pin_memory=config.pin_memory,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        pin_memory=config.pin_memory,
        drop_last=False,
    )
    
    return train_loader, val_loader


def create_data_splits(
    input_path: str,
    output_dir: str,
    train_split: float = 0.95,
    seed: int = 42,
) -> Tuple[str, str]:
    """Split a JSONL file into train/val and save."""
    os.makedirs(output_dir, exist_ok=True)
    
    examples = []
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    
    random.seed(seed)
    random.shuffle(examples)
    
    split_idx = int(len(examples) * train_split)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]
    
    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")
    
    with open(train_path, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")
    
    with open(val_path, "w") as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + "\n")
    
    print(f"Created splits: {len(train_examples)} train, {len(val_examples)} val")
    return train_path, val_path


class InfiniteDataLoader:
    """Wrapper to iterate indefinitely over a DataLoader."""
    
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.iterator = iter(loader)
    
    def __iter__(self) -> Iterator[Dict]:
        return self
    
    def __next__(self) -> Dict:
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function (default works but this is explicit)."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
    }


if __name__ == "__main__":
    # Test data loading
    from src.tokenizer.tokenizer import ChatTokenizer
    
    # Create sample data
    os.makedirs("data/raw", exist_ok=True)
    sample_data = "data/raw/sample_chat.jsonl"
    if not os.path.exists(sample_data):
        with open(sample_data, "w") as f:
            for i in range(100):
                ex = {
                    "user": f"what is the capital of france?",
                    "agent": "The capital of France is paris."
                }
                f.write(json.dumps(ex) + "\n")
    
    # Test
    tokenizer = ChatTokenizer()
    train_loader, val_loader = create_dataloaders(
        sample_data,
        tokenizer,
        DataConfig(max_seq_len=128),
        batch_size=4,
    )
    
    batch = next(iter(train_loader))
    print(f"Batch shapes:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  labels: {batch['labels'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    print(f"  Non-masked labels per sample: {(batch['labels'] != -100).sum(dim=1)}")