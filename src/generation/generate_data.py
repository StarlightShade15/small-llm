#!/usr/bin/env python3
"""
Synthetic data generator for small-lm.
Given a CSV/JSON of (subject, property, value) facts, produces Q&A pairs
formatted exactly in the target style.
"""
import sys
import os
import json
import csv
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.tokenizer.tokenizer import load_config


@dataclass
class Fact:
    subject: str
    property: str
    value: str
    
    def to_qa(self, style_config: dict) -> Dict[str, str]:
        """Convert fact to Q&A pair using style template."""
        template = style_config.get("template", "The {property} of {subject} is {value}.")
        value_casing = style_config.get("value_casing", "lower")
        
        value = self.value
        if value_casing == "lower":
            value = value.lower()
        elif value_casing == "upper":
            value = value.upper()
        elif value_casing == "title":
            value = value.title()
        
        # Generate question variations
        questions = self._generate_questions()
        question = random.choice(questions)
        
        # Generate answer using template
        answer = template.format(
            subject=self.subject,
            property=self.property,
            value=value
        )
        
        return {"user": question, "agent": answer}
    
    def _generate_questions(self) -> List[str]:
        """Generate natural question variations for this fact."""
        subj = self.subject
        prop = self.property
        
        templates = [
            f"what is the {prop} of {subj}?",
            f"what's the {prop} of {subj}?",
            f"tell me the {prop} of {subj}",
            f"{subj} {prop}?",
            f"what is {subj}'s {prop}?",
            f"what's {subj}'s {prop}?",
        ]
        
        # Property-specific templates
        if prop == "capital":
            templates.extend([
                f"what is the capital of {subj}?",
                f"capital of {subj}?",
            ])
        elif prop in ("population", "area", "size"):
            templates.extend([
                f"how many people live in {subj}?",
                f"what is the population of {subj}?",
            ])
        elif prop in ("height", "elevation"):
            templates.extend([
                f"how tall is {subj}?",
                f"what is the height of {subj}?",
            ])
        elif prop in ("author", "creator", "inventor"):
            templates.extend([
                f"who wrote {subj}?",
                f"who created {subj}?",
                f"who invented {subj}?",
            ])
        elif prop in ("year", "date", "released"):
            templates.extend([
                f"when was {subj} released?",
                f"what year was {subj} made?",
            ])
        
        return templates


def load_facts_from_csv(path: str) -> List[Fact]:
    """Load facts from CSV with columns: subject, property, value."""
    facts = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle different column names
            subject = row.get("subject") or row.get("entity") or row.get("name")
            prop = row.get("property") or row.get("attribute") or row.get("key")
            value = row.get("value") or row.get("answer") or row.get("val")
            
            if subject and prop and value:
                facts.append(Fact(subject.strip(), prop.strip(), value.strip()))
    return facts


def load_facts_from_json(path: str) -> List[Fact]:
    """Load facts from JSON array of objects."""
    facts = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for item in data:
        subject = item.get("subject") or item.get("entity") or item.get("name")
        prop = item.get("property") or item.get("attribute") or item.get("key")
        value = item.get("value") or item.get("answer") or item.get("val")
        
        if subject and prop and value:
            facts.append(Fact(subject.strip(), prop.strip(), value.strip()))
    return facts


def load_facts_from_jsonl(path: str) -> List[Fact]:
    """Load facts from JSONL (one fact per line)."""
    facts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            subject = item.get("subject") or item.get("entity") or item.get("name")
            prop = item.get("property") or item.get("attribute") or item.get("key")
            value = item.get("value") or item.get("answer") or item.get("val")
            
            if subject and prop and value:
                facts.append(Fact(subject.strip(), prop.strip(), value.strip()))
    return facts


def generate_synthetic_facts(num_facts: int = 1000) -> List[Fact]:
    """Generate synthetic facts for testing."""
    subjects = [
        "france", "germany", "italy", "spain", "japan", "china", "india", "brazil",
        "canada", "australia", "russia", "mexico", "korea", "turkey", "indonesia",
        "paris", "london", "tokyo", "new york", "beijing", "moscow", "sydney",
        "earth", "mars", "jupiter", "saturn", "venus", "mercury", "neptune", "uranus",
        "mount everest", "k2", "kilimanjaro", "denali", "elbrus", "aconcagua",
        "pacific ocean", "atlantic ocean", "indian ocean", "arctic ocean",
        "amazon river", "nile river", "yangtze river", "mississippi river",
        "python", "javascript", "rust", "go", "java", "c++", "typescript",
        "guido van rossum", "brendan eich", "graydon hoare", "ken thompson",
        "albert einstein", "marie curie", "isaac newton", "charles darwin",
        "shakespeare", "tolstoy", "hemingway", "austen", "dickens",
    ]
    
    properties = [
        "capital", "population", "area", "language", "currency",
        "height", "elevation", "depth", "length", "width",
        "author", "creator", "inventor", "founder", "designer",
        "year", "date", "released", "published", "discovered",
        "speed", "temperature", "mass", "volume", "density",
    ]
    
    values = [
        "paris", "berlin", "rome", "madrid", "tokyo", "beijing", "new delhi", "brasilia",
        "ottawa", "canberra", "moscow", "mexico city", "seoul", "ankara", "jakarta",
        "8 billion", "1.4 billion", "1.3 billion", "67 million", "83 million", "60 million",
        "510 million sq km", "144 million sq km", "3.7 million sq km", "9.6 million sq km",
        "english", "spanish", "mandarin", "hindi", "french", "arabic", "bengali", "russian",
        "usd", "eur", "jpy", "cny", "inr", "brl", "cad", "aud",
        "8848 m", "8611 m", "5895 m", "6190 m", "5642 m", "6961 m",
        "165 million sq km", "85 million sq km", "70 million sq km", "14 million sq km",
        "6992 km", "6650 km", "6300 km", "3730 km",
        "guido van rossum", "brendan eich", "graydon hoare", "ken thompson",
        "1905", "1915", "1687", "1859",
        "william shakespeare", "leo tolstoy", "ernest hemingway", "jane austen", "charles dickens",
    ]
    
    combinations = [
        Fact(subject, prop, value)
        for subject in subjects
        for prop in properties
        for value in values
    ]
    if num_facts > len(combinations):
        print(f"Requested {num_facts} facts, but only {len(combinations)} unique synthetic combinations exist")
    return random.sample(combinations, min(num_facts, len(combinations)))


def generate_qa_dataset(
    facts: List[Fact],
    style_config: dict,
    num_augmentations: int = 1,
) -> List[Dict[str, str]]:
    """Generate Q&A dataset from facts with augmentations."""
    qa_pairs = []
    
    for fact in facts:
        for _ in range(num_augmentations):
            qa = fact.to_qa(style_config)
            qa_pairs.append(qa)
    
    # Repeated augmentation can choose the same wording more than once.
    unique_pairs = list({(qa["user"], qa["agent"]): qa for qa in qa_pairs}.values())
    random.shuffle(unique_pairs)
    return unique_pairs


def split_facts(
    facts: List[Fact],
    train_ratio: float = 0.95,
    seed: int = 42,
) -> Tuple[List[Fact], List[Fact]]:
    """Split underlying facts before question augmentation to prevent leakage."""
    unique = {}
    for fact in facts:
        key = (fact.subject.casefold(), fact.property.casefold(), fact.value.casefold())
        unique[key] = fact
    unique_facts = list(unique.values())
    random.Random(seed).shuffle(unique_facts)
    if len(unique_facts) < 2:
        return unique_facts, []
    split_idx = max(1, min(len(unique_facts) - 1, int(len(unique_facts) * train_ratio)))
    return unique_facts[:split_idx], unique_facts[split_idx:]


def save_qa_dataset(qa_pairs: List[Dict[str, str]], output_path: str):
    """Save Q&A pairs as JSONL."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")
    print(f"Saved {len(qa_pairs)} Q&A pairs to {output_path}")


def create_train_val_split(
    qa_pairs: List[Dict[str, str]],
    train_ratio: float = 0.95,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Split QA examples by answer so paraphrases of a fact stay together."""
    groups = {}
    for index, pair in enumerate(qa_pairs):
        # Generated variations for the same fact share a templated answer.
        key = str(pair.get("agent", "")).casefold() or f"__row_{index}"
        groups.setdefault(key, []).append(pair)
    group_values = list(groups.values())
    random.Random(seed).shuffle(group_values)
    if len(group_values) < 2:
        return qa_pairs, []
    split_idx = max(1, min(len(group_values) - 1, int(len(group_values) * train_ratio)))
    train_pairs = [pair for group in group_values[:split_idx] for pair in group]
    val_pairs = [pair for group in group_values[split_idx:] for pair in group]
    return train_pairs, val_pairs


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Q&A data")
    parser.add_argument("--input", "-i", help="Input facts file (CSV, JSON, or JSONL)")
    parser.add_argument("--output", "-o", default="data/raw/synthetic_chat.jsonl",
                        help="Output Q&A dataset path")
    parser.add_argument("--config", "-c", default="configs/model_config.yaml",
                        help="Config file for style")
    parser.add_argument("--num-facts", "-n", type=int, default=1000,
                        help="Number of synthetic facts to generate (if no input)")
    parser.add_argument("--augmentations", "-a", type=int, default=3,
                        help="Number of question variations per fact")
    parser.add_argument("--train-ratio", type=float, default=0.95,
                        help="Train split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()

    random.seed(args.seed)
    
    # Load style config
    config = load_config(args.config)
    style_config = config.get("style", {})
    
    # Load or generate facts
    if args.input:
        ext = Path(args.input).suffix.lower()
        if ext == ".csv":
            facts = load_facts_from_csv(args.input)
        elif ext == ".json":
            facts = load_facts_from_json(args.input)
        elif ext == ".jsonl":
            facts = load_facts_from_jsonl(args.input)
        else:
            print(f"Unsupported input format: {ext}")
            sys.exit(1)
        print(f"Loaded {len(facts)} facts from {args.input}")
    else:
        facts = generate_synthetic_facts(args.num_facts)
        print(f"Generated {len(facts)} synthetic facts")
    
    # Split facts first: question variations from one fact must not appear in
    # both train and validation, or the validation loss gives a false signal.
    train_facts, val_facts = split_facts(facts, args.train_ratio, args.seed)
    train_pairs = generate_qa_dataset(train_facts, style_config, args.augmentations)
    val_pairs = generate_qa_dataset(val_facts, style_config, args.augmentations)
    qa_pairs = train_pairs + val_pairs
    random.shuffle(qa_pairs)
    print(f"Generated {len(qa_pairs)} unique Q&A pairs from {len(facts)} unique facts")

    # Save combined data for inspection, plus fact-disjoint training splits.
    save_qa_dataset(qa_pairs, args.output)
    
    train_path = args.output.replace(".jsonl", "_train.jsonl")
    val_path = args.output.replace(".jsonl", "_val.jsonl")
    
    save_qa_dataset(train_pairs, train_path)
    save_qa_dataset(val_pairs, val_path)
    
    # Also save to processed dir for training scripts
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    save_qa_dataset(train_pairs, str(processed_dir / "train.jsonl"))
    save_qa_dataset(val_pairs, str(processed_dir / "val.jsonl"))
    
    print("\nDone! Files created:")
    print(f"  {args.output}")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  data/processed/train.jsonl")
    print(f"  data/processed/val.jsonl")


if __name__ == "__main__":
    main()
