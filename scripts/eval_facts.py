#!/usr/bin/env python3
"""Score a checkpoint against a facts CSV.

Answers every generated question for every fact with greedy decoding and
reports hard numbers: exact-answer rate, value-accuracy, template conformance
and style violations. This is the measurement to watch while fine-tuning -- the
sample block in the training log shows six prompts and no score.

Examples:
    python scripts/eval_facts.py \
        --checkpoint checkpoints/finetune_real/best_model.pt \
        --facts data/raw/sample_facts.csv

    python scripts/eval_facts.py \
        --checkpoint checkpoints/finetune_real/best_model.pt \
        --facts data/raw/sample_facts.csv --questions first --verbose
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.generation.generate_data import Fact, load_facts_from_csv, load_config
from src.model.model import SmallLM
from src.tokenizer.tokenizer import ChatTokenizer


def expected_answer(style_config: dict, fact: Fact) -> str:
    """Render the reference answer for a fact using the project's style rules."""
    template = style_config.get("template", "The {property} of {subject} is {value}.")
    value = fact.value
    casing = style_config.get("value_casing", "lower")
    if casing == "lower":
        value = value.lower()
    elif casing == "upper":
        value = value.upper()
    elif casing == "title":
        value = value.title()
    return template.format(subject=fact.subject, property=fact.property, value=value)


def template_pattern(style_config: dict) -> re.Pattern:
    """Regex that accepts any slot filling of the configured template."""
    template = style_config.get("template", "The {property} of {subject} is {value}.")
    slots = {
        "property": r"[A-Za-z0-9' ]+",
        "subject": r"[A-Za-z0-9' ]+",
        "value": r"[A-Za-z0-9'., ]+",
    }
    pattern = re.escape(template)
    for name, slot in slots.items():
        pattern = pattern.replace(re.escape("{" + name + "}"), slot)
    return re.compile(pattern + r"\s*$")


def style_violations(style_config: dict, answer: str) -> list:
    """Style rules from configs/model_config.yaml, applied to one answer."""
    problems = []
    if style_config.get("require_period", True) and not answer.strip().endswith("."):
        problems.append("missing period")
    if style_config.get("forbid_question_marks", True) and "?" in answer:
        problems.append("question mark")
    if style_config.get("forbid_exclamation_marks", False) and "!" in answer:
        problems.append("exclamation mark")
    max_length = style_config.get("max_response_length")
    if max_length and len(answer.split()) > int(max_length):
        problems.append(f"too long ({len(answer.split())} words)")
    return problems


def generate_answer(model, tokenizer, question, device, max_new_tokens):
    input_ids = tokenizer.encode_user(question)
    tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        generated = model.generate(
            tensor,
            max_new_tokens=max_new_tokens,
            temperature=0.0,  # greedy: the eval must be deterministic
            top_k=0,
            top_p=1.0,
            repetition_penalty=1.0,
            eos_token_id=tokenizer.end_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = generated[0, len(input_ids):].tolist()
    text = tokenizer.decode(new_tokens, skip_special=True)
    return text.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Score a checkpoint against a facts CSV")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory or file")
    parser.add_argument("--facts", default="data/raw/sample_facts.csv", help="Facts CSV")
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--questions", choices=["all", "first"], default="all",
                        help="Ask every paraphrase per fact, or only the first")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--max-facts", type=int, default=0, help="0 means every fact")
    parser.add_argument("--verbose", action="store_true", help="Print every answer")
    parser.add_argument("--show-failures", type=int, default=10)
    args = parser.parse_args()

    config = load_config(args.config)
    style_config = config.get("style", {})
    pattern = template_pattern(style_config)

    facts = load_facts_from_csv(args.facts)
    if args.max_facts:
        facts = facts[: args.max_facts]
    if not facts:
        print(f"No facts loaded from {args.facts}")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = ChatTokenizer()
    model = SmallLM.from_pretrained(args.checkpoint, device=str(device))
    model.eval()

    total = 0
    exact = 0
    value_hits = 0
    template_hits = 0
    style_clean = 0
    failures = []

    for fact in facts:
        expected = expected_answer(style_config, fact)
        expected_value = fact.value.lower()
        questions = fact._generate_questions()
        if args.questions == "first":
            questions = questions[:1]

        for question in questions:
            answer = generate_answer(
                model, tokenizer, question, device, args.max_new_tokens
            )
            total += 1
            exact += 1 if answer == expected else 0
            value_hits += 1 if expected_value in answer.lower() else 0
            template_hits += 1 if pattern.search(answer) else 0
            problems = style_violations(style_config, answer)
            style_clean += 1 if not problems else 0

            if args.verbose:
                print(f"Q: {question}")
                print(f"  expected: {expected}")
                print(f"  got:      {answer}")
                if problems:
                    print(f"  style:    {', '.join(problems)}")
            if answer != expected and len(failures) < args.show_failures:
                failures.append((question, expected, answer, problems))

    def pct(count):
        return 100.0 * count / max(1, total)

    print(f"\nFacts: {len(facts)} | questions scored: {total}")
    print(f"  exact answer     : {exact}/{total} ({pct(exact):.1f}%)")
    print(f"  value correct    : {value_hits}/{total} ({pct(value_hits):.1f}%)")
    print(f"  template match   : {template_hits}/{total} ({pct(template_hits):.1f}%)")
    print(f"  style clean      : {style_clean}/{total} ({pct(style_clean):.1f}%)")

    if failures:
        print(f"\nFirst {len(failures)} misses:")
        for question, expected, answer, problems in failures:
            print(f"  Q: {question}")
            print(f"     expected: {expected}")
            print(f"     got:      {answer}")
            if problems:
                print(f"     style:    {', '.join(problems)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
