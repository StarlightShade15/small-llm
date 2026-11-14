# small-lm: From-Scratch Small Language Model (~116M params)

A complete, working implementation of a small decoder-only transformer for chat-style Q&A with a fixed response style.

## Target Behavior

```
user: what is the capital of France?
agent: The capital of France is paris.
```

**Style Rules (configurable in `configs/model_config.yaml`):**
- Declarative full sentence, one line, ends with period
- No questions back, no lists, no hedging, no "Sure!"/"Great question"
- Proper nouns in answer position are lowercase (e.g., "paris", "london")
- Template: `"The {property} of {subject} is {value}."`

## Architecture (default config)

| Component | Specification |
|-----------|---------------|
| Type | Decoder-only Transformer |
| Layers | 12 |
| Hidden dim (d_model) | 768 |
| Attention heads | 12 |
| KV heads | 4 (GQA) |
| FFN hidden (SwiGLU) | 3072 |
| Max sequence length | 512 |
| Vocab size | 16,384 |
| Norm | RMSNorm (pre-norm) |
| Positional | RoPE (θ=500,000) |
| Weight tying | Yes (embeddings ↔ output) |
| Activation | SwiGLU |

**Parameter count:** about 116M (printed at startup)

## File Tree

```
small-lm/
├── configs/
│   └── model_config.yaml          # All configuration (model, training, style, data)
├── scripts/
│   ├── setup.py                   # Install dependencies (auto-detects CUDA)
│   └── run.py                     # Main entry point (pipeline orchestrator)
├── src/
│   ├── tokenizer/
│   │   └── tokenizer.py           # Byte-level BPE tokenizer + chat encoding
│   ├── model/
│   │   └── model.py               # SmallLM: decoder-only transformer
│   ├── data/
│   │   └── dataset.py             # JSONL chat dataset with loss masking
│   ├── training/
│   │   ├── train.py               # From-scratch training
│   │   ├── finetune.py            # Fine-tuning with early stopping
│   │   └── distributed.py         # torchrun/DDP helpers
│   ├── inference/
│   │   ├── chat.py                # Interactive CLI with strict mode
│   │   └── chat_pretrained.py     # Chat with a fine-tuned HF checkpoint
│   ├── generation/
│   │   └── generate_data.py       # Synthetic Q&A from facts
│   └── finetune_pretrained/
│       └── finetune_pretrained.py # Fine-tune HF model (e.g., SmolLM-135M)
├── data/
│   ├── raw/                       # Input corpus & facts
│   └── processed/                 # Tokenizer, train/val splits
├── checkpoints/
│   ├── pretrain/                  # From-scratch checkpoints
│   ├── finetune/                  # Fine-tuned checkpoints
│   └── finetune_pretrained/       # HF fine-tuned checkpoints
├── logs/
├── requirements.txt
└── README.md
```

## Quick Start: Full Pipeline

```bash
cd /home/starlight/projects/small-lm

# 1. Install dependencies (auto-detects CUDA/CPU)
python scripts/setup.py

# 2. Generate synthetic data (random combinations for pipeline smoke tests)
python -m src.generation.generate_data --num-facts 2000 --augmentations 3

# 3. Train tokenizer on generated data
python -m src.tokenizer.tokenizer

# 4. Train from scratch (step count means optimizer updates)
python -m src.training.train --max-steps 10000

# 5. Fine-tune on same data (lower LR, early stopping)
python -m src.training.finetune --pretrained checkpoints/pretrain/best_model.pt --max-steps 3000

# 6. Chat interactively
python -m src.inference.chat --checkpoint checkpoints/finetune/best_model.pt
```

## Alternative: Fine-tune Pretrained HF Model (SmolLM-135M)

```bash
# Skip from-scratch training, fine-tune SmolLM directly
python -m src.finetune_pretrained.finetune_pretrained \
  --model HuggingFaceTB/SmolLM-135M \
  --max-steps 2000

# Chat with the fine-tuned HF model
python -m src.inference.chat_pretrained --checkpoint checkpoints/finetune_pretrained/hf_model
```

## Individual Commands

### Setup
```bash
python scripts/setup.py
# Installs PyTorch (CUDA if available, else CPU) + all deps from requirements.txt
```

### Tokenizer
```bash
python -m src.tokenizer.tokenizer
# Trains byte-level BPE on data/raw/*.txt, *.jsonl
# Saves to data/processed/tokenizer/tokenizer.json
# Special tokens: <|pad|>=0, <|user|>=1, <|agent|>=2, <|end|>=3
# min_frequency: 2 makes adjacent pairs eligible only after at least two corpus occurrences
```

### Generate Synthetic Data
```bash
# From CSV/JSON/JSONL of facts
python -m src.generation.generate_data --input facts.csv --output data/raw/chat.jsonl

# Or generate synthetic facts
python -m src.generation.generate_data --num-facts 5000 --augmentations 5
# Outputs: data/raw/synthetic_chat.jsonl (+ train/val splits in data/processed/)
```

Synthetic facts are arbitrary combinations and are for checking the pipeline. Use a
real facts CSV/JSON file or the pretrained fine-tuning option for useful factual QA.

### Train from Scratch
```bash
python -m src.training.train \
  --config configs/model_config.yaml \
  --output-dir checkpoints/pretrain \
  --max-steps 10000 \
  --batch-size 4
```

**Key features:**
- Mixed precision (bf16/fp16/fp32 via config)
- AdamW (lr=3e-4, wd=0.1, β=(0.9,0.95))
- Cosine LR with 500-step warmup
- Gradient clipping (1.0)
- Gradient accumulation (8 steps by default)
- Gradient checkpointing
- Periodic eval (every 500 steps) + sample generation
- Resume from `checkpoints/pretrain/latest.pt`
- Dynamic batch padding and lazy JSONL reads reduce GPU and host RAM
- Loss projection is limited to answer positions to avoid a full `[batch, seq, vocab]` logits allocation

### Multi-GPU training

Launch one process per GPU from the project root. For two GPUs:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=2 -m src.training.train --max-steps 10000
```

The same launch pattern works for `src.training.finetune` and
`src.finetune_pretrained.finetune_pretrained`. `batch_size` is per GPU, so the
effective batch is `batch_size × grad_accum × number of GPUs`. Training uses NCCL
when available and otherwise falls back to Gloo.

### Fine-tune (Main Loop for New Data)
```bash
python -m src.training.finetune \
  --pretrained checkpoints/pretrain \
  --train-data data/processed/train.jsonl \
  --val-data data/processed/val.jsonl \
  --output-dir checkpoints/finetune \
  --max-steps 3000 \
  --lr 1e-4
```

**Key features:**
- Lower LR (1e-4), less weight decay (0.05)
- Early stopping (patience=3, min_delta=1e-4)
- Can freeze embeddings/layers via config
- Best model saved to `checkpoints/finetune/best_model.pt`

### Fine-tune Pretrained HF Model
```bash
python -m src.finetune_pretrained.finetune_pretrained \
  --model HuggingFaceTB/SmolLM-135M \
  --train-data data/processed/train.jsonl \
  --val-data data/processed/val.jsonl \
  --output-dir checkpoints/finetune_pretrained \
  --max-steps 2000 \
  --lr 5e-5
```

### Chat / Inference
```bash
# Interactive REPL
python -m src.inference.chat \
  --checkpoint checkpoints/finetune/best_model.pt \
  --temperature 0.7 --top-k 50 --top-p 0.9

# Single prompt (non-interactive)
python -m src.inference.chat \
  --checkpoint checkpoints/finetune/best_model.pt \
  --prompt "what is the capital of france?" \
  --strict

# REPL commands:
#   /help           - Show help
#   /config         - Show generation config
#   /strict on|off  - Toggle strict style checking
#   /temp 0.5       - Set temperature
#   /topk 40        - Set top-k
#   /topp 0.95      - Set top-p
```

**Strict mode (`--strict`)** flags violations:
- Missing ending period
- Question marks in response
- Value not lowercase (per config)
- Doesn't match template pattern
- Exceeds max length

## Expected Metrics (8GB GPU, e.g., RTX 3070/4070)

| Stage | Time | Memory | Loss (target) |
|-------|------|--------|---------------|
| Tokenizer training | ~30 sec | <1 GB | N/A |
| From-scratch training (10k steps) | 2-4 hours | ~6-7 GB | Train: ~1.5-2.0, Val: ~1.8-2.3 |
| Fine-tuning (3k steps) | 30-60 min | ~6-7 GB | Val: ~1.2-1.6 |
| HF Fine-tune (2k steps) | 20-40 min | ~5-6 GB | Val: ~1.0-1.4 |

**Loss interpretation:**
- ~3.0: Random guessing (vocab=16384, ln(16384)≈9.7, but masked loss lower)
- ~2.0: Learning patterns
- ~1.5: Good fit on synthetic data
- ~1.0: Excellent (near memorization)

## Troubleshooting

### OOM (Out of Memory)
```bash
# Reduce per-GPU batch size
--batch-size 2 --grad-accum 16  # Keeps the same effective batch with lower peak VRAM

# Enable gradient checkpointing (already on by default)
# Use fp16 instead of bf16 (slightly less stable but uses less memory)
# mixed_precision: "fp16" in config

# Reduce sequence length
max_seq_len: 256 in config

# Use 4-bit quantization (requires bitsandbytes)
# Not implemented here but can be added
```

### Loss Not Decreasing
1. **Check data format**: Ensure JSONL has `{"user": "...", "agent": "..."}` and agent matches style exactly
2. **Verify tokenizer**: Special tokens must be `<|user|>`, `<|agent|>`, `<|end|>`, `<|pad|>` with IDs 1,2,3,0
3. **Learning rate too high**: Try `1e-4` or `5e-5`
4. **Gradient accumulation**: Ensure effective batch size ≥ 32
5. **Check masking**: Loss should only be on agent span (verify in dataset.py)

### Loss Falls to Almost Zero Immediately

The old attention mask was reversed, allowing a token to see future answer tokens
while predicting them. A loss such as `0.0002` from that run is not evidence of
generalization. The causal attention path is fixed now. Regenerate the tokenizer and
data, then restart from a fresh checkpoint; old checkpoints were trained with the
wrong attention behavior. The data generator also splits facts before making question
variations, so related examples no longer leak between train and validation.

### Tokenizer Issues
```bash
# Retrain with more data
python -m src.generation.generate_data --num-facts 10000
python -m src.tokenizer.tokenizer

# Check vocab coverage
python -c "
from src.tokenizer.tokenizer import ChatTokenizer
tok = ChatTokenizer()
print('Vocab:', tok.vocab_size)
print('Special:', tok.pad_token_id, tok.user_token_id, tok.agent_token_id, tok.end_token_id)
"
```

### Model Outputs `<|user|>` Tokens
- **Cause**: Model hasn't learned chat template structure
- **Fix**: More training steps, ensure data uses exact template `<|user|>\n{user}\n<|agent|>\n{agent}<|end|>`
- **Check**: During training, sample generations should show proper format

### Model Outputs Gibberish / Repeats
- **Temperature too high**: Try `--temperature 0.3`
- **Repetition penalty**: Increase `--repetition-penalty 1.2`
- **Top-k/Top-p too restrictive**: Try `--top-k 100 --top-p 0.95`

### CUDA Not Detected
```bash
# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

# Reinstall with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Slow Training
- Enable `torch.compile` in config only if spare VRAM is available
- Use `bf16` on Ampere+ GPUs (A100, RTX 30/40 series)
- Increase `num_workers` from 2 if CPU loading is the bottleneck
- Use `pin_memory: true`

### `Unexpected key(s) ... "_orig_mod.*"` When Loading

Checkpoints written while the model was wrapped in `torch.compile` carry an
`_orig_mod.` prefix on every parameter name, and DDP writes `module.`. Loading is
handled: `strip_wrapper_prefixes` removes either prefix in `SmallLM.from_pretrained`
and in every training `load_checkpoint`, so old wrapped checkpoints resume without
rewriting. New checkpoints are saved from the unwrapped model, so they never carry
the prefix in the first place.

### Resume Training Fails
```bash
# Check checkpoint exists
ls -la checkpoints/pretrain/latest.pt

# Resume explicitly
python -m src.training.train --resume checkpoints/pretrain/step_5000.pt

# Or delete latest.pt and restart
rm checkpoints/pretrain/latest.pt
```

## Configuration

All settings in `configs/model_config.yaml`:

```yaml
model:           # Architecture
training:        # From-scratch training hyperparams
finetune:        # Fine-tuning hyperparams (lower LR, early stopping)
style:           # Response style (template, casing, validation rules)
data:            # Data loading params
inference:       # Default generation params
```

Memory-oriented defaults use a small per-GPU batch, gradient accumulation,
gradient checkpointing, dynamic padding, and lazy JSONL reads. `compile` is off by
default because it can increase peak memory; turn it on when you have spare VRAM.

**Style config example:**
```yaml
style:
  template: "The {property} of {subject} is {value}."
  value_casing: "lower"      # lower|upper|title|preserve
  require_period: true
  forbid_question_marks: true
  forbid_exclamation_marks: false
  max_response_length: 64
```

## Adding New Conversation Data

1. Create JSONL file with your conversations:
```jsonl
{"user": "what is the capital of spain?", "agent": "The capital of Spain is madrid."}
{"user": "who wrote hamlet?", "agent": "The author of hamlet is william shakespeare."}
```

2. Place in `data/raw/new_conversations.jsonl`

3. Run fine-tuning (the main loop):
```bash
python -m src.training.finetune \
  --train-data data/raw/new_conversations.jsonl \
  --val-data data/processed/val.jsonl \
  --pretrained checkpoints/finetune/best_model.pt \
  --output-dir checkpoints/finetune_v2
```

4. Test:
```bash
python -m src.inference.chat --checkpoint checkpoints/finetune_v2/best_model.pt --strict
```

## License

MIT License - Feel free to use, modify, and distribute.
