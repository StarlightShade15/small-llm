"""Chat with a Hugging Face model fine-tuned by finetune_pretrained.py."""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(checkpoint: str):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=dtype)
    model.config.use_cache = True
    model.to(device).eval()
    return model, tokenizer, device


@torch.inference_mode()
def answer(model, tokenizer, device, prompt, max_new_tokens, temperature, top_k, top_p):
    formatted = f"<|user|>\n{prompt}\n<|agent|>\n"
    inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=False).to(device)
    end_token_id = tokenizer.convert_tokens_to_ids("<|end|>")
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
        top_k=top_k,
        top_p=top_p,
        eos_token_id=end_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    new_tokens = generated[0, inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()


def main():
    parser = argparse.ArgumentParser(description="Chat with a fine-tuned Hugging Face model")
    parser.add_argument("--checkpoint", default="checkpoints/finetune_pretrained/hf_model")
    parser.add_argument("--prompt")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.checkpoint)
    if args.prompt:
        print(answer(model, tokenizer, device, args.prompt, args.max_tokens, args.temperature, args.top_k, args.top_p))
        return

    print("Hugging Face model chat (type 'quit' to exit)")
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.lower() in {"quit", "exit"}:
            break
        if prompt:
            response = answer(model, tokenizer, device, prompt, args.max_tokens, args.temperature, args.top_k, args.top_p)
            print(f"Agent: {response}")


if __name__ == "__main__":
    main()
