#!/usr/bin/env python
"""
Quick interactive generation script to sanity-check a victim model.

Usage examples:

  # 1) Minimal
  python victim_test.py --model ./save/gpt-neo-enron-e4-5k --prompt "The email address of John Doe is"

  # 2) With sampling controls
  python victim_test.py \
    --model ./save/gpt2-sft-pii-v1/latest \
    --prompt "The email address of John Doe is" \
    --max_new_tokens 80 --temperature 0.8 --top_p 0.9 --top_k 50 --seed 42

  # 3) Batch from file (one prompt per line)
  python victim_test.py --model ./save/gpt-neo-enron-e4-5k --prompt_file prompts.txt --n 1

If both --prompt and --prompt_file are omitted, the script will enter an interactive REPL.
"""

import argparse
import os
import sys
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path or hub id of the model to load")
    p.add_argument("--prompt", default=None, help="Single prompt (overrides --prompt_file)")
    p.add_argument("--prompt_file", default=None, help="Text file with one prompt per line")
    p.add_argument("--n", type=int, default=1, help="#generations per prompt")
    p.add_argument("--max_new_tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--device", default="auto", help="'auto' or an explicit device like cuda:0 or cpu")
    return p.parse_args()


def set_seed(seed: int):
    if seed is None or seed < 0:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_dtype(dtype_str: str):
    if dtype_str == "auto":
        if torch.cuda.is_available():
            # prefer bfloat16 if supported, else float16
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32
    return getattr(torch, dtype_str)


def load_model_and_tokenizer(model_id: str, dtype, device_str: str):
    device_map = "auto" if device_str == "auto" else None
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    # Ensure pad token exists for generation (common for GPT-style models)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # As a last resort, add a pad token
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            model.resize_token_embeddings(len(tokenizer))

    # Left padding is often safer for causal LMs with attention_mask
    tokenizer.padding_side = "left"

    # If a manual device is requested, move model
    if device_map is None:
        model.to(device_str)

    return model, tokenizer


def generate(model, tokenizer, prompts: List[str], n: int, max_new_tokens: int, temperature: float,
             top_p: float, top_k: int, repetition_penalty: float) -> List[List[str]]:
    model.eval()
    outputs_per_prompt: List[List[str]] = []
    with torch.no_grad():
        for prompt in prompts:
            batch_inputs = tokenizer([prompt] * n, return_tensors="pt", padding=True).to(model.device)
            gen_ids = model.generate(
                **batch_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True if temperature > 0 else False,
                temperature=max(1e-8, temperature),
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            # Keep only the generated continuation per item by stripping the prompt prefix once
            cleaned: List[str] = []
            for t in texts:
                if t.startswith(prompt):
                    cleaned.append(t[len(prompt):])
                else:
                    cleaned.append(t)
            outputs_per_prompt.append(cleaned)
    return outputs_per_prompt


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.prompt is not None:
        prompts = [args.prompt]
    elif args.prompt_file is not None:
        assert os.path.exists(args.prompt_file), f"Prompt file not found: {args.prompt_file}"
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompts = [line.rstrip("\n") for line in f if line.strip()]
        if not prompts:
            print("[warn] Prompt file is empty after stripping blanks.")
    else:
        # REPL mode
        print("[REPL] Type a prompt and press Enter (Ctrl-D/Ctrl-Z to exit).\n")
        prompts = []

    dtype = resolve_dtype(args.dtype)
    model, tokenizer = load_model_and_tokenizer(args.model, dtype=dtype, device_str=args.device)

    # Print basic model info
    n_params = sum(p.numel() for p in model.parameters())
    printable_dtype = str(dtype).replace("torch.", "")
    device_desc = model.device if args.device != "auto" else "auto-map"
    print(f"Loaded model: {args.model}\n  params: {n_params:,}\n  dtype: {printable_dtype}\n  device: {device_desc}\n")

    if not prompts:
        try:
            while True:
                prompt = input("prompt> ").strip()
                if not prompt:
                    continue
                outs = generate(
                    model, tokenizer, [prompt], args.n,
                    args.max_new_tokens, args.temperature, args.top_p, args.top_k, args.repetition_penalty
                )[0]
                for i, o in enumerate(outs):
                    print(f"=== generation {i+1}/{args.n} ===\n{o}\n")
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            return
    else:
        outs = generate(
            model, tokenizer, prompts, args.n,
            args.max_new_tokens, args.temperature, args.top_p, args.top_k, args.repetition_penalty
        )
        for p, gens in zip(prompts, outs):
            print("\n" + "#" * 80)
            print(f"PROMPT: {p}")
            for i, o in enumerate(gens):
                print(f"\n=== generation {i+1}/{args.n} ===\n{o}")


if __name__ == "__main__":
    main()