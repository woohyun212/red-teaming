

#!/usr/bin/env python3
"""
Attack Prompt Generator (black-box):

- Loads a trained attack (generator) model and its tokenizer
- Reads seed prompts (txt or jsonl)
- Generates `num_per_prompt` attack prompts per seed
- Saves outputs to JSONL for downstream black-box querying

Usage:
    python atk_prmpt_generator.py \
        --attack_ckpt /path/to/attack_model \
        --tokenizer_ckpt /path/to/tokenizer (optional) \
        --input_prompts ./prompts.txt \
        --out ./attack_prompts.jsonl \
        --num-per-prompt 5 \
        --batch-size 8 \
        --max-new-tokens 256 \
        --temperature 1.0 \
        --top-p 0.95 \
        --seed 42

Input formats:
- .txt   : one prompt per line
- .jsonl : each line should contain {"prompt": "..."}

Output JSONL schema (one line per generated attack prompt):
{
  "seed_index": int,            # index of the input seed prompt
  "seed_prompt": str,           # the input seed prompt
  "attack_prompt": str,         # generated attack prompt text
  "gen_idx": int,               # 0..(num_per_prompt-1)
  "meta": {
      "attack_ckpt": str,
      "tokenizer_ckpt": str | null,
      "max_new_tokens": int,
      "temperature": float,
      "top_p": float,
      "num_per_prompt": int,
      "batch_size": int,
      "seed": int
  }
}
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import math
import random
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_prompts(path: str) -> List[str]:
    prompts: List[str] = []
    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(line)
    elif path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if "prompt" in obj and isinstance(obj["prompt"], str):
                    prompts.append(obj["prompt"])
    else:
        raise ValueError(f"Unsupported input format: {path}")
    if not prompts:
        raise ValueError("No prompts loaded from input file.")
    return prompts


def save_jsonl(path: str, records: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class AttackPromptGenerator:
    def __init__(
        self,
        attack_ckpt: str,
        tokenizer_ckpt: str | None = None,
        device: str = "cuda",
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.95,
    ):
        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_ckpt or attack_ckpt)
        self.model = AutoModelForCausalLM.from_pretrained(attack_ckpt)
        self.model.to(self.device)
        # ensure PAD token exists
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # last resort: add a pad token
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
                self.model.resize_token_embeddings(len(self.tokenizer))
        self.gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

    @torch.no_grad()
    def generate(self, seeds: List[str], num_per_prompt: int = 5, batch_size: int = 8) -> List[Dict[str, Any]]:
        """Return list of dicts with seed_index, seed_prompt, attack_prompt, gen_idx."""
        results: List[Dict[str, Any]] = []
        n = len(seeds)
        # Process in mini-batches for memory efficiency
        for i in range(0, n, batch_size):
            batch = seeds[i:i+batch_size]
            # replicate each seed prompt `num_per_prompt` times
            rep_prompts: List[str] = []
            parent_idx: List[int] = []
            per_prompt_gen_idx: List[int] = []
            for j, p in enumerate(batch):
                for k in range(num_per_prompt):
                    rep_prompts.append(p)
                    parent_idx.append(i + j)
                    per_prompt_gen_idx.append(k)
            tok = self.tokenizer(rep_prompts, return_tensors="pt", padding=True, truncation=False)
            tok = {k: v.to(self.device) for k, v in tok.items()}
            out = self.model.generate(**tok, **self.gen_kwargs)
            # slice generated continuation only
            cont = out[:, tok['input_ids'].shape[1]:]
            decoded = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
            for text, idx, k in zip(decoded, parent_idx, per_prompt_gen_idx):
                results.append({
                    "seed_index": idx,
                    "seed_prompt": seeds[idx],
                    "attack_prompt": text.strip(),
                    "gen_idx": k,
                })
        return results


def parse_args():
    p = argparse.ArgumentParser(description="Black-box Attack Prompt Generator")
    p.add_argument("--attack_ckpt", required=True, help="Path or HF id of the trained attack model")
    p.add_argument("--tokenizer_ckpt", default=None, help="Optional tokenizer checkpoint")
    p.add_argument("--input_prompts", required=True, help=".txt or .jsonl with seed prompts")
    p.add_argument("--out", required=True, help="Output JSONL path")
    p.add_argument("--num-per-prompt", type=int, default=5, help="#generations per seed prompt")
    p.add_argument("--batch-size", type=int, default=8, help="Batch size (seeds) for generation")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    seeds = read_prompts(args.input_prompts)

    gen = AttackPromptGenerator(
        attack_ckpt=args.attack_ckpt,
        tokenizer_ckpt=args.tokenizer_ckpt,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    outputs = gen.generate(
        seeds=seeds,
        num_per_prompt=args.num_per_prompt,
        batch_size=args.batch_size,
    )

    # attach meta once per record
    meta = {
        "attack_ckpt": args.attack_ckpt,
        "tokenizer_ckpt": args.tokenizer_ckpt,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_per_prompt": args.num_per_prompt,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }
    for r in outputs:
        r["meta"] = meta

    save_jsonl(args.out, outputs)
    print(f"[ok] wrote {len(outputs)} generations to {args.out}")


if __name__ == "__main__":
    main()