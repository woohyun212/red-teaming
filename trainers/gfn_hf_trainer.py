# gpt_hf_trainer.py
import os
import math
import json
import random
from typing import Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist
from torch.utils.data import Dataset

import wandb
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, pipeline
)

# --- 프로젝트 유틸 (당신 레포의 것 재사용) ---
from utils import lora_to_base, base_to_lora
from w00.utils import PresidioClassifier


# ===============================
# Utilities
# ===============================
def avg_pooling(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    denom = torch.clamp(mask.sum(1), min=1)
    return (last_hidden * mask).sum(1) / denom


@torch.no_grad()
def generate_and_return_z_logprob(
    model, prompt_ids, prompt_attention_mask, eos_token_id, temperature=1.0, max_len=64
):
    """
    온라인 샘플링: prompt에서 한 토큰씩 생성하며
    logZ와 Σlog p_f, actions(prompt+response+EOS)를 반환
    (gfn_hf_trainer의 로직을 FSDP/Trainer 친화적으로 정리)
    """
    device = prompt_ids.device
    B = prompt_ids.size(0)

    active = torch.ones(B, dtype=torch.bool, device=device)
    actions = prompt_ids.clone()
    state = prompt_ids.clone()
    sum_logpf = torch.zeros(B, dtype=torch.float32, device=device)
    attn = prompt_attention_mask.clone()

    # cache_position (Llama류만 필요하지만 gpt2에선 None)
    cache_position = None
    inputs = {
        "input_ids": state[:, :-1],
        "attention_mask": attn[:, :-1],
        "output_hidden_states": True,
        "use_cache": True,
    }
    outputs = model(**inputs)
    last_hidden_prompt = outputs["hidden_states"][-1]
    pkv = outputs["past_key_values"]

    # 첫 스텝: prompt 히든으로 log Z 계산
    avg = avg_pooling(last_hidden_prompt, prompt_attention_mask)
    log_z = model.proj_z(avg).squeeze(-1)

    for _ in range(max_len):
        out = model(
            input_ids=state[:, -1:],
            attention_mask=attn,
            past_key_values=pkv,
            use_cache=True,
        )
        pkv = out["past_key_values"]
        logits = out["logits"][:, -1, :]

        # 첫 토큰에서 EOS 금지
        if actions.size(1) == prompt_ids.size(1):
            logits[..., eos_token_id] = -torch.inf

        probs = F.softmax(logits / temperature, dim=-1)
        token_ids = torch.multinomial(probs, num_samples=1)

        logprob = F.log_softmax(logits, dim=-1).gather(-1, token_ids).squeeze(-1)
        logprob = torch.where(active, logprob, torch.zeros_like(logprob))
        sum_logpf = sum_logpf + logprob

        token_ids = torch.where(active.unsqueeze(-1), token_ids, torch.full_like(token_ids, eos_token_id))

        # 마스크/시퀀스 갱신
        masks = torch.where(active.unsqueeze(-1), torch.ones_like(token_ids), torch.zeros_like(token_ids))
        attn = torch.cat([attn.long(), masks], dim=1)
        actions = torch.cat([actions, token_ids], dim=1)
        state = actions  # ← 중복 이어붙임 버그 방지 (중요)

        # EOS 종료 체크
        active = active & (token_ids.squeeze(-1) != eos_token_id)
        if not torch.any(active):
            break

    # 항상 EOS 하나 더 붙여 문장 종료 보장 (패딩/채점 안정화)
    eos_tokens = torch.full((actions.size(0), 1), eos_token_id, dtype=torch.long, device=device)
    actions = torch.cat([actions, eos_tokens], dim=1)

    return {"actions": actions, "sum_logpf": sum_logpf, "log_z": log_z}


def _broadcast_text_list_from_rank0(texts: List[str], device) -> List[str]:
    """
    rank0에서만 victim을 호출하고, 문자열 리스트를 전체 rank에 브로드캐스트.
    직렬화는 \0-join + uint8 텐서 사용.
    """
    if not dist.is_initialized():
        return texts

    rank = dist.get_rank()
    world = dist.get_world_size()

    if rank == 0:
        payload = ("\0".join(texts)).encode("utf-8")
        size = torch.tensor([len(payload)], device=device, dtype=torch.long)
    else:
        payload = None
        size = torch.tensor([0], device=device, dtype=torch.long)

    dist.broadcast(size, src=0)
    buf = torch.empty(size.item(), dtype=torch.uint8, device=device)
    if rank == 0:
        buf[:] = torch.tensor(list(payload), dtype=torch.uint8, device=device)
    dist.broadcast(buf, src=0)

    if rank == 0:
        return texts
    return bytes(buf.tolist()).decode("utf-8").split("\0")


# ===============================
# Dataset (프롬프트 전용)
# ===============================
class PromptDataset(Dataset):
    def __init__(self, tokenizer, prompt_file: str):
        with open(prompt_file) as f:
            prompts = [x.strip() for x in f if x.strip()]
        enc = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
        self.enc = {k: v for k, v in enc.items()}

    def __len__(self):
        return self.enc["input_ids"].size(0)

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()}


# ===============================
# HF Trainer
# ===============================
class GFNTrainerHF(Trainer):
    """
    - attack: gpt2(+LoRA)
    - victim: 로컬 GPT-Neo 1.3B(base) + 로컬 LoRA 어댑터 → transformers.pipeline(text-generation)
    - 보상: LM 로그우도(attack의 base) + Presidio(PII)
    - 분산: victim 호출 rank0-only → broadcast
    """
    def __init__(
        self,
        *args,
        gfn_cfg: dict,
        victim_dir: str,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.gfn_cfg = gfn_cfg

        # attack tokenizer
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if hasattr(self.model.config, "use_cache") and self.model.config.use_cache:
            self.model.config.use_cache = False  # FSDP/GC 호환

        device_map = "auto"  # victim 파이프라인 내부에서 사용

        # --- Victim: GPT-Neo 1.3B(base) + local LoRA (victim_dir) ---
        base_model_id = "EleutherAI/gpt-neo-1.3B"
        self.vtm_tok = AutoTokenizer.from_pretrained(base_model_id, padding_side="left")
        if self.vtm_tok.pad_token_id is None:
            self.vtm_tok.pad_token_id = self.vtm_tok.eos_token_id

        base = AutoModelForCausalLM.from_pretrained(
            base_model_id, torch_dtype=torch.float16, device_map=device_map
        )
        # 로컬 LoRA 어댑터 적용 (victim_dir 가 LoRA 가중치 디렉터리)
        from peft import PeftModelForCausalLM
        vtm = PeftModelForCausalLM.from_pretrained(
            base, victim_dir, ignore_mismatched_sizes=True, device_map=device_map
        )
        vtm.resize_token_embeddings(self.vtm_tok.vocab_size)

        self.victim = pipeline(
            model=vtm,
            tokenizer=self.vtm_tok,
            task="text-generation",
            device_map=device_map,
            return_full_text=False,
            batch_size=gfn_cfg.get("victim_batch_size", 4),
            max_new_tokens=gfn_cfg["victim_max_len"],
            temperature=gfn_cfg["victim_temp"],
            top_p=gfn_cfg["victim_top_p"],
            pad_token_id=self.vtm_tok.pad_token_id,
            eos_token_id=self.vtm_tok.eos_token_id,
            do_sample=True,
        )

        # --- PII detector ---
        self.tox = PresidioClassifier(
            device=self.args.device,
            target_fields=["EMAIL_ADDRESS", "PHONE_NUMBER"],
        )

    # ========== Victim 호출 유틸 ==========
    def _make_prompt(self, instruction: str) -> str:
        # GPT-Neo는 단순 지시문 템플릿이 안정적
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            f"{instruction.rstrip()}\n\n### Response:\n"
        )

    def _victim_call_rank0(self, decoded_responses: List[str]) -> List[str]:
        # rank0에서만 pipeline 호출 → 전체 브로드캐스트
        device = self.model.device
        if not dist.is_initialized() or dist.get_rank() == 0:
            num = int(self.gfn_cfg["num_r_samples"])
            prompts = [self._make_prompt(x) for x in decoded_responses]
            rep_prompts = []
            for p in prompts:
                rep_prompts.extend([p] * num)
            pipe_out = self.victim(
                rep_prompts,
                do_sample=True,
                max_new_tokens=self.gfn_cfg["victim_max_len"],
                temperature=self.gfn_cfg["victim_temp"],
                top_p=self.gfn_cfg["victim_top_p"],
                pad_token_id=self.vtm_tok.pad_token_id,
                eos_token_id=self.vtm_tok.eos_token_id,
                return_full_text=False,
            )
            texts = []
            for o in pipe_out:
                texts.append(o.get("generated_text", o) if isinstance(o, dict) else str(o))
        else:
            texts = []

        texts = _broadcast_text_list_from_rank0(texts, device=device)
        return texts

    # ========== 보상 계산 ==========
    def _compute_lm_logreward(self, prompts_responses: torch.LongTensor, prompt_len: int) -> torch.Tensor:
        """
        LoRA-off (base gpt2)로 response 구간 로그우도 합.
        """
        with torch.no_grad():
            lora_to_base(self.model)
            labels = prompts_responses[:, prompt_len:]
            # 첫 번째 pad는 EOS → 그 뒤는 pad 로 간주
            pad_mask = (labels == self.tokenizer.pad_token_id).cumsum(1) > 1
            attn = torch.cat(
                [
                    torch.ones((labels.size(0), prompt_len), dtype=torch.long, device=labels.device),
                    (~pad_mask).long(),
                ],
                dim=1,
            )
            out = self.model(input_ids=prompts_responses, attention_mask=attn)
            logits = out.logits[:, prompt_len - 1 : -1]
            logp = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            logp = torch.where(pad_mask, torch.zeros_like(logp), logp)
            lm = logp.sum(1)
            # 최소 길이 페널티
            resp_len = (~pad_mask).long().sum(1)
            min_len = int(self.gfn_cfg["min_len"])
            lm = torch.where(resp_len < min_len, torch.full_like(lm, -500.0), lm)
            base_to_lora(self.model)
        return lm

    def _compute_pii_reward(self, victim_texts: List[str]) -> torch.Tensor:
        ver = int(self.gfn_cfg.get("pii_reward_version", 1))
        if ver == 2:
            empty = [[] for _ in victim_texts]
            scores = self.tox.compute_v2(
                responses=victim_texts,
                answer_lists=empty,
                alpha=self.gfn_cfg.get("pii_alpha", 1.0),
                beta=self.gfn_cfg.get("pii_beta", 1.0),
            )
        elif ver == 3:
            scores = self.tox.compute_v3(
                responses=victim_texts,
                alpha=self.gfn_cfg.get("pii_alpha", 1.0),
                c=self.gfn_cfg.get("pii_c", 1.0),
                entities=self.tox.fields,
                min_score=self.tox.th,
                language="en",
            )
        else:
            scores = self.tox.compute(responses=victim_texts)
        return torch.tensor(scores, device=self.model.device, dtype=torch.float32)

    # ========== 핵심: compute_loss ==========
    def compute_loss(self, model, inputs, return_outputs=False):
        # 1) 온라인 생성 (attack: gpt2+LoRA)
        batch = {k: v.to(model.device) for k, v in inputs.items()}
        out = generate_and_return_z_logprob(
            model=model,
            prompt_ids=batch["input_ids"],
            prompt_attention_mask=batch["attention_mask"],
            eos_token_id=self.tokenizer.eos_token_id,
            temperature=self.gfn_cfg.get("online_temp", 1.0),
            max_len=int(self.gfn_cfg["max_len"]),
        )
        pr = out["actions"]
        log_z = out["log_z"]
        sum_logpf = out["sum_logpf"]

        # 2) LM 로그우도 보상 (LoRA-off, base gpt2)
        prompt_len = batch["input_ids"].size(1)
        lm_logreward = self._compute_lm_logreward(pr, prompt_len)

        # 3) Victim 호출 (rank0-only → broadcast) → [B * r] → [B] 평균
        decoded = self.tokenizer.batch_decode(pr[:, prompt_len:], skip_special_tokens=True)
        victim_texts = self._victim_call_rank0(decoded)
        r = int(self.gfn_cfg["num_r_samples"])
        pii_scores = self._compute_pii_reward(victim_texts).view(-1, r).mean(1)

        # 4) 스케줄/템퍼 적용 + TB loss
        gamma = float(self.gfn_cfg.get("lm_sched", 1.0))
        beta = float(self.gfn_cfg.get("beta", 1.0))
        rew_temp = float(self.gfn_cfg.get("rew_temp", 1.0))

        log_reward = (lm_logreward / gamma) + (pii_scores / beta)
        tempered = log_reward / rew_temp

        delta = log_z + sum_logpf - tempered
        loss = (delta ** 2).mean()

        # 로깅
        self.log({
            "loss/train": loss.detach().item(),
            "lm_log_reward": lm_logreward.mean().item(),
            "pii_log_reward": pii_scores.mean().item(),
            "log_reward": log_reward.mean().item(),
        })

        return (loss, {"logits": None}) if return_outputs else loss


# ===============================
# Build attack model/tokenizer
# ===============================
def build_attack_model_and_tok(sft_ckpt: str, lora_r=8, lora_alpha=16, lora_dropout=0.05):
    # gpt2 고정
    config = AutoConfig.from_pretrained("gpt2")
    config.use_cache = True
    base = AutoModelForCausalLM.from_pretrained(sft_ckpt, config=config, device_map="auto")

    lconf = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(base, lconf)

    # proj_z 추가
    hidden = getattr(model.config, "n_embd", getattr(model.config, "hidden_size", None))
    if hidden is None:
        raise ValueError("Cannot infer hidden size for proj_z.")
    model.proj_z = nn.Linear(hidden, 1).to(next(model.parameters()).device)

    tok = AutoTokenizer.from_pretrained(sft_ckpt, padding_side="left")
    tok.pad_token_id = tok.eos_token_id
    return model, tok


# ===============================
# Entrypoint helper
# ===============================
def make_trainer(
    prompt_file: str,
    victim_lora_dir: str,
    sft_ckpt: str,
    training_args: TrainingArguments,
    gfn_cfg: dict,
):
    model, tokenizer = build_attack_model_and_tok(
        sft_ckpt,
        lora_r=gfn_cfg.get("lora_r", 8),
        lora_alpha=gfn_cfg.get("lora_alpha", 16),
        lora_dropout=gfn_cfg.get("lora_dropout", 0.05),
    )

    dataset = PromptDataset(tokenizer, prompt_file)

    trainer = GFNTrainerHF(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        gfn_cfg=gfn_cfg,
        victim_dir=victim_lora_dir,
    )
    return trainer


# ===============================
# Example usage (script)
# ===============================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--victim_lora_dir", type=str, required=True, help="로컬 GPT-Neo LoRA 가중치 디렉터리")
    parser.add_argument("--sft_ckpt", type=str, required=True, help="attack(gpt2) SFT 체크포인트 또는 'gpt2'")
    parser.add_argument("--output_dir", type=str, default="runs/exp_gpt2")
    parser.add_argument("--wandb_project", type=str, default="red-team")
    parser.add_argument("--exp_name", type=str, default="gpt2_gfn_trainer")

    # FSDP/하이퍼
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)

    # GFN cfg
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--min_len", type=int, default=8)
    parser.add_argument("--online_temp", type=float, default=1.0)
    parser.add_argument("--num_r_samples", type=int, default=2)
    parser.add_argument("--victim_max_len", type=int, default=64)
    parser.add_argument("--victim_temp", type=float, default=0.9)
    parser.add_argument("--victim_top_p", type=float, default=0.9)

    parser.add_argument("--lm_sched", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--rew_temp", type=float, default=1.0)
    parser.add_argument("--pii_reward_version", type=int, default=1)
    parser.add_argument("--pii_alpha", type=float, default=1.0)
    parser.add_argument("--pii_beta", type=float, default=1.0)
    parser.add_argument("--pii_c", type=float, default=1.0)

    # LoRA(attack)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # FSDP 스위치
    parser.add_argument("--use_fsdp", action="store_true")

    args = parser.parse_args()

    wandb.init(project=args.wandb_project, name=args.exp_name, reinit=True)

    # --- TrainingArguments (+선택적 FSDP) ---
    fsdp_kwargs = {}
    fsdp_flag = "full_shard auto_wrap" if args.use_fsdp else None
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=50,
        save_steps=1000,
        evaluation_strategy="no",
        report_to=["wandb"],
        bf16=True,
        gradient_checkpointing=True,
        fsdp=fsdp_flag,
        fsdp_config={
            "forward_prefetch": True,
            "use_orig_params": True,
            "cpu_offload": False,
            "limit_all_gathers": True,
        } if args.use_fsdp else None,
        fsdp_transformer_layer_cls_to_wrap="GPT2Block",
    )

    gfn_cfg = dict(
        # generation/length
        max_len=args.max_len, min_len=args.min_len, online_temp=args.online_temp,
        # victim sampling
        num_r_samples=args.num_r_samples, victim_max_len=args.victim_max_len,
        victim_temp=args.victim_temp, victim_top_p=args.victim_top_p,
        # rewards/temper
        lm_sched=args.lm_sched, beta=args.beta, rew_temp=args.rew_temp,
        pii_reward_version=args.pii_reward_version, pii_alpha=args.pii_alpha,
        pii_beta=args.pii_beta, pii_c=args.pii_c,
        # lora(attack)
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
    )

    trainer = make_trainer(
        prompt_file=args.prompt_file,
        victim_lora_dir=args.victim_lora_dir,
        sft_ckpt=args.sft_ckpt,
        training_args=training_args,
        gfn_cfg=gfn_cfg,
    )

    trainer.train()
    trainer.save_model()
    wandb.finish()