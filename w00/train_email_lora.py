# train_email_lora.py  (patched)

import argparse, torch, os
from datasets import load_from_disk
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# import bitsandbytes as bnb
from torch.nn.utils.rnn import pad_sequence
import wandb

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)    # 사용할 pretrained LLM 경로/이름
    p.add_argument("--train_dir", required=True)     # 전처리된 학습 데이터(arrow 포맷) 경로
    p.add_argument("--val_dir", required=True)       # 검증 데이터 경로
    p.add_argument("--output_dir", required=True)    # 학습 결과(모델/토크나이저) 저장 경로
    p.add_argument("--batch_size", type=int, default=4)    # GPU 1장당 미니배치 크기
    p.add_argument("--grad_acc_steps", type=int, default=8)# gradient accumulation step
    p.add_argument("--lr", type=float, default=2e-5)       # 학습률
    p.add_argument("--epoch", type=int, default=1)         # 전체 epoch 수
    p.add_argument("--lora_r", type=int, default=16)       # LoRA의 low-rank 차원 수(r)
    p.add_argument("--lora_alpha", type=int, default=32)   # LoRA scaling factor
    p.add_argument("--lora_dropout", type=float, default=0.05) # LoRA dropout
    p.add_argument("--bits", type=int, choices=[4, 8, 16, 32], default=16,
                   help="4/8-bit 양자화, 16=bfloat16, 32=float32") # 모델 파라미터 비트수/양자화 방식
    p.add_argument("--wandb_project", default="red-team", help='W&B project name')
    p.add_argument("--run_name", default="email-lora")

    return p.parse_args()


# ---------- 모델 로더 ----------
def load_model(base, bits):
    dtype_map = {16: torch.bfloat16, 32: torch.float32}
    if bits in (4, 8):
        model = AutoModelForCausalLM.from_pretrained(
            base,
            load_in_4bit=(bits == 4),
            load_in_8bit=(bits == 8),
            # device_map="auto",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        # 필수: k-bit 모델을 LoRA 학습용으로 준비
        model = prepare_model_for_kbit_training(model)
        return model
    else:

        return AutoModelForCausalLM.from_pretrained(
            base,
            torch_dtype=dtype_map[bits],
            # device_map="auto"
        )

# ---------- 안전한 collator ----------
def make_collator(tokenizer):
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    def collate(batch):
        ids  = [torch.tensor(b["input_ids"])      for b in batch]
        msk  = [torch.tensor(b["attention_mask"]) for b in batch]
        ids  = pad_sequence(ids, batch_first=True, padding_value=pad_id)
        msk  = pad_sequence(msk, batch_first=True, padding_value=0)
        labels = ids.clone()
        labels[labels == pad_id] = -100   # CrossEntropy에서 무시할 패딩
        return {"input_ids": ids, "attention_mask": msk, "labels": labels}
    return collate

# ---------- 메인 ----------
def main():
    args = parse_args()
    # wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
    wandb.init(reinit=True, config=vars(args),
               project=args.wandb_project, name=args.run_name)

    train_ds = load_from_disk(args.train_dir)
    val_ds   = load_from_disk(args.val_dir)

    # tok = AutoTokenizer.from_pretrained(args.base_model, padding_side="left")
    # tok.pad_token_id = tok.eos_token_id
    #
    # base_model = load_model(args.base_model, args.bits)

    tok = AutoTokenizer.from_pretrained(args.base_model, padding_side="left")

    # pad 토큰이 이미 있으면 그대로 사용 (Llama-2 는 0)
    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": "<pad>"})
        pad_added = True
    else:
        pad_added = False

    # 모델 임베딩 크기를 tokenizer 길이에 맞춰 늘리기
    base_model = load_model(args.base_model, args.bits)
    if pad_added:
        base_model.resize_token_embeddings(len(tok))

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc_steps,
        num_train_epochs=args.epoch,
        learning_rate=args.lr,
        lr_scheduler_type="cosine", #  러닝레이트 변화 방식
        warmup_ratio=0.03,
        max_grad_norm=1.0, # gradient clipping
        logging_steps=50,
        save_strategy="steps",
        save_steps=500,
        bf16=(args.bits == 16),
        fp16=(args.bits == 32),
        optim="adamw_bnb_8bit" if args.bits in (4, 8) else "adamw_torch",
        report_to="wandb",
        run_name=args.run_name
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=make_collator(tok),
    )

    trainer.train()
    wandb.finish()

    model.save_pretrained(os.path.join(args.output_dir, "latest"))
    tok.save_pretrained(os.path.join(args.output_dir, "latest"))

if __name__ == "__main__":
    main()

"""
CUDA_VISIBLE_DEVICES=0  python w00/train_email_lora.py \
  --base_model meta-llama/Llama-2-7b-chat-hf \
  --train_dir data/train --val_dir data/val \
  --output_dir save/email-lora-v2 \
  --batch_size 4             \
  --grad_acc_steps 8         \
  --lr 2e-5                  \
  --epoch 1                  \
  --lora_r 8                 \
  --lora_alpha 32            \
  --lora_dropout 0.05        \
  --bits 16                  
"""