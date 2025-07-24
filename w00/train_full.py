from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import load_from_disk
import torch

model_name = "meta-llama/Llama-2-7b-chat-hf"
train_dir  = "data/train"
val_dir    = "data/val"

tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tok.pad_token_id = tok.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,       # fp16도 가능
)

train_ds = load_from_disk(train_dir)
val_ds   = load_from_disk(val_dir)

data_collator = DataCollatorForLanguageModeling(tok, mlm=False)

args = TrainingArguments(
    output_dir="save/full_ft",
    per_device_train_batch_size=2,      # 8B 모델이면 80GB에서 2~3 토큰 1024 길이
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=16,     # → 가상 배치 32
    learning_rate=1e-5,
    num_train_epochs=1,
    lr_scheduler_type="cosine",
    warmup_steps=500,
    logging_steps=50,
    save_strategy="steps",
    save_steps=1000,
    # evaluation_strategy="steps",
    eval_steps=1000,
    bf16=True,                          # mixed_precision에서 지정한 것과 일치
    max_grad_norm=1.0,
    gradient_checkpointing=True,        # 메모리 추가 절감(약간 느려짐)
    report_to="none",                   # wandb 사용 시 "wandb"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
)
trainer.train()
trainer.save_model("save/full_ft/latest")
tok.save_pretrained("save/full_ft/latest")
