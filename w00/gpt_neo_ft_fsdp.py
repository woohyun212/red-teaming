import os
import torch
import wandb
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

MODEL_NAME = "EleutherAI/gpt-neo-2.7B"

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------- 1) enron.jsonl 데이터 로딩 ---------
# 각 줄이 {"id": ..., "text": "...", "pii": {...}} 형태라고 가정
raw_dataset = load_dataset(
    "json",
    data_files={"train": "enron.jsonl"},
)["train"]

# train / validation split
dataset_dict = raw_dataset.train_test_split(test_size=0.01, seed=42)
train_dataset = dataset_dict["train"]
eval_dataset = dataset_dict["test"]

print(train_dataset[0])  # 디버깅용: {'id': ..., 'text': '...', 'pii': {...}}

# --------- 2) 토크나이저 / 모델 ---------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)

max_length = 2048  # context 길이, 최대 2048

# text 필드만 사용해서 토크나이즈
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

# train / eval 각각 토크나이즈
train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    num_proc=4,
    remove_columns=train_dataset.column_names,  # id, text, pii 등 전부 제거하고 tokenized만 남김
)

eval_dataset = eval_dataset.map(
    tokenize_function,
    batched=True,
    num_proc=4,
    remove_columns=eval_dataset.column_names,
)

# --------- 3) Data Collator (Causal LM) ---------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # GPT 계열은 MLM 아님
)

# --------- 4) FSDP 설정이 들어간 TrainingArguments ---------
training_args = TrainingArguments(
    output_dir="./gptneo-2.7b-enron-fsdp",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_steps=50,
    save_steps=1000,
    save_total_limit=2,
    num_train_epochs=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=100,
    lr_scheduler_type="cosine",

    fp16=True,
    bf16=False,
    gradient_checkpointing=True,
    report_to=["wandb"],
    run_name="gptneo-2.7b-enron-fsdp",

    # ------- FSDP 관련 -------
    fsdp="full_shard auto_wrap",
    fsdp_config={
        "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        "fsdp_backward_prefetch": "BACKWARD_PRE",
        "fsdp_state_dict_type": "FULL_STATE_DICT",
        # 필요하면 CPU offload 켜기
        # "fsdp_cpu_offload": True,
        # "fsdp_limit_all_gathers": True,
    },
)

# --------- 5) Trainer ---------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model("./gptneo-2.7b-enron-fsdp-final")
    tokenizer.save_pretrained("./gptneo-2.7b-enron-fsdp-final")