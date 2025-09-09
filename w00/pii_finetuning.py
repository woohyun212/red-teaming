from datasets import load_dataset
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from transformers import (AutoModelForCausalLM, TrainingArguments, Trainer,
                          DataCollatorForLanguageModeling)
from transformers import DataCollatorWithPadding
from peft import LoraConfig, get_peft_model

from random import randint
import torch
from pprint import  pp
import json
TRAIN_NAME = "email_lora"
SAVE_PATH = f"save/{TRAIN_NAME}"
DATA_PATH = f"data/v6_enron_pii.jsonl"
SAMPLE_SIZE = 50000
SAMPLE_OUTPUT_PATH = "data/sampled_dataset.jsonl"
suffle_seed = randint(0, 50)


def merge_fields(example):
    # 한 통(e-mail) → 하나의 plain text로 합치기
    # (선택) 헤더까지 포함하고 싶으면 아래처럼 붙이기
    # hdr = example["headers"]
    # header_txt = f"From: {hdr['x_from']} <{hdr['from_email']}>\n" \
    #              f"To: {hdr['x_to']} <{', '.join(hdr['to_email'])}>\n" \
    #              f"Date: {hdr['date']}\n" \
    #              f""
    # example["text"] = header_txt + example["text"]
    # pp(dict(example)["text"])
    return {"text": example["text"]}



def tokenize(sample):
    return tokenizer(sample["text"])


def group_texts(examples):
    concat = sum(examples["input_ids"], [])
    total = len(concat) // block_size * block_size
    result = [concat[i: i + block_size] for i in range(0, total, block_size)]
    return {"input_ids": result, "labels": result}


# 데이터 로딩
print("데이터 로드")
raw_ds = load_dataset(
    "json",
    data_files=DATA_PATH,
    split="train"
)
print("샘플링")
raw_ds = raw_ds.shuffle(seed=suffle_seed)  # 전체를 무작위 섞은 뒤
raw_ds = raw_ds.select(range(SAMPLE_SIZE))  # 앞에서 pick

# JSONL 파일로 저장
raw_ds.to_json(SAMPLE_OUTPUT_PATH, orient="records", lines=True, force_ascii=False)
print(f"샘플링된 데이터가 {SAMPLE_OUTPUT_PATH} 로 저장되었습니다.")

print("데이터 변환")
# 단일 text 생성
raw_ds = raw_ds.map(merge_fields, remove_columns=raw_ds.column_names)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer.pad_token = tokenizer.eos_token  # Llama-2는 eos(</s>)를 pad로 재사용

tokenized = raw_ds.map(tokenize, batched=True, remove_columns=["text"])

# 토크나이즈 & 2048-토큰 블록화
# 이메일 길이가 제각각이므로 토큰을 쭉 연결한 뒤 2 048개씩 잘라서 학습 효율 ↑
block_size = 2048

lm_ds = tokenized.map(group_texts, batched=True, batch_size=1000, remove_columns=tokenized.column_names)

# Lora Setting
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    quantization_config=bnb_cfg,
    device_map="auto"
)

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_cfg)

training_args = TrainingArguments(
    output_dir=SAVE_PATH,
    per_device_train_batch_size=4,  # A100 80 GB 기준; GPU 여유에 맞춰 조정
    gradient_accumulation_steps=8,
    num_train_epochs=1,  # 이메일 전체를 여러 epoch 돌리면 과도한 memorisation 위험↑
    learning_rate=2e-4,
    fp16=True,
    logging_steps=25,
    save_strategy="epoch"
)

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


def clm_collator(features):
    pad_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8  # TensorCore 최적화 (선택)
    )
    batch = pad_collator(features)
    # labels = input_ids  복사
    batch["labels"] = batch["input_ids"].clone()
    # pad( attention_mask == 0 ) → ignore_index(-100)
    batch["labels"][batch["attention_mask"] == 0] = -100
    return batch

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_ds,
    data_collator=collator
)
trainer.train()
trainer.save_model(f"{SAVE_PATH}/latest")
