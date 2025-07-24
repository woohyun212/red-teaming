# build_email_dataset.py
from datasets import load_dataset, disable_caching
from pathlib import Path
import random, json

disable_caching()                 # 캐시 파일 생성을 막아 디스크 사용 최소화
raw_path = "data/v3_email_pii.jsonl" # 앞서 만든 JSONL
train_dir = Path("data/train")    # Arrow 셋이 저장될 경로
val_dir   = Path("data/val")

# ① JSONL 로드
ds = load_dataset("json", data_files=raw_path, split="train")

# ② 95 : 5 분할
ds = ds.train_test_split(test_size=0.05, seed=42)
train_ds, val_ds = ds["train"], ds["test"]

# ③ 토크나이즈 함수
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

def tok_func(batch):
    return tok(batch["text"], truncation=True, max_length=1024)

train_ds = train_ds.map(tok_func, batched=True, remove_columns=["text"])
val_ds   = val_ds.map(tok_func,   batched=True, remove_columns=["text"])

# ④ Arrow 저장
train_ds.save_to_disk(train_dir)
val_ds.save_to_disk(val_dir)

print(f"✅  saved → {train_dir} / {val_dir}  "
      f"({len(train_ds):,} train  |  {len(val_ds):,} val)")


"""
내가 가진 jsonl 이메일 데이터셋을,
- (1) train/val 분할
- (2) Llama2용 토크나이저로 토크나이즈
- (3) Arrow 파일로 저장
- 파인튜닝용 데이터셋(Huggingface Trainer에서 바로 사용 가능)으로 자동 가공하는 스크립트임

python ./w00/build_email_dataset.py

"""