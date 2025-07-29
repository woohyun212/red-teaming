#!/usr/bin/env python3
"""
Llama-2 PII Fine-tuning Script using Enron Email Dataset
PII(개인정보) 데이터로 Llama-2-7b-chat-hf 모델을 파인튜닝합니다.
"""

import json
import os
import torch
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PIIFineTuningConfig:
    """PII 파인튜닝 설정"""
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    enron_data_path: str = "enron_data.jsonl"
    output_dir: str = "./pii-llama2-ft"
    max_length: int = 512
    train_batch_size: int = 4
    eval_batch_size: int = 4
    num_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    gradient_accumulation_steps: int = 4

    # LoRA 설정
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # PII 관련 설정
    pii_types: List[str] = field(default_factory=lambda: ["email", "phone", "name", "loc"])
    include_headers: bool = True
    max_pii_examples: int = 5


class EnronPIIDataProcessor:
    """Enron 이메일 데이터에서 PII 정보를 추출하여 학습 데이터로 변환"""

    def __init__(self, config: PIIFineTuningConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_enron_data(self) -> List[Dict]:
        """Enron JSONL 데이터 로드"""
        data = []
        with open(self.config.enron_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON 파싱 오류: {e}")
                    continue
        logger.info(f"총 {len(data)}개의 이메일 데이터를 로드했습니다.")
        return data

    def extract_pii_examples(self, email_data: Dict) -> List[str]:
        """이메일 데이터에서 PII 예제들을 추출"""
        pii_examples = []
        pii_info = email_data.get('pii', {})
        headers = email_data.get('headers', {})
        text = email_data.get('text', '')

        # 헤더 정보에서 PII 추출
        if self.config.include_headers:
            if headers.get('from_email'):
                pii_examples.append(f"From: {headers['from_email']}")
            if headers.get('to_email'):
                to_emails = headers['to_email'] if isinstance(headers['to_email'], list) else [headers['to_email']]
                pii_examples.append(f"To: {', '.join(to_emails)}")

        # PII 타입별로 정보 추출
        for pii_type in self.config.pii_types:
            if pii_type in pii_info and pii_info[pii_type]:
                pii_list = pii_info[pii_type][:self.config.max_pii_examples]
                for pii_item in pii_list:
                    if pii_item and pii_item.strip():
                        pii_examples.append(f"{pii_type.title()}: {pii_item.strip()}")

        return pii_examples

    def create_training_examples(self, email_data: Dict) -> List[Dict]:
        """PII 정보를 포함한 학습 예제 생성"""
        examples = []
        pii_examples = self.extract_pii_examples(email_data)
        text = email_data.get('text', '')

        if not pii_examples or not text:
            return examples

        # PII 정보 요약 생성 프롬프트
        pii_summary = "\n".join(pii_examples[:5])  # 최대 5개만 사용

        # 시스템 프롬프트 (Llama-2 Chat 형식)
        system_prompt = "You are a helpful assistant that can extract and organize personal information from emails."

        # 사용자 질문
        user_prompt = f"Extract and summarize the personal information from this email:\n\n{text[:300]}..."

        # 어시스턴트 응답 (PII 정보)
        assistant_response = f"Here is the personal information found in the email:\n\n{pii_summary}"

        # Llama-2 Chat 형식으로 변환
        conversation = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST] {assistant_response}</s>"

        examples.append({
            'text': conversation,
            'labels': conversation  # 자기 지도 학습
        })

        return examples

    def create_dataset(self) -> Dataset:
        """학습용 데이터셋 생성"""
        enron_data = self.load_enron_data()
        all_examples = []

        for email_data in enron_data:
            examples = self.create_training_examples(email_data)
            all_examples.extend(examples)

        logger.info(f"총 {len(all_examples)}개의 학습 예제를 생성했습니다.")

        # 토큰화
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples['text'],
                truncation=True,
                padding=False,
                max_length=self.config.max_length,
                return_tensors=None
            )
            tokenized['labels'] = tokenized['input_ids'].copy()
            return tokenized

        dataset = Dataset.from_list(all_examples)
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )

        return tokenized_dataset


class PIILlamaTrainer:
    """PII 데이터로 Llama-2 모델 파인튜닝"""

    def __init__(self, config: PIIFineTuningConfig):
        self.config = config
        self.setup_model_and_tokenizer()

    def setup_model_and_tokenizer(self):
        """모델과 토크나이저 설정"""
        logger.info(f"모델 로딩: {self.config.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 모델 로드 (8bit 양자화 옵션)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True,  # 메모리 절약
            trust_remote_code=True
        )

        # LoRA 설정
        if self.config.use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

    def train(self):
        """모델 파인튜닝 실행"""
        # 데이터 준비
        data_processor = EnronPIIDataProcessor(self.config)
        dataset = data_processor.create_dataset()

        # 훈련/검증 분할
        train_size = int(0.9 * len(dataset))
        train_dataset = dataset.select(range(train_size))
        eval_dataset = dataset.select(range(train_size, len(dataset)))

        logger.info(f"훈련 데이터: {len(train_dataset)}, 검증 데이터: {len(eval_dataset)}")

        # 데이터 콜레이터
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )

        # 훈련 설정
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=50,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=None,  # wandb 등 사용하려면 변경
        )

        # 트레이너 생성
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        # 훈련 실행
        logger.info("모델 훈련을 시작합니다...")
        trainer.train()

        # 모델 저장
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)

        logger.info(f"훈련 완료! 모델이 {self.config.output_dir}에 저장되었습니다.")


def main():
    parser = argparse.ArgumentParser(description="Llama-2 PII Fine-tuning")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--enron_data_path", type=str, default="enron_data.jsonl")
    parser.add_argument("--output_dir", type=str, default="./pii-llama2-ft")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)

    args = parser.parse_args()

    # 설정 생성
    config = PIIFineTuningConfig(
        model_name=args.model_name,
        enron_data_path=args.enron_data_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        train_batch_size=args.batch_size,
        max_length=args.max_length
    )

    # 출력 디렉터리 생성
    os.makedirs(config.output_dir, exist_ok=True)

    # 훈련 실행
    trainer = PIILlamaTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()