#!/usr/bin/env python3
"""
FSDP를 활용한 Llama-2 PII Fine-tuning Script
다중 GPU에서 효율적인 메모리 사용과 훈련 속도 향상을 위해 FSDP를 적용합니다.
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
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import argparse
from functools import partial
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FSDPPIIFineTuningConfig:
    """FSDP 기반 PII 파인튜닝 설정"""
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    enron_data_path: str = "enron_data.jsonl"
    output_dir: str = "./fsdp-pii-llama2-ft"
    max_length: int = 512
    train_batch_size: int = 2  # FSDP에서는 더 작은 배치 사이즈 사용
    eval_batch_size: int = 2
    num_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    gradient_accumulation_steps: int = 8  # 더 큰 accumulation으로 effective batch size 증가

    # LoRA 설정
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # PII 관련 설정
    pii_types: List[str] = field(default_factory=lambda: ["email", "phone", "name", "loc"])
    include_headers: bool = True
    max_pii_examples: int = 5

    # FSDP 관련 설정
    fsdp_sharding_strategy: str = "FULL_SHARD"  # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD, HYBRID_SHARD
    fsdp_auto_wrap_policy: str = "transformer_based"  # transformer_based, size_based, no_wrap
    fsdp_backward_prefetch: str = "BACKWARD_PRE"  # BACKWARD_PRE, BACKWARD_POST, NO_PREFETCH
    fsdp_cpu_offload: bool = False
    fsdp_mixed_precision: bool = True
    fsdp_activation_checkpointing: bool = True
    fsdp_transformer_layer_cls_to_wrap: str = "LlamaDecoderLayer"
    fsdp_min_num_params: int = 1e6

    # 기타 최적화 설정
    use_flash_attention: bool = False  # Flash Attention 2 사용 여부
    use_gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True


class FSDPEnronPIIDataProcessor:
    """FSDP용 Enron 이메일 데이터 프로세서"""

    def __init__(self, config: FSDPPIIFineTuningConfig):
        self.config = config
        # 분산 환경에서는 rank 0에서만 토크나이저 로드
        if not dist.is_initialized() or dist.get_rank() == 0:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = None

    def load_enron_data(self) -> List[Dict]:
        """Enron JSONL 데이터 로드"""
        data = []
        # 분산 환경에서는 rank 0에서만 데이터 로드
        if not dist.is_initialized() or dist.get_rank() == 0:
            with open(self.config.enron_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data.append(json.loads(line.strip()))
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON 파싱 오류: {e}")
                        continue
            logger.info(f"총 {len(data)}개의 이메일 데이터를 로드했습니다.")

        # 다른 rank들과 데이터 동기화
        if dist.is_initialized():
            # 브로드캐스트를 위해 모든 rank에서 같은 크기의 객체 필요
            if dist.get_rank() == 0:
                data_size = len(data)
            else:
                data_size = 0

            # 데이터 크기 브로드캐스트
            data_size_tensor = torch.tensor(data_size, device='cuda')
            dist.broadcast(data_size_tensor, src=0)

            # rank 0이 아닌 경우 빈 리스트 생성
            if dist.get_rank() != 0:
                data = [None] * data_size_tensor.item()

            # 실제 데이터 브로드캐스트 (간단화를 위해 여기서는 생략)
            # 실제로는 pickle을 사용하여 브로드캐스트하거나 각 rank에서 독립적으로 로드

        return data

    def extract_pii_examples(self, email_data: Dict) -> List[str]:
        """이메일 데이터에서 PII 예제들을 추출"""
        pii_examples = []
        pii_info = email_data.get('pii', {})
        headers = email_data.get('headers', {})

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

        # PII 정보 요약 생성
        pii_summary = "\n".join(pii_examples[:5])

        # 시스템 프롬프트 (Llama-2 Chat 형식)
        system_prompt = "You are a helpful assistant that can extract and organize personal information from emails."
        user_prompt = f"Extract and summarize the personal information from this email:\n\n{text[:300]}..."
        assistant_response = f"Here is the personal information found in the email:\n\n{pii_summary}"

        # Llama-2 Chat 형식으로 변환
        conversation = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST] {assistant_response}</s>"

        examples.append({
            'text': conversation,
            'labels': conversation
        })

        return examples

    def create_dataset(self) -> Dataset:
        """학습용 데이터셋 생성"""
        enron_data = self.load_enron_data()
        all_examples = []

        for email_data in enron_data:
            if email_data is not None:  # None 체크 추가
                examples = self.create_training_examples(email_data)
                all_examples.extend(examples)

        logger.info(f"총 {len(all_examples)}개의 학습 예제를 생성했습니다.")

        # 토큰화
        def tokenize_function(examples):
            if self.tokenizer is None:
                # 분산 환경에서 tokenizer가 없는 경우 다시 로드
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

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
            remove_columns=dataset.column_names,
            num_proc=self.config.dataloader_num_workers if not dist.is_initialized() else 1
        )

        return tokenized_dataset


class FSDPPIILlamaTrainer:
    """FSDP를 활용한 PII Llama-2 모델 파인튜닝"""

    def __init__(self, config: FSDPPIIFineTuningConfig):
        self.config = config
        self.setup_distributed()
        self.setup_model_and_tokenizer()

    def setup_distributed(self):
        """분산 훈련 환경 설정"""
        if 'LOCAL_RANK' in os.environ:
            local_rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(local_rank)

            if not dist.is_initialized():
                dist.init_process_group(backend='nccl')

            self.local_rank = local_rank
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()

            logger.info(f"분산 훈련 초기화: rank={self.rank}, world_size={self.world_size}, local_rank={self.local_rank}")
        else:
            self.local_rank = 0
            self.world_size = 1
            self.rank = 0
            logger.info("단일 GPU 모드")

    def get_fsdp_config(self):
        """FSDP 설정 생성"""
        # Sharding Strategy
        sharding_strategy_map = {
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
            "NO_SHARD": ShardingStrategy.NO_SHARD,
            "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
        }

        # Backward Prefetch
        backward_prefetch_map = {
            "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
            "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
            "NO_PREFETCH": None,
        }

        # Auto Wrap Policy
        if self.config.fsdp_auto_wrap_policy == "transformer_based":
            auto_wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={LlamaDecoderLayer}
            )
        elif self.config.fsdp_auto_wrap_policy == "size_based":
            auto_wrap_policy = partial(
                size_based_auto_wrap_policy,
                min_num_params=self.config.fsdp_min_num_params
            )
        else:
            auto_wrap_policy = None

        # Mixed Precision
        mixed_precision = None
        if self.config.fsdp_mixed_precision:
            mixed_precision = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )

        # CPU Offload
        cpu_offload = CPUOffload(offload_params=True) if self.config.fsdp_cpu_offload else None

        return {
            "sharding_strategy": sharding_strategy_map[self.config.fsdp_sharding_strategy],
            "auto_wrap_policy": auto_wrap_policy,
            "mixed_precision": mixed_precision,
            "backward_prefetch": backward_prefetch_map[self.config.fsdp_backward_prefetch],
            "cpu_offload": cpu_offload,
            "device_id": self.local_rank,
        }

    def setup_model_and_tokenizer(self):
        """모델과 토크나이저 설정"""
        logger.info(f"모델 로딩: {self.config.model_name}")

        # 토크나이저 설정
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 모델 로드
        model_kwargs = {
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
        }

        # Flash Attention 2 설정
        if self.config.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        # FSDP 환경에서는 device_map을 사용하지 않음
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )

        # Gradient Checkpointing 활성화
        if self.config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # LoRA 설정
        if self.config.use_lora:
            # FSDP와 LoRA 호환성을 위한 모델 준비
            self.model = prepare_model_for_kbit_training(self.model)

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
            self.model = get_peft_model(self.model, lora_config)

            if self.rank == 0:
                self.model.print_trainable_parameters()

    def train(self):
        """FSDP를 사용한 모델 파인튜닝 실행"""
        # 데이터 준비
        data_processor = FSDPEnronPIIDataProcessor(self.config)
        dataset = data_processor.create_dataset()

        # 훈련/검증 분할
        train_size = int(0.9 * len(dataset))
        train_dataset = dataset.select(range(train_size))
        eval_dataset = dataset.select(range(train_size, len(dataset)))

        if self.rank == 0:
            logger.info(f"훈련 데이터: {len(train_dataset)}, 검증 데이터: {len(eval_dataset)}")

        # 데이터 콜레이터
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )

        # FSDP 관련 훈련 설정
        fsdp_config = self.get_fsdp_config()

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

            # FSDP 설정
            fsdp=["full_shard", "auto_wrap"] if self.world_size > 1 else [],
            fsdp_config=fsdp_config if self.world_size > 1 else None,

            # 성능 최적화 설정
            bf16=True if torch.cuda.is_bf16_supported() else False,
            fp16=True if not torch.cuda.is_bf16_supported() else False,
            dataloader_pin_memory=self.config.dataloader_pin_memory,
            dataloader_num_workers=self.config.dataloader_num_workers,
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,

            # 로깅 설정
            report_to=None,
            logging_first_step=True,
            logging_dir=f"{self.config.output_dir}/logs",

            # 메모리 최적화
            gradient_checkpointing=self.config.use_gradient_checkpointing,
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
        if self.rank == 0:
            logger.info("FSDP 모델 훈련을 시작합니다...")

        trainer.train()

        # 모델 저장 (rank 0에서만)
        if self.rank == 0:
            trainer.save_model()
            self.tokenizer.save_pretrained(self.config.output_dir)
            logger.info(f"훈련 완료! 모델이 {self.config.output_dir}에 저장되었습니다.")


def main():
    parser = argparse.ArgumentParser(description="FSDP Llama-2 PII Fine-tuning")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--enron_data_path", type=str, default="enron_data.jsonl")
    parser.add_argument("--output_dir", type=str, default="./fsdp-pii-llama2-ft")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)

    # FSDP 관련 인자
    parser.add_argument("--fsdp_sharding_strategy", type=str, default="FULL_SHARD",
                        choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"])
    parser.add_argument("--fsdp_cpu_offload", action="store_true", help="CPU offload 사용")
    parser.add_argument("--fsdp_mixed_precision", action="store_true", default=True, help="Mixed precision 사용")
    parser.add_argument("--use_flash_attention", action="store_true", help="Flash Attention 2 사용")

    args = parser.parse_args()

    # 설정 생성
    config = FSDPPIIFineTuningConfig(
        model_name=args.model_name,
        enron_data_path=args.enron_data_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        train_batch_size=args.batch_size,
        max_length=args.max_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fsdp_sharding_strategy=args.fsdp_sharding_strategy,
        fsdp_cpu_offload=args.fsdp_cpu_offload,
        fsdp_mixed_precision=args.fsdp_mixed_precision,
        use_flash_attention=args.use_flash_attention,
    )

    # 출력 디렉터리 생성
    os.makedirs(config.output_dir, exist_ok=True)

    # 훈련 실행
    trainer = FSDPPIILlamaTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()