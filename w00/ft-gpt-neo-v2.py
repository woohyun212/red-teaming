# ft-gpt-neo.py
# Fine-tune GPT‑Neo on Enron maildir (epoch=4) following the paper setup.
# - Streaming IterableDataset (memory efficient)
# - Token-block packing with EOS
# - Causal LM training via Transformers Trainer
#
# Usage (example):
#   python ft-gpt-neo.py --maildir ../maildir --output_dir ./neo-enron-e4
#   # with W&B logging:
#   # WANDB_API_KEY=... python ft-gpt-neo.py --maildir ../maildir --wandb --wandb_project neo-enron --wandb_run_name "gpt-neo-1.3B-e4"
#
# Optional env:
#   OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0
#
# Notes:
# - This script performs *full* fine-tuning. If you want LoRA, we can extend it.

import os                         # OS 환경 변수 설정 등에 사용
import re                         # 정규식(포워드 체인 절단 등)에 사용
import math                       # 추가된 import
import argparse                   # CLI 인자 파싱
from pathlib import Path          # 파일/디렉터리 경로 처리
from typing import Iterator, List, Optional  # 타입 힌트
from datetime import datetime

import torch                      # 텐서 연산 및 디바이스 제어
from torch.utils.data import IterableDataset  # 스트리밍 데이터셋 구현

from transformers import (        # 허깅페이스 트랜스포머 주요 컴포넌트
    AutoTokenizer,                # 사전학습 토크나이저 로드
    AutoModelForCausalLM,         # Causal LM(오토리그레시브) 모델 로드
    TrainingArguments,            # 학습 하이퍼파라미터 묶음
    Trainer,                      # 학습 루프(Trainer API)
    default_data_collator,        # 배치 구성(패딩 등) 기본 콜레이터
    set_seed,                     # 재현성(seed 고정)
)

# --- LoRA/PEFT imports ---
from peft import LoraConfig, get_peft_model, TaskType

# -------- 텍스트 유틸 (간단한 전처리) --------
# 'Forwarded message' 등 체인 구간을 잘라내기 위해 사용
FORWARD_SPLIT = re.compile(r"(?i)(^-{2,}.*forwarded.*$|^-{2,}.*original message.*$)", re.M)

def read_raw(
    path: Path,
    max_size_bytes: int = 1_000_000,
    exclude_suffixes=(".pdf", ".pst", ".doc", ".docx", ".xls", ".xlsx",
                      ".ppt", ".pptx", ".png", ".jpg", ".jpeg", ".gif")
) -> Optional[str]:
    """
    단일 파일을 문자열로 읽는다.
    - 바이너리/대용량 파일은 스킵하여 I/O 병목을 줄인다.
    - 인코딩은 utf-8 → latin-1 순으로 시도한다.
    """
    try:
        # 확장자 필터(문서/이미지/바이너리 등) → 스킵
        if path.suffix.lower() in exclude_suffixes:
            return None
        st = path.stat()
        # 0바이트 또는 너무 큰 파일은 스킵
        if st.st_size == 0 or st.st_size > max_size_bytes:
            return None
    except Exception:
        # 메타 정보 읽기 실패 시 해당 파일은 건너뜀
        return None

    # 텍스트로 읽기 시도
    for enc in ("utf-8", "latin-1"):
        try:
            return path.read_text(encoding=enc, errors="ignore")
        except Exception:
            continue

    # 최후의 수단: 바이트로 읽어서 latin-1로 디코딩
    try:
        return path.read_bytes().decode("latin-1", errors="ignore")
    except Exception:
        return None

def split_header_body(raw: str):
    """
    이메일 원문에서 헤더와 본문을 빈 줄(\n\n) 기준으로 나눈다.
    그 뒤 'Forwarded/Original Message' 체인 이후는 잘라낸다.
    """
    parts = raw.split("\n\n", 1)             # 첫 빈 줄에서 헤더/본문 분리
    header = parts[0] if parts else raw      # 헤더가 없으면 전체를 헤더로 간주
    body = parts[1] if len(parts) > 1 else ""# 본문이 없으면 빈 문자열
    body = FORWARD_SPLIT.split(body)[0]      # 포워드 체인 이후는 제거
    return header, body

# -------- 메일디렉터리 스트리밍 데이터셋 --------
class MaildirTextIterableDataset(IterableDataset):
    """
    maildir 디렉터리를 순회하며 텍스트를 스트리밍으로 생성하는 데이터셋.
    메모리에 모두 올리지 않고 한 문서(이메일)씩 처리한다.
    """
    def __init__(self, root: Path, add_header: bool = True, add_body: bool = True):
        super().__init__()
        self.root = root              # maildir 루트 경로
        self.add_header = add_header  # 헤더 포함 여부
        self.add_body = add_body      # 본문 포함 여부

    def _iter_files(self) -> Iterator[Path]:
        """재현성을 위해 정렬된 순서로 파일을 순회한다."""
        for p in sorted(self.root.rglob("*")):
            if p.is_file():
                yield p

    def __iter__(self) -> Iterator[str]:
        """
        각 파일을 읽어서 (헤더 + 본문) 텍스트를 만들어 한 항목씩 생성한다.
        """
        for p in self._iter_files():
            raw = read_raw(p)
            if not raw:
                continue
            header, body = split_header_body(raw)
            parts = []
            if self.add_header and header.strip():
                parts.append(header.strip())  # 헤더 추가
            if self.add_body and body.strip():
                parts.append(body.strip())    # 본문 추가
            if not parts:
                continue
            # 이메일 간 구분을 위해 빈 줄로 연결
            text = ("\n\n").join(parts).strip()
            if text:
                yield text

# -------- 토큰 블록 패킹 데이터셋 --------
class CausalLMTokenBlockDataset(IterableDataset):
    """
    상위 텍스트 이터레이터에서 문서를 받아 토큰화하고,
    고정 길이(block_size)의 토큰 블록으로 잘라 스트리밍한다.
    (대용량 코퍼라에서도 메모리 사용을 최소화)
    """
    def __init__(self, tokenizer, text_iterable: IterableDataset, block_size: int = 1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_iterable = text_iterable
        self.block_size = block_size
        # GPT‑Neo는 pad 토큰이 없는 경우가 많아 eos를 pad로 재사용
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __iter__(self):
        eos_id = self.tokenizer.eos_token_id      # EOS 토큰 ID
        buffer: List[int] = []                    # 누적 토큰 버퍼
        for text in self.text_iterable:
            # 문서 단위로 토큰화(스페셜 토큰은 수동 관리)
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            # 문서 경계에 EOS 추가하여 학습 안정화
            if eos_id is not None:
                ids.append(eos_id)
            # 버퍼에 누적
            buffer.extend(ids)
            # 버퍼가 block_size 이상이면 고정 길이 블록을 생성
            while len(buffer) >= self.block_size:
                chunk = buffer[:self.block_size]          # 앞에서 block_size만큼 슬라이스
                buffer = buffer[self.block_size:]         # 사용한 만큼 버퍼에서 제거
                input_ids = torch.tensor(chunk, dtype=torch.long)
                attention_mask = torch.ones_like(input_ids)
                # 라벨은 입력과 동일(다음 토큰 예측)
                yield {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": input_ids.clone()
                }
        # 남은 토큰(< block_size)은 기본적으로 버림(원하면 마지막 블록도 내보내도록 변경 가능)

# -------- IterableDataset length/steps estimation (for Trainer) --------
def _iter_all_files_sorted(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file():
            yield p

def _sample_total_chars(root: Path, sample_files: int = 2000) -> tuple[int, int]:
    """
    빠른 샘플링으로 텍스트 총 길이를 추정하기 위한 유틸.
    반환: (샘플링에 사용된 파일 수, 샘플 텍스트 총 문자 수)
    """
    total_chars = 0
    used = 0
    for i, p in enumerate(_iter_all_files_sorted(root)):
        if i >= sample_files:
            break
        raw = read_raw(p)
        if not raw:
            continue
        header, body = split_header_body(raw)
        text = ("\n\n".join([s for s in (header.strip(), body.strip()) if s])).strip()
        total_chars += len(text)
        used += 1
    return used, total_chars

def estimate_max_steps_for_streaming(
    root: Path,
    tokenizer,
    block_size: int,
    per_device_train_batch_size: int,
    grad_accum: int,
    epochs: int,
    sample_files: int = 2000,
    chars_per_token: float = 4.0,
) -> int:
    """
    IterableDataset은 __len__이 없어 Trainer가 학습 스텝 수를 알 수 없다.
    샘플 파일의 총 문자 수를 이용해 토큰 수를 근사하고 → 블록 수 → 스텝 수를 추정한다.
    - chars_per_token: 영어 기준 평균 1토큰 ≈ 4 문자(경험적) 기본값.
    반환값: max_steps(optimizer step 기준, gradient_accum을 고려)
    """
    # 전체 파일 수
    total_files = sum(1 for _ in _iter_all_files_sorted(root))
    if total_files == 0:
        return 0
    used, sample_chars = _sample_total_chars(root, sample_files=sample_files)
    if used == 0:
        return 0
    # 파일당 평균 문자 수 × 전체 파일 수 → 전체 문자 수 근사
    avg_chars_per_file = sample_chars / used
    est_total_chars = avg_chars_per_file * total_files
    # 문자→토큰 근사
    est_total_tokens = est_total_chars / max(chars_per_token, 1e-6)
    # 문서 경계 EOS도 있으나 근사치에 포함(무시 가능)
    est_blocks = math.floor(est_total_tokens / max(block_size, 1))
    # 스텝 계산: 한 스텝(optimizer step)은 grad_accum 배치 누적 이후
    steps_per_epoch = math.floor(est_blocks / max(per_device_train_batch_size, 1))
    optimizer_steps_per_epoch = math.floor(steps_per_epoch / max(grad_accum, 1))
    est_max_steps = int(max(optimizer_steps_per_epoch * max(epochs, 1), 1))
    return est_max_steps

# -------- 학습 루틴 --------
def main():
    # CLI 인자 정의
    parser = argparse.ArgumentParser(description="Fine-tune GPT‑Neo on Enron maildir (4 epochs).")
    # maildir 경로는 필수 인자
    parser.add_argument("--maildir", type=str, required=True, help="Path to Enron maildir root")
    # 사용할 사전학습 모델 이름(125M 기본)
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-neo-125M", help="Pretrained GPT‑Neo model")
    # 결과(체크포인트/최종가중치) 저장 경로
    parser.add_argument("--output_dir", type=str, default="./neo-enron-e4", help="Where to save the model")
    # 에폭 수(논문 셋업에 맞춰 기본 4)
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs (default: 4)")
    # 한 블록에 들어갈 토큰 수(컨텍스트 길이)
    parser.add_argument("--block_size", type=int, default=1024, help="Token block length")
    # 디바이스당 배치 사이즈(메모리/VRAM에 맞게 조절)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    # 그래디언트 누적 스텝(유효 배치 크기 확장)
    parser.add_argument("--grad_accum", type=int, default=16)
    # 학습률
    parser.add_argument("--lr", type=float, default=2e-5)
    # 가중치 감쇠(정규화)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    # 웜업 비율
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    # 시드 고정(재현성)
    parser.add_argument("--seed", type=int, default=42)
    # 저장/로깅 빈도
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--logging_first_step", action="store_true", help="Log at step 0 for early visibility")
    # 디버깅용 스텝 제한(>0이면 스텝 수를 강제로 제한)
    parser.add_argument("--max_steps", type=int, default=-1, help="Set >0 to cap steps (debug)")
    # 혼합정밀 옵션
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 if available")
    parser.add_argument("--fp16", action="store_true", help="Use float16")
    # 그래디언트 체크포인팅(메모리 절감)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    # 체크포인트에서 학습 재개
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    # IterableDataset용: 샘플링 기반으로 max_steps 자동 추정
    parser.add_argument("--auto_steps", action="store_true", help="IterableDataset용: 샘플링 기반으로 max_steps 자동 추정")
    parser.add_argument("--estimate_files", type=int, default=2000, help="max_steps 추정 시 샘플링할 파일 수")
    parser.add_argument("--chars_per_token", type=float, default=4.0, help="문자→토큰 환산 비율(영문≈4)")
    # --- Weights & Biases logging options ---
    parser.add_argument("--wandb", dest="wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--no-wandb", dest="wandb", action="store_false", help="Disable Weights & Biases logging")
    parser.set_defaults(wandb=False)
    parser.add_argument("--wandb_project", type=str, default=os.getenv("WANDB_PROJECT", "neo-enron"), help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=os.getenv("WANDB_ENTITY", None), help="W&B entity (team/org)")
    parser.add_argument("--wandb_run_name", type=str, default=os.getenv("WANDB_NAME", None), help="W&B run display name")
    parser.add_argument("--wandb_group", type=str, default=os.getenv("WANDB_RUN_GROUP", None), help="W&B group name")
    parser.add_argument("--wandb_tags", type=str, default=os.getenv("WANDB_TAGS", ""), help="Comma-separated W&B tags")
    parser.add_argument("--wandb_mode", type=str, default=os.getenv("WANDB_MODE", "online"), choices=["online","offline","disabled"], help="W&B mode")
    # --- LoRA (PEFT) options ---
    parser.add_argument("--lora", action="store_true", help="Enable LoRA fine-tuning (PEFT)")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (scaling)")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora_target", type=str, default="q_proj,k_proj,v_proj,out_proj", help="Comma-separated module names to apply LoRA to")
    parser.add_argument("--lora_bias", type=str, default="none", choices=["none","all","lora_only"], help="Bias handling in LoRA layers")
    args = parser.parse_args()  # 인자 파싱

    set_seed(args.seed)  # 재현성 보장
    # Console reminder for logging delay due to gradient accumulation
    if hasattr(args, "grad_accum") and args.grad_accum and args.grad_accum > 1:
        print(f"[Info] Using gradient_accumulation_steps={args.grad_accum}. "
              f"Trainer logs every {args.logging_steps} optimizer steps "
              f"(= {args.logging_steps * args.grad_accum} micro-batches). "
              f"Use --logging_steps to change frequency or --logging_first_step for an initial log.")

    # --- W&B helpers ---
    def _parse_tags(s: str):
        return [t.strip() for t in s.split(",") if t.strip()] if s else []

    def _maybe_init_wandb():
        """
        Initialize W&B if enabled. Gracefully degrade if wandb is unavailable.
        """
        if not args.wandb or args.wandb_mode == "disabled":
            os.environ["WANDB_DISABLED"] = "true"
            return None
        # honor explicit mode
        os.environ["WANDB_MODE"] = args.wandb_mode
        run_name = args.wandb_run_name or f"neo-{Path(args.model_name).name}-{datetime.now().strftime('%m%d-%H%M')}"
        try:
            import wandb
            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                group=args.wandb_group,
                tags=_parse_tags(args.wandb_tags),
                config={k: v for k, v in vars(args).items() if k not in {"resume_from_checkpoint"}},
            )
            # Avoid heavy gradient/parameter watching by default
            os.environ.setdefault("WANDB_WATCH", "false")
            # Log a one-time startup ping for immediate chart visibility
            try:
                import wandb
                wandb.log({"_startup_ping": 1}, step=0)
            except Exception:
                pass
            return run
        except Exception as e:
            print(f"[Warn] W&B init failed: {e}. Proceeding without W&B.")
            os.environ["WANDB_DISABLED"] = "true"
            return None

    # 사용 디바이스 로그(디버깅 편의)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Device: {device}")
    # Initialize W&B if requested (before Trainer so HF will also pick it up)
    wandb_run = _maybe_init_wandb()

    print("[Info] Loading tokenizer & model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)  # 토크나이저 로드
    # 긴 시퀀스 경고 억제(우리는 아래에서 직접 블록 패킹함)
    tokenizer.model_max_length = 1_000_000_000  # effectively "no limit" for encode()
    # eos/pad 토큰이 없으면 안전하게 설정
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "</s>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)  # 사전학습 모델 로드
    # gradient checkpointing 시, 캐시 비활성화로 메모리 절약
    if args.gradient_checkpointing:
        model.config.use_cache = False
    model.resize_token_embeddings(len(tokenizer))  # 토크나이저 토큰 수 변경 시 임베딩 리사이즈
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()      # 메모리 절약(속도는 약간 손해)

    # --- Apply LoRA (PEFT) if requested ---
    if args.lora:
        target_modules = [m.strip() for m in args.lora_target.split(",") if m.strip()]
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias=args.lora_bias,
            task_type=TaskType.CAUSAL_LM,
        )
        # gradient checkpointing와 함께 사용할 때 입력 텐서에 grad 필요 플래그 설정
        try:
            model.enable_input_require_grads()
        except Exception:
            pass
        model = get_peft_model(model, lora_cfg)
        # 간단한 학습 가능 파라미터 비율 로그
        try:
            trainable, total = 0, 0
            for _, p in model.named_parameters():
                total += p.numel()
                if p.requires_grad:
                    trainable += p.numel()
            ratio = (trainable / total) if total else 0.0
            print(f"[LoRA] Trainable params: {trainable}/{total} ({ratio:.2%}) | targets={target_modules} r={args.lora_r} alpha={args.lora_alpha} drop={args.lora_dropout}")
        except Exception:
            pass

    # 데이터셋 구성: maildir → 텍스트 스트림 → 토큰 블록 스트림
    maildir_root = Path(args.maildir)
    text_stream = MaildirTextIterableDataset(maildir_root, add_header=True, add_body=True)
    train_stream = CausalLMTokenBlockDataset(tokenizer, text_stream, block_size=args.block_size)

    # IterableDataset은 __len__이 없으므로 Trainer가 학습 스케줄 길이를 알아야 한다.
    # 1) 사용자가 --max_steps를 지정하면 그대로 사용
    # 2) --auto_steps면 샘플 기반 추정치 사용
    # 3) 둘 다 아니면 오류 대신 보수적으로 추정하여 진행(경고 출력)
    if args.max_steps is not None and args.max_steps > 0:
        resolved_max_steps = int(args.max_steps)
        why = "user-specified"
    else:
        if args.auto_steps:
            resolved_max_steps = estimate_max_steps_for_streaming(
                maildir_root,
                tokenizer,
                block_size=args.block_size,
                per_device_train_batch_size=args.per_device_train_batch_size,
                grad_accum=args.grad_accum,
                epochs=args.epochs,
                sample_files=args.estimate_files,
                chars_per_token=args.chars_per_token,
            )
            why = f"auto-estimated (sample_files={args.estimate_files}, chars_per_token={args.chars_per_token})"
        else:
            # 기본적으로 자동 추정 수행(안전한 기본값)
            resolved_max_steps = estimate_max_steps_for_streaming(
                maildir_root,
                tokenizer,
                block_size=args.block_size,
                per_device_train_batch_size=args.per_device_train_batch_size,
                grad_accum=args.grad_accum,
                epochs=args.epochs,
                sample_files=args.estimate_files,
                chars_per_token=args.chars_per_token,
            )
            why = f"auto-estimated(default)"
        if resolved_max_steps <= 0:
            raise ValueError(
                "IterableDataset 사용 시 max_steps가 필요합니다. "
                "옵션 --max_steps N 또는 --auto_steps를 지정하세요."
            )
    print(f"[Info] max_steps resolved to {resolved_max_steps} ({why})")

    # 학습 하이퍼파라미터 묶음(Trainer가 사용)
    training_args = TrainingArguments(
        output_dir=args.output_dir,                         # 체크포인트/최종 모델 저장 디렉터리
        overwrite_output_dir=True,                         # 기존 디렉터리 덮어쓰기
        num_train_epochs=args.epochs,                      # 에폭 수
        per_device_train_batch_size=args.per_device_train_batch_size,  # 배치
        gradient_accumulation_steps=args.grad_accum,       # 그래디언트 누적
        learning_rate=args.lr,                             # 학습률
        weight_decay=args.weight_decay,                    # 가중치 감쇠
        warmup_ratio=args.warmup_ratio,                    # 웜업 비율
        logging_steps=args.logging_steps,                  # 로깅 빈도
        logging_first_step=args.logging_first_step,         # Log at step 0 for early visibility
        logging_strategy="steps",                          # Explicitly use step-based logging
        disable_tqdm=False,                                # Ensure progress bar is visible
        save_steps=args.save_steps,                        # 저장 빈도
        save_total_limit=2,                                # 체크포인트 보존 개수 제한
        dataloader_num_workers=2,                          # 토크나이즈 워커 수(과도하게 크면 GIL 영향)
        bf16=args.bf16,                                    # bf16 사용 여부
        fp16=args.fp16 and not args.bf16,                  # fp16 사용 여부(bf16과 동시사용 X)
        gradient_checkpointing=args.gradient_checkpointing,# 그래디언트 체크포인팅
        report_to=(["wandb"] if args.wandb and args.wandb_mode != "disabled" else []),
        run_name=(args.wandb_run_name or None),
        max_steps=resolved_max_steps,  # 스텝 제한
    )

    # Trainer 생성(학습 루프 구성요소)
    trainer = Trainer(
        model=model,                               # 학습 대상 모델
        args=training_args,                        # 학습 설정
        train_dataset=train_stream,                # 학습 데이터(스트리밍)
        data_collator=default_data_collator,       # 배치 패딩/정렬
    )

    print("[Info] Starting training...")
    # 체크포인트에서 재개 옵션 적용하여 학습 시작
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    print("[Info] Saving model & tokenizer...")
    # 최종 모델/토크나이저 저장
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[Done] Fine-tuning complete.")
    # Finish W&B run if active
    try:
        if wandb_run is not None:
            wandb_run.finish()
    except Exception:
        pass

if __name__ == "__main__":
    # 토크나이즈/CPU 워커 다중화 시 BLAS 과다 스레딩 방지
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()  # 엔트리포인트 실행


"""
python3 w00/ft-gpt-neo-v2.py \
--maildir /home/w00/data1/maildir \
--model_name EleutherAI/gpt-neo-1.3B \
--max_steps 5000 \
--estimate_files 3000 \
--chars_per_token 4.0 \
--epochs 4 \
--block_size 1024 \
--per_device_train_batch_size 1 \
--grad_accum 32 \
--fp16 \
--lora \
--wandb \
--wandb_project neo-enron \
--wandb_run_name "gpt-neo-1.3B-e4-5k" \
--wandb_tags "enron,neo,1.3B,fp16,ckpt"
"""