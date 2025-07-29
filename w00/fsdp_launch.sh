#!/bin/bash
# FSDP 기반 PII Fine-tuning 실행 스크립트

# 기본 설정
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
DATA_PATH="enron_data.jsonl"
OUTPUT_DIR="./fsdp-pii-llama2-ft"
BATCH_SIZE=2
GRAD_ACCUM=8
MAX_LENGTH=512
EPOCHS=3
LR=2e-4

echo "=== FSDP PII Fine-tuning 시작 ==="
echo "GPU 개수: $NUM_GPUS"
echo "모델: $MODEL_NAME"
echo "데이터: $DATA_PATH"
echo "출력 디렉터리: $OUTPUT_DIR"

# GPU 개수에 따른 설정 조정
if [ $NUM_GPUS -eq 1 ]; then
    echo "단일 GPU 모드로 실행합니다."
    python fsdp_pii_finetuning.py \
        --model_name $MODEL_NAME \
        --enron_data_path $DATA_PATH \
        --output_dir $OUTPUT_DIR \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --max_length $MAX_LENGTH \
        --num_epochs $EPOCHS \
        --learning_rate $LR \
        --fsdp_mixed_precision \
        --use_flash_attention

elif [ $NUM_GPUS -eq 2 ]; then
    echo "2-GPU FSDP 모드로 실행합니다."
    torchrun \
        --standalone \
        --nproc_per_node=2 \
        fsdp_pii_finetuning.py \
        --model_name $MODEL_NAME \
        --enron_data_path $DATA_PATH \
        --output_dir $OUTPUT_DIR \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --max_length $MAX_LENGTH \
        --num_epochs $EPOCHS \
        --learning_rate $LR \
        --fsdp_sharding_strategy FULL_SHARD \
        --fsdp_mixed_precision \
        --use_flash_attention

elif [ $NUM_GPUS -eq 4 ]; then
    echo "4-GPU FSDP 모드로 실행합니다."
    torchrun \
        --standalone \
        --nproc_per_node=4 \
        fsdp_pii_finetuning.py \
        --model_name $MODEL_NAME \
        --enron_data_path $DATA_PATH \
        --output_dir $OUTPUT_DIR \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --max_length $MAX_LENGTH \
        --num_epochs $EPOCHS \
        --learning_rate $LR \
        --fsdp_sharding_strategy FULL_SHARD \
        --fsdp_mixed_precision \
        --use_flash_attention

elif [ $NUM_GPUS -eq 8 ]; then
    echo "8-GPU FSDP 모드로 실행합니다."
    torchrun \
        --standalone \
        --nproc_per_node=8 \
        fsdp_pii_finetuning.py \
        --model_name $MODEL_NAME \
        --enron_data_path $DATA_PATH \
        --output_dir $OUTPUT_DIR \
        --batch_size 1 \
        --gradient_accumulation_steps 16 \
        --max_length $MAX_LENGTH \
        --num_epochs $EPOCHS \
        --learning_rate $LR \
        --fsdp_sharding_strategy FULL_SHARD \
        --fsdp_mixed_precision \
        --use_flash_attention

else
    echo "지원되지 않는 GPU 개수입니다: $NUM_GPUS"
    echo "1, 2, 4, 8 GPU를 지원합니다."
    exit 1
fi

echo "=== FSDP PII Fine-tuning 완료 ==="