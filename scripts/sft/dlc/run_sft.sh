#!/usr/bin/env bash

SCRIPT_DIR="/mnt/nas/bihaoran/common_agent/scripts/sft/dlc"
MODEL_PATH="/mnt/nas/bihaoran/common_agent/model/Qwen3.5-4B-Base"
DATA_ROOT="/mnt/nas/bihaoran/agent_data"
TOKENIZED_DATA="/mnt/nas/bihaoran/common_agent/data/sft_coldstart/qwen35_4b_8192"
OUTPUT_DIR="/mnt/nas/bihaoran/common_agent/output/sft/qwen3.5-4b-base-coldstart"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

MAX_LENGTH=${MAX_LENGTH:-8192}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-8}
LR=${LR:-1e-5}
EPOCHS=${EPOCHS:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}
[ "$NPROC_PER_NODE" -lt 1 ] && NPROC_PER_NODE=1

if [ "${NODE_RANK:-0}" -eq 0 ]; then
  python3 "$SCRIPT_DIR/prepare_sft_data.py" \
    --data-root "$DATA_ROOT" \
    --model-path "$MODEL_PATH" \
    --output-dir "$TOKENIZED_DATA" \
    --max-length "$MAX_LENGTH" \
    --max-tools 32
else
  while [ ! -f "$TOKENIZED_DATA/dataset_info.json" ]; do sleep 10; done
fi

if [ "${NNODES:-1}" -gt 1 ]; then
  torchrun \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK:-0}" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT:-29500}" \
    "$SCRIPT_DIR/train_sft.py" \
    --model-path "$MODEL_PATH" \
    --dataset-path "$TOKENIZED_DATA" \
    --output-dir "$OUTPUT_DIR" \
    --deepspeed "$SCRIPT_DIR/ds_zero2.json" \
    --epochs "$EPOCHS" \
    --per-device-batch-size "$PER_DEVICE_BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --learning-rate "$LR" \
    --logging-steps 10 \
    --save-steps 500
else
  torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
    "$SCRIPT_DIR/train_sft.py" \
    --model-path "$MODEL_PATH" \
    --dataset-path "$TOKENIZED_DATA" \
    --output-dir "$OUTPUT_DIR" \
    --deepspeed "$SCRIPT_DIR/ds_zero2.json" \
    --epochs "$EPOCHS" \
    --per-device-batch-size "$PER_DEVICE_BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --learning-rate "$LR" \
    --logging-steps 10 \
    --save-steps 500
fi
