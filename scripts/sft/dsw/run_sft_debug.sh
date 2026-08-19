#!/usr/bin/env bash

SCRIPT_DIR="/mnt/nas/bihaoran/common_agent/scripts/sft/dsw"
MODEL_PATH="/mnt/nas/bihaoran/common_agent/model/Qwen3.5-4B-Base"
DATA_ROOT="/mnt/nas/bihaoran/agent_data"
TOKENIZED_DATA="/mnt/nas/bihaoran/common_agent/data/sft_coldstart/debug_qwen35_4b_2048"
OUTPUT_DIR="/mnt/nas/bihaoran/common_agent/output/sft/debug_qwen3.5-4b-base-coldstart"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}
[ "$NPROC_PER_NODE" -lt 1 ] && NPROC_PER_NODE=1

python3 "$SCRIPT_DIR/prepare_sft_data.py" \
  --data-root "$DATA_ROOT" \
  --model-path "$MODEL_PATH" \
  --output-dir "$TOKENIZED_DATA" \
  --max-length 2048 \
  --max-tools 16 \
  --max-records-per-source 8 \
  --overwrite

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
  "$SCRIPT_DIR/train_sft.py" \
  --model-path "$MODEL_PATH" \
  --dataset-path "$TOKENIZED_DATA" \
  --output-dir "$OUTPUT_DIR" \
  --deepspeed "$SCRIPT_DIR/ds_zero2.json" \
  --max-steps 5 \
  --per-device-batch-size 1 \
  --grad-accum 1 \
  --learning-rate 1e-5 \
  --logging-steps 1 \
  --save-steps 5 \
  --save-total-limit 1 \
  --num-workers 2
