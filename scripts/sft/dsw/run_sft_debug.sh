#!/usr/bin/env bash

SCRIPT_DIR="/mnt/nas/bihaoran/common_agent/scripts/sft/dsw"
PY="/mnt/nas/bihaoran/common_agent/envs/qwen35_swift/bin/python"
MODEL_PATH="/mnt/nas/bihaoran/common_agent/model/Qwen3.5-4B-Base"
DATA_ROOT="/mnt/nas/bihaoran/agent_data"
TOKENIZED_DATA="/mnt/nas/bihaoran/common_agent/data/sft_coldstart/debug_qwen35_4b_2048"
OUTPUT_DIR="/mnt/nas/bihaoran/common_agent/output/sft/debug_qwen3.5-4b-base-coldstart-lora"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

"$PY" "$SCRIPT_DIR/prepare_sft_data.py" \
  --data-root "$DATA_ROOT" \
  --model-path "$MODEL_PATH" \
  --output-dir "$TOKENIZED_DATA" \
  --max-length 2048 \
  --max-tools 16 \
  --max-records-per-source 8 \
  --overwrite && \
"$PY" "$SCRIPT_DIR/train_sft.py" \
  --model-path "$MODEL_PATH" \
  --dataset-path "$TOKENIZED_DATA" \
  --output-dir "$OUTPUT_DIR" \
  --max-steps 5 \
  --per-device-batch-size 1 \
  --grad-accum 1 \
  --learning-rate 1e-4 \
  --logging-steps 1 \
  --save-steps 5 \
  --save-total-limit 1 \
  --num-workers 2 \
  --optim adamw_torch \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --lora-target-modules q_proj,v_proj
