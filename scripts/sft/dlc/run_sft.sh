#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/mnt/nas/bihaoran/common_agent/scripts/sft/dlc"
PY="/mnt/nas/bihaoran/common_agent/envs/qwen35_swift/bin/python"
MODEL_PATH="/mnt/nas/bihaoran/common_agent/model/Qwen3.5-4B-Base"
DATA_ROOT="/mnt/nas/bihaoran/agent_data"
TOKENIZED_DATA="/mnt/nas/bihaoran/common_agent/data/sft_coldstart/qwen35_4b_8192"
OUTPUT_DIR="/mnt/nas/bihaoran/common_agent/output/sft/qwen3.5-4b-base-coldstart"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

# Keep all experiment logs on NAS. W&B never needs network access during training.
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_PROJECT=${WANDB_PROJECT:-common-agent-sft}
export WANDB_DIR=${WANDB_DIR:-$OUTPUT_DIR/wandb}
export WANDB_NAME=${WANDB_NAME:-qwen3.5-4b-base-coldstart-${MASTER_ADDR:-dlc}}
export REPORT_TO=${REPORT_TO:-tensorboard,wandb}
mkdir -p "$OUTPUT_DIR" "$WANDB_DIR"

# PAI DLC injects these variables for distributed PyTorch jobs.
: "${WORLD_SIZE:?WORLD_SIZE is required; configure the DLC job with 2 nodes}"
: "${RANK:?RANK is required; it must be injected by DLC}"
: "${MASTER_ADDR:?MASTER_ADDR is required; it must be injected by DLC}"
: "${MASTER_PORT:?MASTER_PORT is required; it must be injected by DLC}"

NPROC_PER_NODE=${NPROC_PER_NODE:-$("$PY" -c 'import torch; print(torch.cuda.device_count())')}

if [ "$WORLD_SIZE" -ne 2 ]; then
  echo "[ERROR] Expected WORLD_SIZE=2, got WORLD_SIZE=$WORLD_SIZE" >&2
  exit 2
fi
if [ "$NPROC_PER_NODE" -ne 8 ]; then
  echo "[ERROR] Expected NPROC_PER_NODE=8, got NPROC_PER_NODE=$NPROC_PER_NODE" >&2
  exit 2
fi

"$PY" -c 'import torch, transformers, peft, datasets, wandb; print(f"env ok: torch={torch.__version__} transformers={transformers.__version__} peft={peft.__version__} datasets={datasets.__version__} wandb={wandb.__version__} devices={torch.cuda.device_count()}")'

echo "[DLC] rank=$RANK/$WORLD_SIZE master=$MASTER_ADDR:$MASTER_PORT nproc_per_node=$NPROC_PER_NODE"
echo "[TRAIN] max_length=${MAX_LENGTH:-8192} per_device_batch=${PER_DEVICE_BATCH_SIZE:-1} grad_accum=${GRAD_ACCUM:-8} lr=${LR:-1e-5} epochs=${EPOCHS:-1}"
echo "[WANDB] mode=$WANDB_MODE dir=$WANDB_DIR project=$WANDB_PROJECT name=$WANDB_NAME"

MAX_LENGTH=${MAX_LENGTH:-8192}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-8}
LR=${LR:-1e-5}
EPOCHS=${EPOCHS:-1}

# Only node rank 0 preprocesses. Other nodes wait for the shared NAS dataset.
if [ "$RANK" -eq 0 ]; then
  "$PY" "$SCRIPT_DIR/prepare_sft_data.py" \
    --data-root "$DATA_ROOT" \
    --model-path "$MODEL_PATH" \
    --output-dir "$TOKENIZED_DATA" \
    --max-length "$MAX_LENGTH" \
    --max-tools 32
else
  echo "[DATA] waiting for $TOKENIZED_DATA/dataset_info.json"
  while [ ! -f "$TOKENIZED_DATA/dataset_info.json" ]; do sleep 10; done
fi

exec "$PY" -m torch.distributed.run \
  --nnodes="$WORLD_SIZE" \
  --node_rank="$RANK" \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  "$SCRIPT_DIR/train_sft.py" \
  --model-path "$MODEL_PATH" \
  --dataset-path "$TOKENIZED_DATA" \
  --output-dir "$OUTPUT_DIR" \
  --deepspeed "$SCRIPT_DIR/ds_zero2.json" \
  --epochs "$EPOCHS" \
  --per-device-batch-size "$PER_DEVICE_BATCH_SIZE" \
  --grad-accum "$GRAD_ACCUM" \
  --learning-rate "$LR" \
  --optim adamw_torch \
  --logging-steps 10 \
  --save-steps 500 \
  --save-total-limit 2 \
  --num-workers 4
