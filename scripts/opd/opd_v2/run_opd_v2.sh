#!/usr/bin/env bash

# COMMON-AGENT / ms-swift OPD stage 2
# DLC topology:
#   node rank 0:
#       GPU 0-3  Qwen3.5-122B-A10B teacher, swift deploy, TP=4
#       GPU 4-7  Qwen3.5-4B rollout/eval server, TP=4
#   node rank 1:
#       GPU 0-7  Qwen3.5-4B full-parameter GKD training
#
# The stage is split into 500-optimizer-step chunks.
# Every chunk ends with an exact checkpoint-N ToolSandbox evaluation.

SCRIPT_DIR="/mnt/nas/bihaoran/common_agent/scripts/opd/opd_v2"
COMMON_DIR="/mnt/nas/bihaoran/common_agent/scripts/opd/common"
ENV_DIR="/mnt/nas/bihaoran/common_agent/envs/qwen35_swift"
PY="$ENV_DIR/bin/python"
SWIFT="$ENV_DIR/bin/swift"

TEACHER_MODEL="/mnt/nas/bihaoran/common_agent/model/teacher/Qwen3.5-122B-A10B"
if [ -n "${STUDENT_MODEL:-}" ]; then
  START_MODEL="$STUDENT_MODEL"
elif [ -s "/mnt/nas/bihaoran/common_agent/output/opd/opd_v1/FINAL_CHECKPOINT" ]; then
  START_MODEL="$(cat /mnt/nas/bihaoran/common_agent/output/opd/opd_v1/FINAL_CHECKPOINT)"
else
  START_MODEL="/mnt/nas/bihaoran/common_agent/output/opd/opd_v1/checkpoint-2000"
fi

V1_DATA="/mnt/nas/bihaoran/agent_data/opd_v1"
V2_DATA="/mnt/nas/bihaoran/agent_data/opd_v2"
BENCHMARK="/mnt/nas/bihaoran/common_agent/benchmark/ToolSandbox"

OUTPUT_DIR="${OUTPUT_DIR:-/mnt/nas/bihaoran/common_agent/output/opd/opd_v2}"
WORK_DIR="$OUTPUT_DIR/runtime"
CONTROL_DIR="$OUTPUT_DIR/control"
LOG_DIR="$OUTPUT_DIR/logs"
TRAIN_JSONL="$WORK_DIR/train.jsonl"
PLUGIN="$COMMON_DIR/common_agent_opd_plugin.py"
PREP="$COMMON_DIR/prepare_opd_data.py"
EVAL_SCRIPT="$COMMON_DIR/toolsandbox_eval.py"

TEACHER_PORT="${TEACHER_PORT:-18000}"
ROLLOUT_PORT="${ROLLOUT_PORT:-18001}"
EVAL_PORT="${EVAL_PORT:-18002}"
TOPK="${TOPK:-64}"

TOTAL_STEPS="${TOTAL_STEPS:-3000}"
CHUNK_STEPS="${CHUNK_STEPS:-500}"
MAX_TURNS="${MAX_TURNS:-32}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-24576}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-512}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-32768}"

PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-2e-6}"
BETA="${BETA:-0.5}"
EVAL_PARALLEL="${EVAL_PARALLEL:-16}"
EVAL_TEST_MODE="${EVAL_TEST_MODE:-0}"

mkdir -p "$SCRIPT_DIR" "$WORK_DIR" "$CONTROL_DIR" "$LOG_DIR" "$OUTPUT_DIR/eval"

for required_file in "$PLUGIN" "$PREP" "$EVAL_SCRIPT"; do
  if [ ! -s "$required_file" ]; then
    echo "[ERROR] missing helper: $required_file"
  fi
done

NODE_RANK="${RANK:-0}"
DLC_WORLD_SIZE="${WORLD_SIZE:-2}"
NODE0_ADDR="${MASTER_ADDR:-127.0.0.1}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-common-agent-opd}"
export WANDB_DIR="${WANDB_DIR:-$OUTPUT_DIR/wandb}"
export WANDB_NAME="${WANDB_NAME:-qwen3.5-4b-opd-v2}"
mkdir -p "$WANDB_DIR"

if [ ! -x "$PY" ]; then
  PY="$(command -v python3 2>/dev/null)"
fi
if [ ! -x "$SWIFT" ]; then
  SWIFT="$(command -v swift 2>/dev/null)"
fi

if [ -z "$PY" ]; then
  echo "[ERROR] python not found"
fi
if [ -z "$SWIFT" ]; then
  echo "[ERROR] swift not found in $ENV_DIR or PATH"
fi
if [ "$DLC_WORLD_SIZE" != "2" ]; then
  echo "[WARN] expected DLC WORLD_SIZE=2, got $DLC_WORLD_SIZE"
fi

export PYTHONPATH="$V1_DATA/ALFWorld/environment:$V1_DATA/WebShop/environment:${PYTHONPATH:-}"

"$PY" -c "import pyarrow, httpx, fastapi, sqlalchemy" >/dev/null 2>&1
if [ $? -ne 0 ]; then
  "$PY" -m pip install -q \
    pyarrow httpx fastapi sqlalchemy beautifulsoup4 flask \
    -i https://mirrors.aliyun.com/pypi/simple/
fi

if [ "$NODE_RANK" = "0" ]; then
  "$PY" -c "import textworld, alfworld" >/dev/null 2>&1
  if [ $? -ne 0 ] && [ -f "$V1_DATA/ALFWorld/environment/setup.py" ]; then
    "$PY" -m pip install -q -e "$V1_DATA/ALFWorld/environment" \
      -i https://mirrors.aliyun.com/pypi/simple/
  fi
fi

if [ ! -s "$TRAIN_JSONL" ]; then
  "$PY" "$PREP" \
    --stage 2 \
    --v1-root "$V1_DATA" \
    --v2-root "$V2_DATA" \
    --output "$TRAIN_JSONL"
fi

http_ready() {
  url="$1"
  count=0
  while [ "$count" -lt 180 ]; do
    curl -fsS "$url" >/dev/null 2>&1
    if [ $? -eq 0 ]; then
      return 0
    fi
    count=$((count + 1))
    sleep 2
  done
  echo "[WARN] service did not become ready: $url"
  return 1
}

stop_process() {
  pid="$1"
  if [ -n "$pid" ]; then
    kill "$pid" >/dev/null 2>&1
    sleep 3
    kill -9 "$pid" >/dev/null 2>&1
  fi
}

start_teacher() {
  echo "[NODE0] teacher TP=4 -> GPU 0,1,2,3"
  env -u RANK -u WORLD_SIZE -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    "$SWIFT" deploy \
      --model "$TEACHER_MODEL" \
      --infer_backend vllm \
      --host 0.0.0.0 \
      --port "$TEACHER_PORT" \
      --served_model_name qwen35-teacher \
      --max_logprobs "$TOPK" \
      --max_length "$MODEL_MAX_LENGTH" \
      --vllm_max_model_len "$MODEL_MAX_LENGTH" \
      --vllm_tensor_parallel_size 4 \
      --vllm_gpu_memory_utilization 0.90 \
      > "$LOG_DIR/teacher.log" 2>&1 &
  TEACHER_PID=$!
  http_ready "http://127.0.0.1:$TEACHER_PORT/v1/models"
  if [ $? -eq 0 ]; then
    touch "$CONTROL_DIR/teacher_ready"
  fi
}

start_rollout() {
  model_path="$1"
  log_tag="$2"
  echo "[NODE0] rollout TP=4 -> GPU 4,5,6,7 ; model=$model_path"
  env -u RANK -u WORLD_SIZE -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
    CUDA_VISIBLE_DEVICES=4,5,6,7 \
    PYTHONPATH="$PYTHONPATH" \
    "$SWIFT" rollout \
      --model "$model_path" \
      --host 0.0.0.0 \
      --port "$ROLLOUT_PORT" \
      --served_model_name qwen35-student \
      --vllm_tensor_parallel_size 4 \
      --vllm_gpu_memory_utilization 0.86 \
      --vllm_max_model_len "$MODEL_MAX_LENGTH" \
      --vllm_use_async_engine true \
      --external_plugins "$PLUGIN" \
      --multi_turn_scheduler gym_scheduler \
      --gym_env common_agent \
      --use_gym_env true \
      --max_turns "$MAX_TURNS" \
      > "$LOG_DIR/rollout_${log_tag}.log" 2>&1 &
  ROLLOUT_PID=$!
  http_ready "http://127.0.0.1:$ROLLOUT_PORT/v1/models"
}

start_eval_server() {
  model_path="$1"
  step="$2"
  echo "[NODE0] eval server TP=4 ; checkpoint=$model_path"
  env -u RANK -u WORLD_SIZE -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
    CUDA_VISIBLE_DEVICES=4,5,6,7 \
    "$SWIFT" deploy \
      --model "$model_path" \
      --infer_backend vllm \
      --host 0.0.0.0 \
      --port "$EVAL_PORT" \
      --served_model_name qwen35-student \
      --max_length "$MODEL_MAX_LENGTH" \
      --vllm_max_model_len "$MODEL_MAX_LENGTH" \
      --vllm_tensor_parallel_size 4 \
      --vllm_gpu_memory_utilization 0.86 \
      > "$LOG_DIR/eval_server_step_${step}.log" 2>&1 &
  EVAL_SERVER_PID=$!
  http_ready "http://127.0.0.1:$EVAL_PORT/v1/models"
}

prepare_toolsandbox_env() {
  EVAL_ENV="/mnt/nas/bihaoran/common_agent/envs/toolsandbox_eval"
  if [ ! -x "$EVAL_ENV/bin/python" ]; then
    "$PY" -m venv "$EVAL_ENV"
  fi
  EVAL_PY="$EVAL_ENV/bin/python"
  "$EVAL_PY" -c "import tool_sandbox, openai" >/dev/null 2>&1
  if [ $? -ne 0 ]; then
    "$EVAL_PY" -m pip install -q -e "$BENCHMARK" openai \
      -i https://mirrors.aliyun.com/pypi/simple/
  fi
}

run_toolsandbox() {
  step="$1"
  eval_out="$OUTPUT_DIR/eval/step_$step"
  mkdir -p "$eval_out"
  echo "[EVAL] ToolSandbox checkpoint-$step"

  if [ "$EVAL_TEST_MODE" = "1" ]; then
    STUDENT_OPENAI_URL="http://127.0.0.1:$EVAL_PORT/v1" \
    TEACHER_OPENAI_URL="http://127.0.0.1:$TEACHER_PORT/v1" \
      "$EVAL_PY" "$EVAL_SCRIPT" \
        --output-dir "$eval_out" \
        --parallel "$EVAL_PARALLEL" \
        --test-mode \
        > "$LOG_DIR/toolsandbox_step_${step}.log" 2>&1
  else
    STUDENT_OPENAI_URL="http://127.0.0.1:$EVAL_PORT/v1" \
    TEACHER_OPENAI_URL="http://127.0.0.1:$TEACHER_PORT/v1" \
      "$EVAL_PY" "$EVAL_SCRIPT" \
        --output-dir "$eval_out" \
        --parallel "$EVAL_PARALLEL" \
        > "$LOG_DIR/toolsandbox_step_${step}.log" 2>&1
  fi

  if [ -s "$eval_out/latest_result_summary.json" ]; then
    cp "$eval_out/latest_result_summary.json" "$OUTPUT_DIR/eval_step_$step.json"
  else
    echo "[WARN] ToolSandbox step $step did not produce latest_result_summary.json"
  fi
}

if [ "$NODE_RANK" = "0" ]; then
  rm -f "$CONTROL_DIR"/teacher_ready \
        "$CONTROL_DIR"/rollout_ready_* \
        "$CONTROL_DIR"/train_done_* \
        "$CONTROL_DIR"/eval_done_*

  start_teacher
  prepare_toolsandbox_env

  current_model="$START_MODEL"
  target="$CHUNK_STEPS"

  while [ "$target" -le "$TOTAL_STEPS" ]; do
    start_rollout "$current_model" "train_to_$target"
    if curl -fsS "http://127.0.0.1:$ROLLOUT_PORT/v1/models" >/dev/null 2>&1; then
      touch "$CONTROL_DIR/rollout_ready_$target"
    fi

    echo "[NODE0] waiting for training checkpoint-$target"
    while [ ! -s "$CONTROL_DIR/train_done_$target" ]; do
      sleep 5
    done
    checkpoint="$(cat "$CONTROL_DIR/train_done_$target")"

    stop_process "$ROLLOUT_PID"
    start_eval_server "$checkpoint" "$target"
    run_toolsandbox "$target"
    stop_process "$EVAL_SERVER_PID"

    current_model="$checkpoint"
    touch "$CONTROL_DIR/eval_done_$target"
    target=$((target + CHUNK_STEPS))
  done

  printf '%s\n' "$current_model" > "$OUTPUT_DIR/FINAL_CHECKPOINT"
  stop_process "$TEACHER_PID"
  echo "[DONE] OPD V2 final checkpoint: $current_model"

elif [ "$NODE_RANK" = "1" ]; then
  echo "[NODE1] waiting for teacher"
  while [ ! -f "$CONTROL_DIR/teacher_ready" ]; do
    sleep 5
  done

  current_checkpoint=""
  target="$CHUNK_STEPS"

  while [ "$target" -le "$TOTAL_STEPS" ]; do
    echo "[NODE1] waiting rollout server for target=$target"
    while [ ! -f "$CONTROL_DIR/rollout_ready_$target" ]; do
      sleep 5
    done

    if [ -n "$current_checkpoint" ]; then
      echo "[TRAIN] resume=$current_checkpoint -> global step $target"
      env -u RANK -u WORLD_SIZE -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
        MASTER_ADDR=127.0.0.1 \
        MASTER_PORT=$((29510 + 2)) \
        NPROC_PER_NODE=8 \
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        "$SWIFT" rlhf \
          --rlhf_type gkd \
          --model "$START_MODEL" \
          --tuner_type full \
          --teacher_model_server "http://$NODE0_ADDR:$TEACHER_PORT" \
          --gkd_logits_topk "$TOPK" \
          --dataset "$TRAIN_JSONL" \
          --lmbda 1.0 \
          --beta "$BETA" \
          --temperature 1.0 \
          --torch_dtype bfloat16 \
          --max_steps "$target" \
          --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
          --gradient_accumulation_steps "$GRAD_ACCUM" \
          --learning_rate "$LR" \
          --warmup_ratio 0.03 \
          --weight_decay 0.1 \
          --logging_steps 5 \
          --save_steps "$CHUNK_STEPS" \
          --save_total_limit 20 \
          --max_length "$MAX_PROMPT_LENGTH" \
          --max_completion_length "$MAX_COMPLETION_LENGTH" \
          --output_dir "$OUTPUT_DIR" \
          --gradient_checkpointing true \
          --dataloader_num_workers 4 \
          --dataset_num_proc 4 \
          --deepspeed zero2 \
          --attn_impl sdpa \
          --use_vllm true \
          --vllm_mode server \
          --vllm_server_host "$NODE0_ADDR" \
          --vllm_server_port "$ROLLOUT_PORT" \
          --vllm_server_pass_dataset true \
          --truncation_strategy delete \
          --log_completions true \
          --report_to tensorboard wandb \
          --resume_from_checkpoint "$current_checkpoint"
    else
      echo "[TRAIN] start=$START_MODEL -> global step $target"
      env -u RANK -u WORLD_SIZE -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
        MASTER_ADDR=127.0.0.1 \
        MASTER_PORT=$((29510 + 2)) \
        NPROC_PER_NODE=8 \
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        "$SWIFT" rlhf \
          --rlhf_type gkd \
          --model "$START_MODEL" \
          --tuner_type full \
          --teacher_model_server "http://$NODE0_ADDR:$TEACHER_PORT" \
          --gkd_logits_topk "$TOPK" \
          --dataset "$TRAIN_JSONL" \
          --lmbda 1.0 \
          --beta "$BETA" \
          --temperature 1.0 \
          --torch_dtype bfloat16 \
          --max_steps "$target" \
          --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
          --gradient_accumulation_steps "$GRAD_ACCUM" \
          --learning_rate "$LR" \
          --warmup_ratio 0.03 \
          --weight_decay 0.1 \
          --logging_steps 5 \
          --save_steps "$CHUNK_STEPS" \
          --save_total_limit 20 \
          --max_length "$MAX_PROMPT_LENGTH" \
          --max_completion_length "$MAX_COMPLETION_LENGTH" \
          --output_dir "$OUTPUT_DIR" \
          --gradient_checkpointing true \
          --dataloader_num_workers 4 \
          --dataset_num_proc 4 \
          --deepspeed zero2 \
          --attn_impl sdpa \
          --use_vllm true \
          --vllm_mode server \
          --vllm_server_host "$NODE0_ADDR" \
          --vllm_server_port "$ROLLOUT_PORT" \
          --vllm_server_pass_dataset true \
          --truncation_strategy delete \
          --log_completions true \
          --report_to tensorboard wandb
    fi

    expected="$OUTPUT_DIR/checkpoint-$target"
    if [ -d "$expected" ]; then
      current_checkpoint="$expected"
    else
      current_checkpoint="$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
    fi

    printf '%s\n' "$current_checkpoint" > "$CONTROL_DIR/train_done_$target"

    echo "[NODE1] waiting ToolSandbox evaluation for step=$target"
    while [ ! -f "$CONTROL_DIR/eval_done_$target" ]; do
      sleep 5
    done

    target=$((target + CHUNK_STEPS))
  done

  printf '%s\n' "$current_checkpoint" > "$OUTPUT_DIR/FINAL_CHECKPOINT"
  echo "[DONE] OPD V2 trainer final checkpoint: $current_checkpoint"

else
  echo "[WARN] unknown DLC RANK=$NODE_RANK; expected 0 or 1"
fi
