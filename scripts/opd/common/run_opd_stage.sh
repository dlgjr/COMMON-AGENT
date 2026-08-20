#!/usr/bin/env bash
set -Eeuo pipefail

# Shared COMMON-AGENT OPD-RL runner.
# Expected DLC topology:
#   node rank 0: devices 0-3 teacher TP=4; devices 4-7 rollout/eval TP=4
#   node rank 1: all visible devices full-parameter GRPO + teacher-KL training

OPD_STAGE="${OPD_STAGE:?OPD_STAGE must be 1 or 2}"
COMMON_DIR="/mnt/nas/bihaoran/common_agent/scripts/opd/common"
ENV_DIR="/mnt/nas/bihaoran/common_agent/envs/qwen35_swift"
PY="$ENV_DIR/bin/python"
SWIFT="$ENV_DIR/bin/swift"

TEACHER_MODEL="/mnt/nas/bihaoran/common_agent/model/teacher/Qwen3.5-122B-A10B"
V1_DATA="/mnt/nas/bihaoran/agent_data/opd_v1"
V2_DATA="/mnt/nas/bihaoran/agent_data/opd_v2"
BENCHMARK="/mnt/nas/bihaoran/common_agent/benchmark/ToolSandbox"
PLUGIN="$COMMON_DIR/common_agent_opd_plugin.py"
PREP="$COMMON_DIR/prepare_opd_data.py"
EVAL_SCRIPT="$COMMON_DIR/toolsandbox_eval.py"
EVAL_ENV="/mnt/nas/bihaoran/common_agent/envs/toolsandbox_eval"
EVAL_PY="$EVAL_ENV/bin/python"

case "$OPD_STAGE" in
  1)
    START_MODEL="${STUDENT_MODEL:-/mnt/nas/bihaoran/common_agent/output/sft/qwen3.5-4b-base-coldstart}"
    DEFAULT_OUTPUT_DIR="/mnt/nas/bihaoran/common_agent/output/opd/opd_v1"
    DEFAULT_TOTAL_STEPS=2000
    DEFAULT_MAX_TURNS=8
    DEFAULT_MAX_PROMPT_LENGTH=8192
    DEFAULT_MODEL_MAX_LENGTH=16384
    DEFAULT_LR=5e-6
    DEFAULT_TRAIN_MASTER_PORT=29511
    DEFAULT_WANDB_NAME="qwen3.5-4b-opd-v1"
    ;;
  2)
    if [ -n "${STUDENT_MODEL:-}" ]; then
      START_MODEL="$STUDENT_MODEL"
    elif [ -s "/mnt/nas/bihaoran/common_agent/output/opd/opd_v1/FINAL_CHECKPOINT" ]; then
      START_MODEL="$(cat /mnt/nas/bihaoran/common_agent/output/opd/opd_v1/FINAL_CHECKPOINT)"
    else
      START_MODEL="/mnt/nas/bihaoran/common_agent/output/opd/opd_v1/checkpoint-2000"
    fi
    DEFAULT_OUTPUT_DIR="/mnt/nas/bihaoran/common_agent/output/opd/opd_v2"
    DEFAULT_TOTAL_STEPS=3000
    DEFAULT_MAX_TURNS=32
    DEFAULT_MAX_PROMPT_LENGTH=24576
    DEFAULT_MODEL_MAX_LENGTH=32768
    DEFAULT_LR=2e-6
    DEFAULT_TRAIN_MASTER_PORT=29512
    DEFAULT_WANDB_NAME="qwen3.5-4b-opd-v2"
    ;;
  *)
    echo "[ERROR] OPD_STAGE must be 1 or 2, got: $OPD_STAGE" >&2
    exit 1
    ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
WORK_DIR="$OUTPUT_DIR/runtime"
CONTROL_DIR="$OUTPUT_DIR/control"
LOG_DIR="$OUTPUT_DIR/logs"
TRAIN_JSONL="$WORK_DIR/train.jsonl"

TEACHER_PORT="${TEACHER_PORT:-18000}"
ROLLOUT_PORT="${ROLLOUT_PORT:-18001}"
EVAL_PORT="${EVAL_PORT:-18002}"
TEACHER_MAX_LOGPROBS="${TEACHER_MAX_LOGPROBS:-1}"

TOTAL_STEPS="${TOTAL_STEPS:-$DEFAULT_TOTAL_STEPS}"
CHUNK_STEPS="${CHUNK_STEPS:-500}"
MAX_TURNS="${MAX_TURNS:-$DEFAULT_MAX_TURNS}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-$DEFAULT_MAX_PROMPT_LENGTH}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-512}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-$DEFAULT_MODEL_MAX_LENGTH}"

PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
STEPS_PER_GENERATION="${STEPS_PER_GENERATION:-$GRAD_ACCUM}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
LR="${LR:-$DEFAULT_LR}"
TEACHER_KL_COEF="${TEACHER_KL_COEF:-1.0}"
POLICY_KL_BETA="${POLICY_KL_BETA:-0}"
EVAL_PARALLEL="${EVAL_PARALLEL:-16}"
EVAL_TEST_MODE="${EVAL_TEST_MODE:-0}"
SERVICE_READY_TIMEOUT="${SERVICE_READY_TIMEOUT:-360}"
CONTROL_WAIT_TIMEOUT="${CONTROL_WAIT_TIMEOUT:-172800}"
TRAIN_MASTER_PORT="${TRAIN_MASTER_PORT:-$DEFAULT_TRAIN_MASTER_PORT}"

NODE_RANK="${RANK:-0}"
DLC_WORLD_SIZE="${WORLD_SIZE:-2}"
NODE0_ADDR="${MASTER_ADDR:-}"

mkdir -p "$WORK_DIR" "$CONTROL_DIR" "$LOG_DIR" "$OUTPUT_DIR/eval"
FAIL_MARKER="$CONTROL_DIR/node_${NODE_RANK}_failed"
rm -f "$FAIL_MARKER"

fail() {
  local message="$1"
  echo "[ERROR] $message" >&2
  printf '%s\n' "$message" > "$FAIL_MARKER" 2>/dev/null || true
  exit 1
}

on_error() {
  local rc=$?
  local line="${BASH_LINENO[0]:-unknown}"
  local cmd="${BASH_COMMAND:-unknown}"
  trap - ERR
  printf 'rc=%s line=%s cmd=%s\n' "$rc" "$line" "$cmd" > "$FAIL_MARKER" 2>/dev/null || true
  exit "$rc"
}
trap on_error ERR

[ "$DLC_WORLD_SIZE" = "2" ] || fail "expected DLC WORLD_SIZE=2, got $DLC_WORLD_SIZE"
case "$NODE_RANK" in
  0|1) ;;
  *) fail "unknown DLC RANK=$NODE_RANK; expected 0 or 1" ;;
esac
[ -n "$NODE0_ADDR" ] || fail "MASTER_ADDR is required for the two-node OPD job"
[ -x "$PY" ] || fail "NAS python not found: $PY"
[ -x "$SWIFT" ] || fail "NAS swift not found: $SWIFT"

for required_file in "$PLUGIN" "$PREP" "$EVAL_SCRIPT"; do
  [ -s "$required_file" ] || fail "missing helper: $required_file"
done

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-common-agent-opd}"
export WANDB_DIR="${WANDB_DIR:-$OUTPUT_DIR/wandb}"
export WANDB_NAME="${WANDB_NAME:-$DEFAULT_WANDB_NAME}"
export PYTHONPATH="$V1_DATA/ALFWorld/environment:$V1_DATA/WebShop/environment:${PYTHONPATH:-}"
mkdir -p "$WANDB_DIR"

# DLC only validates the prepared NAS environment. It never pip-installs into it.
if ! "$PY" -c "import torch, deepspeed, transformers, swift, vllm, pyarrow, wandb" >/dev/null; then
  fail "NAS training env is incomplete; install OPD dependencies before starting DLC"
fi

AVAILABLE_DEVICES="$("$PY" -c 'import torch; print(torch.cuda.device_count())')"
[[ "$AVAILABLE_DEVICES" =~ ^[0-9]+$ ]] || fail "invalid torch.cuda.device_count(): $AVAILABLE_DEVICES"
[ "$AVAILABLE_DEVICES" -gt 0 ] || fail "no PPU devices are visible to torch"

TRAIN_NPROC="${TRAIN_NPROC_PER_NODE:-$AVAILABLE_DEVICES}"
[[ "$TRAIN_NPROC" =~ ^[1-9][0-9]*$ ]] || fail "TRAIN_NPROC_PER_NODE must be a positive integer"
[ "$TRAIN_NPROC" -le "$AVAILABLE_DEVICES" ] || \
  fail "TRAIN_NPROC_PER_NODE=$TRAIN_NPROC exceeds visible devices=$AVAILABLE_DEVICES"

make_visible_devices() {
  local n="$1"
  local out=""
  local i
  for ((i=0; i<n; i++)); do
    [ -z "$out" ] || out+=","
    out+="$i"
  done
  printf '%s\n' "$out"
}
TRAIN_VISIBLE_DEVICES="${TRAIN_VISIBLE_DEVICES:-$(make_visible_devices "$TRAIN_NPROC")}"

if [ "$NODE_RANK" = "0" ] && [ "$AVAILABLE_DEVICES" -lt 8 ]; then
  fail "node0 topology requires at least 8 visible devices; got $AVAILABLE_DEVICES"
fi

if [ "$NODE_RANK" = "1" ]; then
  generation_batch=$((PER_DEVICE_BATCH_SIZE * TRAIN_NPROC * STEPS_PER_GENERATION))
  [ "$generation_batch" -gt 0 ] || fail "invalid GRPO generation batch size: $generation_batch"
  if [ $((generation_batch % NUM_GENERATIONS)) -ne 0 ]; then
    fail "generation batch $generation_batch must be divisible by NUM_GENERATIONS=$NUM_GENERATIONS"
  fi
fi

if [ "$NODE_RANK" = "0" ]; then
  if ! "$PY" -c "import httpx, fastapi, sqlalchemy, bs4, flask, textworld, alfworld; from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv" >/dev/null; then
    fail "node0 environment dependencies are missing; prepare AWM/ALFWorld/WebShop dependencies before DLC"
  fi
fi

peer_failed() {
  local peer marker
  if [ "$NODE_RANK" = "0" ]; then peer=1; else peer=0; fi
  marker="$CONTROL_DIR/node_${peer}_failed"
  if [ -s "$marker" ]; then
    echo "[ERROR] peer node $peer failed: $(cat "$marker")" >&2
    return 0
  fi
  return 1
}

wait_for_file() {
  local path="$1"
  local description="$2"
  local require_nonempty="${3:-0}"
  local started now
  started="$(date +%s)"
  while true; do
    if [ "$require_nonempty" = "1" ]; then
      [ -s "$path" ] && return 0
    else
      [ -e "$path" ] && return 0
    fi
    peer_failed && fail "aborting while waiting for $description"
    now="$(date +%s)"
    [ $((now - started)) -lt "$CONTROL_WAIT_TIMEOUT" ] || fail "timeout waiting for $description: $path"
    sleep 5
  done
}

http_ready() {
  local url="$1"
  local elapsed=0
  while [ "$elapsed" -lt "$SERVICE_READY_TIMEOUT" ]; do
    if curl -fsS "$url" >/dev/null 2>&1; then return 0; fi
    sleep 2
    elapsed=$((elapsed + 2))
  done
  return 1
}

stop_process() {
  local pid="${1:-}"
  local i
  if [ -z "$pid" ] || ! kill -0 "$pid" >/dev/null 2>&1; then return 0; fi
  kill "$pid" >/dev/null 2>&1 || true
  for ((i=0; i<10; i++)); do
    if ! kill -0 "$pid" >/dev/null 2>&1; then return 0; fi
    sleep 1
  done
  kill -9 "$pid" >/dev/null 2>&1 || true
}

TEACHER_PID=""
ROLLOUT_PID=""
EVAL_SERVER_PID=""
cleanup() {
  stop_process "${ROLLOUT_PID:-}" || true
  stop_process "${EVAL_SERVER_PID:-}" || true
  stop_process "${TEACHER_PID:-}" || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

prepare_train_data() {
  if [ "$NODE_RANK" = "0" ]; then
    if [ ! -s "$TRAIN_JSONL" ]; then
      "$PY" "$PREP" \
        --stage "$OPD_STAGE" \
        --v1-root "$V1_DATA" \
        --v2-root "$V2_DATA" \
        --output "$TRAIN_JSONL"
    fi
    [ -s "$TRAIN_JSONL" ] || fail "OPD dataset preparation did not produce $TRAIN_JSONL"
  else
    wait_for_file "$TRAIN_JSONL" "prepared OPD dataset" 1
  fi
}

start_teacher() {
  echo "[NODE0] teacher TP=4 -> devices 0,1,2,3"
  env -u RANK -u WORLD_SIZE -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    "$SWIFT" deploy \
      --model "$TEACHER_MODEL" \
      --infer_backend vllm \
      --host 0.0.0.0 \
      --port "$TEACHER_PORT" \
      --served_model_name qwen35-teacher \
      --max_logprobs "$TEACHER_MAX_LOGPROBS" \
      --max_length "$MODEL_MAX_LENGTH" \
      --vllm_max_model_len "$MODEL_MAX_LENGTH" \
      --vllm_tensor_parallel_size 4 \
      --vllm_gpu_memory_utilization 0.90 \
      > "$LOG_DIR/teacher.log" 2>&1 &
  TEACHER_PID=$!
  http_ready "http://127.0.0.1:$TEACHER_PORT/v1/models" || \
    fail "teacher server did not become ready; see $LOG_DIR/teacher.log"
  touch "$CONTROL_DIR/teacher_ready"
}

start_rollout() {
  local model_path="$1"
  local log_tag="$2"
  echo "[NODE0] rollout TP=4 -> devices 4,5,6,7 ; model=$model_path"
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
  http_ready "http://127.0.0.1:$ROLLOUT_PORT/v1/models" || \
    fail "rollout server did not become ready; see $LOG_DIR/rollout_${log_tag}.log"
}

start_eval_server() {
  local model_path="$1"
  local step="$2"
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
  http_ready "http://127.0.0.1:$EVAL_PORT/v1/models" || \
    fail "eval server did not become ready; see $LOG_DIR/eval_server_step_${step}.log"
}

check_toolsandbox_env() {
  [ -d "$BENCHMARK" ] || fail "ToolSandbox benchmark not found: $BENCHMARK"
  [ -x "$EVAL_PY" ] || fail "ToolSandbox env missing: $EVAL_PY; prepare it before starting DLC"
  if ! "$EVAL_PY" -c "import tool_sandbox, openai" >/dev/null; then
    fail "ToolSandbox env is incomplete; install tool_sandbox/openai before starting DLC"
  fi
}

run_toolsandbox() {
  local step="$1"
  local eval_out="$OUTPUT_DIR/eval/step_$step"
  mkdir -p "$eval_out"
  echo "[EVAL] ToolSandbox checkpoint-$step"
  if [ "$EVAL_TEST_MODE" = "1" ]; then
    STUDENT_OPENAI_URL="http://127.0.0.1:$EVAL_PORT/v1" \
    TEACHER_OPENAI_URL="http://127.0.0.1:$TEACHER_PORT/v1" \
      "$EVAL_PY" "$EVAL_SCRIPT" --output-dir "$eval_out" --parallel "$EVAL_PARALLEL" --test-mode \
      > "$LOG_DIR/toolsandbox_step_${step}.log" 2>&1
  else
    STUDENT_OPENAI_URL="http://127.0.0.1:$EVAL_PORT/v1" \
    TEACHER_OPENAI_URL="http://127.0.0.1:$TEACHER_PORT/v1" \
      "$EVAL_PY" "$EVAL_SCRIPT" --output-dir "$eval_out" --parallel "$EVAL_PARALLEL" \
      > "$LOG_DIR/toolsandbox_step_${step}.log" 2>&1
  fi
  [ -s "$eval_out/latest_result_summary.json" ] || \
    fail "ToolSandbox step $step did not produce latest_result_summary.json"
  cp "$eval_out/latest_result_summary.json" "$OUTPUT_DIR/eval_step_$step.json"
}

run_training_chunk() {
  local target="$1"
  local resume_checkpoint="${2:-}"
  local args=(
    rlhf
    --rlhf_type grpo
    --model "$START_MODEL"
    --tuner_type full
    --teacher_model_server "http://$NODE0_ADDR:$TEACHER_PORT"
    --teacher_kl_coef "$TEACHER_KL_COEF"
    --dataset "$TRAIN_JSONL"
    --temperature 1.0
    --torch_dtype bfloat16
    --max_steps "$target"
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE"
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --steps_per_generation "$STEPS_PER_GENERATION"
    --num_generations "$NUM_GENERATIONS"
    --learning_rate "$LR"
    --warmup_ratio 0.03
    --weight_decay 0.1
    --logging_steps 5
    --save_steps "$CHUNK_STEPS"
    --save_total_limit 20
    --max_length "$MAX_PROMPT_LENGTH"
    --max_completion_length "$MAX_COMPLETION_LENGTH"
    --output_dir "$OUTPUT_DIR"
    --gradient_checkpointing true
    --dataloader_num_workers 4
    --dataset_num_proc 4
    --split_dataset_ratio 0
    --deepspeed zero2
    --attn_impl sdpa
    --use_vllm true
    --vllm_mode server
    --vllm_server_host "$NODE0_ADDR"
    --vllm_server_port "$ROLLOUT_PORT"
    --vllm_server_pass_dataset true
    --use_gym_env true
    --beta "$POLICY_KL_BETA"
    --truncation_strategy delete
    --log_completions true
    --report_to tensorboard wandb
  )
  if [ -n "$resume_checkpoint" ]; then
    args+=(--resume_from_checkpoint "$resume_checkpoint")
  fi

  env -u RANK -u WORLD_SIZE -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
    MASTER_ADDR=127.0.0.1 \
    MASTER_PORT="$TRAIN_MASTER_PORT" \
    NPROC_PER_NODE="$TRAIN_NPROC" \
    CUDA_VISIBLE_DEVICES="$TRAIN_VISIBLE_DEVICES" \
    "$SWIFT" "${args[@]}"
}

if [ "$NODE_RANK" = "0" ]; then
  rm -f "$CONTROL_DIR"/teacher_ready \
        "$CONTROL_DIR"/rollout_ready_* \
        "$CONTROL_DIR"/train_done_* \
        "$CONTROL_DIR"/eval_done_* \
        "$CONTROL_DIR"/node_0_failed \
        "$CONTROL_DIR"/node_1_failed
  FAIL_MARKER="$CONTROL_DIR/node_0_failed"

  prepare_train_data
  check_toolsandbox_env
  start_teacher

  current_model="$START_MODEL"
  target="$CHUNK_STEPS"
  while [ "$target" -le "$TOTAL_STEPS" ]; do
    start_rollout "$current_model" "train_to_$target"
    touch "$CONTROL_DIR/rollout_ready_$target"

    echo "[NODE0] waiting for exact training checkpoint-$target"
    wait_for_file "$CONTROL_DIR/train_done_$target" "training checkpoint-$target signal" 1
    checkpoint="$(cat "$CONTROL_DIR/train_done_$target")"
    [ "$checkpoint" = "$OUTPUT_DIR/checkpoint-$target" ] || \
      fail "trainer reported unexpected checkpoint for step $target: $checkpoint"
    [ -d "$checkpoint" ] || fail "reported checkpoint does not exist: $checkpoint"

    stop_process "$ROLLOUT_PID"
    ROLLOUT_PID=""
    start_eval_server "$checkpoint" "$target"
    run_toolsandbox "$target"
    stop_process "$EVAL_SERVER_PID"
    EVAL_SERVER_PID=""

    current_model="$checkpoint"
    touch "$CONTROL_DIR/eval_done_$target"
    target=$((target + CHUNK_STEPS))
  done

  printf '%s\n' "$current_model" > "$OUTPUT_DIR/FINAL_CHECKPOINT"
  stop_process "$TEACHER_PID"
  TEACHER_PID=""
  echo "[DONE] OPD V$OPD_STAGE final checkpoint: $current_model"
else
  FAIL_MARKER="$CONTROL_DIR/node_1_failed"
  rm -f "$FAIL_MARKER"
  prepare_train_data

  echo "[NODE1] waiting for teacher"
  wait_for_file "$CONTROL_DIR/teacher_ready" "teacher readiness" 0

  current_checkpoint=""
  target="$CHUNK_STEPS"
  while [ "$target" -le "$TOTAL_STEPS" ]; do
    echo "[NODE1] waiting rollout server for target=$target"
    wait_for_file "$CONTROL_DIR/rollout_ready_$target" "rollout server for target=$target" 0

    if [ -n "$current_checkpoint" ]; then
      echo "[TRAIN] resume=$current_checkpoint -> global step $target"
    else
      echo "[TRAIN] start=$START_MODEL -> global step $target"
    fi
    run_training_chunk "$target" "$current_checkpoint"

    expected="$OUTPUT_DIR/checkpoint-$target"
    [ -d "$expected" ] || fail "training returned successfully but exact checkpoint is missing: $expected"
    current_checkpoint="$expected"
    printf '%s\n' "$current_checkpoint" > "$CONTROL_DIR/train_done_$target"

    echo "[NODE1] waiting ToolSandbox evaluation for step=$target"
    wait_for_file "$CONTROL_DIR/eval_done_$target" "ToolSandbox evaluation for step=$target" 0
    target=$((target + CHUNK_STEPS))
  done

  printf '%s\n' "$current_checkpoint" > "$OUTPUT_DIR/FINAL_CHECKPOINT"
  echo "[DONE] OPD V$OPD_STAGE trainer final checkpoint: $current_checkpoint"
fi
