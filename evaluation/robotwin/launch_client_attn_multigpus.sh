#!/bin/sh
set -eu

export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:${LD_LIBRARY_PATH:-}
export ROBOTWIN_CAMERA_SHADER_DIR="${ROBOTWIN_CAMERA_SHADER_DIR:-default}"

# Attention-capture client launcher.
# Run from the lingbot-va root.
#
# Usage:
#   bash evaluation/robotwin/launch_client_attn_multigpus.sh [SAVE_ROOT] [TASKS] [TASK_CONFIG] [GPU_OFFSET] [SEED] [TEST_NUM] [RUN_TAG] [START_PORT]
#
# Examples:
#   bash evaluation/robotwin/launch_client_attn_multigpus.sh ./results click_alarmclock demo_clean.yml 0 10000 1 attn
#   TASKS="click_alarmclock blocks_ranking_size" bash evaluation/robotwin/launch_client_attn_multigpus.sh

save_root=${1:-'./results/attn_video'}
tasks_arg=${2:-"${TASKS:-click_alarmclock}"}
task_config=${3:-"${TASK_CONFIG:-demo_clean.yml}"}
task_config=${task_config%.yml}
gpu_offset=${4:-"${GPU_OFFSET:-0}"}
seed=${5:-"${SEED:-0}"}
test_num=${6:-"${TEST_NUM:-1}"}
run_tag=${7:-"${RUN_TAG:-attn-gpu${gpu_offset}}"}
start_port=${8:-"${START_PORT:-29556}"}
num_gpus=${NUM_GPUS:-2}
policy_name=${POLICY_NAME:-ACT}
train_config_name=${TRAIN_CONFIG_NAME:-0}
model_name=${MODEL_NAME:-0}

log_dir=${LOG_DIR:-"./logs"}
mkdir -p "$log_dir"

if [ -z "$tasks_arg" ]; then
    echo "tasks_arg is empty" >&2
    exit 1
fi

set -- $tasks_arg
task_count=$#

echo "task_names (${task_count}): $tasks_arg"
echo "task_config=${task_config}"
echo "seed=${seed}"
echo "start_port=${start_port}"

pid_file="pids_${run_tag}.txt"
> "$pid_file"

batch_time=$(date +%Y%m%d_%H%M%S)

i=0
for task_name in "$@"; do
    gpu_id=$((i % num_gpus + gpu_offset))
    port=$((start_port + i + gpu_offset))
    export CUDA_VISIBLE_DEVICES="${gpu_id}"

    log_file="${log_dir}/${task_name}_${run_tag}_${batch_time}.log"
    echo "[Task $i] Task: ${task_name}, GPU: ${gpu_id}, PORT: ${port}, Log: ${log_file}"

    PYTHONWARNINGS=ignore::UserWarning \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    python -m evaluation.robotwin.eval_policy_client_attn_video \
        --config policy/${policy_name}/deploy_policy.yml \
        --overrides \
        --task_name "${task_name}" \
        --task_config "${task_config}" \
        --train_config_name "${train_config_name}" \
        --model_name "${model_name}" \
        --ckpt_setting "${model_name}" \
        --seed "${seed}" \
        --policy_name "${policy_name}" \
        --save_root "${save_root}" \
        --video_guidance_scale 5 \
        --action_guidance_scale 1 \
        --test_num "${test_num}" \
        --port "${port}" > "$log_file" 2>&1 &

    pid=$!
    echo "${pid}" | tee -a "$pid_file"
    i=$((i + 1))
done

echo "All attention-capture clients launched. PIDs saved to ${pid_file}"
echo "To terminate all processes, run: kill \$(cat ${pid_file})"
