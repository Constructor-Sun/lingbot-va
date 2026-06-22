#!/bin/sh
set -eu

# Attention-capture server for generated-video visualization.
# This uses the same server as launch_server_attn_multigpus.sh; generated video
# is only decoded when the matching client sends --return_video.
# Run from the lingbot-va root.
#
# Usage:
#   CHECKPOINT_ROOT=/path/to/robot_sft \
#   bash evaluation/robotwin/launch_server_attn_video_multigpus.sh [START_PORT] [MASTER_PORT] [GPU_OFFSET] [NUM_GPUS]

START_PORT=${1:-29556}
MASTER_PORT=${2:-29661}
GPU_OFFSET=${3:-0}
NUM_GPUS=${4:-2}

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
LINGBOT_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)"

LOG_DIR=${LOG_DIR:-'./logs'}
SAVE_ROOT=${SAVE_ROOT:-"$LINGBOT_ROOT/../RoboTwin/results/attn_video_generated"}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-"$LINGBOT_ROOT/../../checkpoints/lingbot-va-posttrain-robotwin"}
CONFIG_NAME=${CONFIG_NAME:-robotwin}
ATTN_TARGETS=${ATTN_TARGETS:-action_to_video}
ATTN_LAST_LAYERS=${ATTN_LAST_LAYERS:-0}

mkdir -p "$LOG_DIR" "$SAVE_ROOT"

export LINGBOT_CHECKPOINT_PATH="$CHECKPOINT_ROOT"

batch_time=$(date +%Y%m%d_%H%M%S)

for i in $(seq 0 $((NUM_GPUS - 1))); do
    gpu_id=$((i + GPU_OFFSET))
    current_port=$((START_PORT + i))
    current_master_port=$((MASTER_PORT + i))

    log_file="${LOG_DIR}/attn_video_server_${gpu_id}_${batch_time}.log"
    echo "[Generated-video server $i] GPU: ${gpu_id} | PORT: ${current_port} | MASTER_PORT: ${current_master_port} | Log: ${log_file}"

    CUDA_VISIBLE_DEVICES=${gpu_id} \
    nohup python -m torch.distributed.run \
        --nproc_per_node 1 \
        --master_port "$current_master_port" \
        evaluation/robotwin/wan_server_attn_capture.py \
        --config-name "$CONFIG_NAME" \
        --save-root "$SAVE_ROOT" \
        --attn-targets "$ATTN_TARGETS" \
        --attn-last-layers "$ATTN_LAST_LAYERS" \
        --port "$current_port" > "$log_file" 2>&1 &

    sleep 2
done

echo "All generated-video attention server instances have been launched in the background."
wait
