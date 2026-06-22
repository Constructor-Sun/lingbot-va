#!/usr/bin/env bash

START_PORT=${START_PORT:-29169}
MASTER_PORT=${MASTER_PORT:-29179}
GPU_ID=${GPU_ID:-0}
CONFIG_NAME=${CONFIG_NAME:-robotwin}
SAVE_ROOT=${SAVE_ROOT:-visualization_token_cas/}
LINGBOT_CHECKPOINT_PATH=${LINGBOT_CHECKPOINT_PATH:-/data1/liu/exp/counterfactual/checkpoints/lingbot-va-posttrain-robotwin}

mkdir -p "$SAVE_ROOT"

CUDA_VISIBLE_DEVICES=$GPU_ID \
LINGBOT_CHECKPOINT_PATH=$LINGBOT_CHECKPOINT_PATH \
python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port "$MASTER_PORT" \
    evaluation/robotwin/wan_server_token_cas.py \
    --config-name "$CONFIG_NAME" \
    --port "$START_PORT" \
    --save-root "$SAVE_ROOT"
