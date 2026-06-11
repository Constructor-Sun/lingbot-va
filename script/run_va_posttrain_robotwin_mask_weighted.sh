#!/usr/bin/bash

set -x

umask 007

NGPU=${NGPU:-"1"}
MASTER_PORT=${MASTER_PORT:-"29501"}
PORT=${PORT:-"1106"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"robotwin_train_mask_weighted"}
SAVE_ROOT=${SAVE_ROOT:-"./checkpoints/mask_weight_5"}

export WANDB_API_KEY="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"
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_TEAM_NAME="haiying"
export WANDB_PROJECT="lingbot-va-robotwin"

num_gpu=${NGPU}
master_port=${MASTER_PORT}
log_rank=${LOG_RANK}
torchft_lighthouse=${TORCHFT_LIGHTHOUSE}
config_name=${CONFIG_NAME}
save_root=${SAVE_ROOT}

export TOKENIZERS_PARALLELISM=false
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${torchft_lighthouse} \
python -m torch.distributed.run \
    --nproc_per_node=${num_gpu} \
    --local-ranks-filter=${log_rank} \
    --master_port ${master_port} \
    --tee 3 \
    -m wan_va.train \
    --config-name ${config_name} \
    --save-root "${save_root}" \
    "$@"
