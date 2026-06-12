#!/usr/bin/env bash
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LINGBOT_VA_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
COUNTERFACTUAL_ROOT=${COUNTERFACTUAL_ROOT:-$(cd "$LINGBOT_VA_ROOT/../.." && pwd)}
cd "$LINGBOT_VA_ROOT"
export PYTHONPATH="$LINGBOT_VA_ROOT:${PYTHONPATH:-}"

START_PORT=${START_PORT:-$((29056))}
MASTER_PORT=${MASTER_PORT:-29063}
NGPU=${NGPU:-4}
SAVE_ROOT=${SAVE_ROOT:-visualization/robocasa}
MODEL_ROOT=${MODEL_ROOT:-${COUNTERFACTUAL_ROOT}/checkpoints/lingbot-va-posttrain-libero-long}
TASK_REGISTRY_JSON=${TASK_REGISTRY_JSON:-evaluation/robocasa/task_mobility_groups.json}
TASK_GROUP=${TASK_GROUP:-guaranteed_no_base_motion}
ROBOCASA_ACTION_MODE=${ROBOCASA_ACTION_MODE:-no_base}
ROBOCASA_VIEW_MODE=${ROBOCASA_VIEW_MODE:-2view_left_eih} # 2view_left_eih or 3view
MUJOCO_GL=${MUJOCO_GL:-egl}
PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}

export MUJOCO_GL
export PYOPENGL_PLATFORM
export MODEL_ROOT
export ROBOCASA_VIEW_MODE
export ROBOCASA_ACTION_MODE

mkdir -p "$SAVE_ROOT"

TASK_COUNT=$(python - "$TASK_REGISTRY_JSON" "$TASK_GROUP" <<'PY'
import json
import sys

registry_path, task_group = sys.argv[1], sys.argv[2]
with open(registry_path, "r", encoding="utf-8") as f:
    registry = json.load(f)
print(len(registry["groups"][task_group]))
PY
)

echo "RoboCasa task group: $TASK_GROUP ($TASK_COUNT tasks) from $TASK_REGISTRY_JSON"
echo "RoboCasa action mode: $ROBOCASA_ACTION_MODE"

if [ "$ROBOCASA_ACTION_MODE" = "no_base" ] || [ "$ROBOCASA_ACTION_MODE" = "arm_only" ]; then
    CONFIG_NAME=robocasa_no_base
else
    CONFIG_NAME=robocasa
fi

python -m torch.distributed.run \
    --nproc_per_node "$NGPU" \
    --master_port "$MASTER_PORT" \
    wan_va/wan_va_server.py \
    --config-name "$CONFIG_NAME" \
    --port "$START_PORT" \
    --save_root "$SAVE_ROOT"
