#!/usr/bin/env bash
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LINGBOT_VA_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
CF_ROOT="${COUNTERFACTUAL_ROOT:-$(cd "$LINGBOT_VA_ROOT/../.." && pwd)}"
cd "$LINGBOT_VA_ROOT"
export PYTHONPATH="$LINGBOT_VA_ROOT:${PYTHONPATH:-}"

PORT=${PORT:-$((29056))}
TEST_NUM=${TEST_NUM:-1}
OUT_DIR=${OUT_DIR:-outputs/robocasa}
SPLIT=${SPLIT:-target}
MAX_STEPS=${MAX_STEPS:-150}
CAMERA_SIZE=${CAMERA_SIZE:-256}
ROBOCASA_ROOT=${ROBOCASA_ROOT:-$CF_ROOT/../robocasa}
TASK_REGISTRY_JSON=${TASK_REGISTRY_JSON:-evaluation/robocasa/task_mobility_groups.json}
TASK_GROUP=${TASK_GROUP:-guaranteed_no_base_motion}
ROBOCASA_ACTION_MODE=${ROBOCASA_ACTION_MODE:-no_base}
ROBOCASA_VIEW_MODE=${ROBOCASA_VIEW_MODE:-2view_left_eih}
MUJOCO_GL=${MUJOCO_GL:-egl}
PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}

export MUJOCO_GL
export PYOPENGL_PLATFORM
export ROBOCASA_VIEW_MODE
export ROBOCASA_ACTION_MODE

if [ "$#" -gt 0 ]; then
    ENV_NAMES=("$@")
fi

if [ "$#" -gt 0 ]; then
    python evaluation/robocasa/client.py \
        --env-names "${ENV_NAMES[@]}" \
        --port "$PORT" \
        --test-num "$TEST_NUM" \
        --out-dir "$OUT_DIR" \
        --robocasa-root "$ROBOCASA_ROOT" \
        --split "$SPLIT" \
        --camera-heights "$CAMERA_SIZE" \
        --camera-widths "$CAMERA_SIZE" \
        --max-steps "$MAX_STEPS"
else
    python evaluation/robocasa/client.py \
        --task-registry-json "$TASK_REGISTRY_JSON" \
        --task-group "$TASK_GROUP" \
        --port "$PORT" \
        --test-num "$TEST_NUM" \
        --out-dir "$OUT_DIR" \
        --robocasa-root "$ROBOCASA_ROOT" \
        --split "$SPLIT" \
        --camera-heights "$CAMERA_SIZE" \
        --camera-widths "$CAMERA_SIZE" \
        --max-steps "$MAX_STEPS"
fi
