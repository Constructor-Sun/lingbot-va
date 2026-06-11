import os
import re

import numpy as np


ALL_ROBOCASA_IMAGE_KEY_MAP = {
    "observation.images.robot0_agentview_left": "video.robot0_agentview_left",
    "observation.images.robot0_agentview_right": "video.robot0_agentview_right",
    "observation.images.robot0_eye_in_hand": "video.robot0_eye_in_hand",
    "observation.images.robot0_agentview_center": "video.robot0_agentview_center",
}


def _get_view_keys():
    view_mode = os.environ.get("ROBOCASA_VIEW_MODE", "2view_left_eih")
    if view_mode == "3view":
        return [
            "observation.images.robot0_agentview_left",
            "observation.images.robot0_agentview_right",
            "observation.images.robot0_eye_in_hand",
        ]
    if view_mode == "2view_left_eih":
        return [
            "observation.images.robot0_agentview_left",
            "observation.images.robot0_eye_in_hand",
        ]
    raise ValueError(
        f"Unsupported ROBOCASA_VIEW_MODE={view_mode!r}. "
        "Expected one of: '3view', '2view_left_eih'."
    )


VA_IMAGE_KEYS = _get_view_keys()
ROBOCASA_IMAGE_KEY_MAP = {
    key: ALL_ROBOCASA_IMAGE_KEY_MAP[key] for key in VA_IMAGE_KEYS
}


def _get_action_mode():
    action_mode = os.environ.get("ROBOCASA_ACTION_MODE", "no_base")
    if action_mode == "arm_only":
        action_mode = "no_base"
    if action_mode not in {"full", "no_base"}:
        raise ValueError(
            f"Unsupported ROBOCASA_ACTION_MODE={action_mode!r}. "
            "Expected one of: 'full', 'no_base', 'arm_only'."
        )
    return action_mode


ROBOCASA_ACTION_MODE = _get_action_mode()
FULL_ACTION_DIM = 12
if ROBOCASA_ACTION_MODE == "full":
    ACTION_DIM = 12
    ACTION_SCHEMA = {
        "end_effector_position": slice(0, 3),
        "end_effector_rotation": slice(3, 6),
        "gripper_close": slice(6, 7),
        "base_motion": slice(7, 11),
        "control_mode": slice(11, 12),
    }
else:
    # LingBot arm-only output order, aligned with LIBERO-style 7D action:
    # [ee_pos(3), ee_rot(3), gripper(1)].
    # RoboCasa-specific base / control channels are injected as zeros below.
    ACTION_DIM = 7
    ACTION_SCHEMA = {
        "end_effector_position": slice(0, 3),
        "end_effector_rotation": slice(3, 6),
        "gripper_close": slice(6, 7),
    }


def sanitize_filename(text, max_len=120):
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(text)).strip("_")
    return text[:max_len] or "episode"


def extract_prompt(obs):
    prompt = obs.get("annotation.human.task_description", "")
    if isinstance(prompt, np.ndarray):
        prompt = prompt.item() if prompt.shape == () else str(prompt.tolist())
    return str(prompt)


def extract_va_obs(obs):
    return {
        va_key: np.ascontiguousarray(obs[robocasa_key])
        for va_key, robocasa_key in ROBOCASA_IMAGE_KEY_MAP.items()
    }


def action_to_robocasa_dict(action_step, control_mode_threshold=0.5):
    action_step = np.asarray(action_step, dtype=np.float32).reshape(-1)
    if action_step.shape[0] < ACTION_DIM:
        raise ValueError(
            f"RoboCasa action requires at least {ACTION_DIM} dims, got {action_step.shape[0]}"
        )

    if ROBOCASA_ACTION_MODE == "full":
        control_mode_raw = action_step[ACTION_SCHEMA["control_mode"]][0]
        control_mode = np.array(
            [1.0 if control_mode_raw >= control_mode_threshold else 0.0],
            dtype=np.float32,
        )
        base_motion = action_step[ACTION_SCHEMA["base_motion"]]
    else:
        control_mode = np.zeros(1, dtype=np.float32)
        base_motion = np.zeros(4, dtype=np.float32)

    return {
        "action.base_motion": base_motion,
        "action.control_mode": control_mode,
        "action.end_effector_position": action_step[
            ACTION_SCHEMA["end_effector_position"]
        ],
        "action.end_effector_rotation": action_step[
            ACTION_SCHEMA["end_effector_rotation"]
        ],
        "action.gripper_close": action_step[ACTION_SCHEMA["gripper_close"]],
    }
