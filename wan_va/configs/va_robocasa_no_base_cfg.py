# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from copy import deepcopy

from .va_robocasa_cfg import va_robocasa_cfg


va_robocasa_no_base_cfg = deepcopy(va_robocasa_cfg)
va_robocasa_no_base_cfg.__name__ = 'Config: VA robocasa no base'

# Reuse the LingBot canonical RoboCasa layout from `va_robocasa_cfg`:
#   0:3 ee_pos, 3:6 ee_rot, 6:7 gripper, 7:11 base_motion, 11:12 control_mode
# The raw RoboCasa LeRobot export can store base / control before the arm dims,
# but `compute_robocasa_norm_stat.py` writes stats back in this canonical order.
# So the first 7 dims here correctly stay as the arm-only action
# [ee_pos(3), ee_rot(3), gripper(1)].
va_robocasa_no_base_cfg.used_action_channel_ids = list(range(0, 7))
inverse_used_action_channel_ids = [
    len(va_robocasa_no_base_cfg.used_action_channel_ids)
] * va_robocasa_no_base_cfg.action_dim
for i, j in enumerate(va_robocasa_no_base_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_robocasa_no_base_cfg.inverse_used_action_channel_ids = (
    inverse_used_action_channel_ids
)

q01 = va_robocasa_no_base_cfg.norm_stat["q01"]
q99 = va_robocasa_no_base_cfg.norm_stat["q99"]
arm_q01 = q01[:7]
arm_q99 = q99[:7]
va_robocasa_no_base_cfg.norm_stat = {
    "q01": [
        -1, -1, -1, -0.3428571429, -0.4742857143, -0.4171428571,
        -1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    ],
    "q99": [
        1, 1, 1, 0.4285714286, 0.3542857143, 0.42,
        1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    ],
}

va_robocasa_no_base_cfg.output_action_labels = [
    "eef_pos_x",
    "eef_pos_y",
    "eef_pos_z",
    "eef_rot_x",
    "eef_rot_y",
    "eef_rot_z",
    "gripper",
]
