# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

from easydict import EasyDict

from .shared_config import va_shared_cfg


def _get_robocasa_obs_cam_keys():
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


va_robocasa_cfg = EasyDict(__name__='Config: VA robocasa')
va_robocasa_cfg.update(va_shared_cfg)
va_shared_cfg.infer_mode = 'server'

va_robocasa_cfg.wan22_pretrained_model_name_or_path = os.path.expanduser(
    os.environ.get("MODEL_ROOT", "~/exp/checkpoints/lingbot-va-base")
)

va_robocasa_cfg.attn_window = 30
va_robocasa_cfg.frame_chunk_size = 4
va_robocasa_cfg.env_type = 'none'

va_robocasa_cfg.height = 256
va_robocasa_cfg.width = 256
va_robocasa_cfg.action_dim = 30
va_robocasa_cfg.action_per_frame = 4
va_robocasa_cfg.obs_cam_keys = _get_robocasa_obs_cam_keys()
va_robocasa_cfg.guidance_scale = 5
va_robocasa_cfg.action_guidance_scale = 1

va_robocasa_cfg.num_inference_steps = 20
va_robocasa_cfg.video_exec_step = -1
va_robocasa_cfg.action_num_inference_steps = 50

va_robocasa_cfg.snr_shift = 5.0
va_robocasa_cfg.action_snr_shift = 0.05

# LingBot canonical RoboCasa action layout used by training / inference:
#   0:3 ee_pos, 3:6 ee_rot, 6:7 gripper, 7:11 base_motion, 11:12 control_mode
# The raw RoboCasa LeRobot datasets may store these 12 action dims in a different
# order (e.g. base / control first); `compute_robocasa_norm_stat.py` reorders the
# dataset statistics back into this canonical layout before writing `norm_stat`.
va_robocasa_cfg.used_action_channel_ids = list(range(0, 12))
inverse_used_action_channel_ids = [
    len(va_robocasa_cfg.used_action_channel_ids)
] * va_robocasa_cfg.action_dim
for i, j in enumerate(va_robocasa_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_robocasa_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_robocasa_cfg.action_norm_method = 'quantiles'
# va_robocasa_cfg.norm_stat = {
#     "q01": [
#         -0.6589285731315613,
#         -0.84375,
#         -0.9375,
#         -0.12107142806053162,
#         -0.15964286029338837,
#         -0.26571428775787354,
#         -1.0
#     ] + [0.] * 23,
#     "q99": [
#         0.8999999761581421,
#         0.8544642925262451,
#         0.9375,
#         0.17142857611179352,
#         0.1842857152223587,
#         0.34392857551574707,
#         1.0
#     ] + [0.] * 23,
# }

va_robocasa_cfg.norm_stat = {
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
