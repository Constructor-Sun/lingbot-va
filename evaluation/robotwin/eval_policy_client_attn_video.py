"""
Attention-saliency client launcher.

Like eval_polict_client_openpi.py but the per-episode comparison video also
renders an attention-saliency overlay row returned by the attn-capture server
(wan_server_attn_capture.py).

Usage (same CLI as the original):
  python -m evaluation.robotwin.eval_policy_client_attn_video \
      --config policy/ACT/deploy_policy.yml \
      --overrides --task_name click_alarmclock ...
"""

import sys
import os
import subprocess
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
from pathlib import Path


def _find_counterfactual_root() -> Path:
    env = os.environ.get("COUNTERFACTUAL_ROOT")
    if env:
        return Path(env)
    start = Path(__file__).resolve().parent
    for ancestor in [start] + list(start.parents):
        if (ancestor / ".git").is_dir() and (ancestor / "robust_wam").is_dir():
            return ancestor
    raise RuntimeError("Cannot find counterfactual project root. Set COUNTERFACTUAL_ROOT.")


_CF_ROOT = _find_counterfactual_root()
robowin_root = _CF_ROOT / "external" / "RoboTwin"
if str(robowin_root) not in sys.path:
    sys.path.insert(0, str(robowin_root))

# Add robust_wam/src for mask utilities
_robust_wam_src = _CF_ROOT / "robust_wam" / "src"
if str(_robust_wam_src) not in sys.path:
    sys.path.insert(0, str(_robust_wam_src))


import os
os.chdir(robowin_root)

from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb
from evaluation.robotwin.geometry import euler2quat
import numpy as np

from description.utils.generate_episode_instructions import *
import traceback

import imageio
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import json
from pathlib import Path

from true_mask.mask_utils import (
    binary_mask_from_labels,
    resolve_mask_actor_ids,
    compute_saliency_mask_metrics,
    CAMERA_KEY_MAP as MASK_CAMERA_KEY_MAP,
)

from evaluation.robotwin.websocket_client_policy import WebsocketClientPolicy
from evaluation.robotwin.test_render import Sapien_TEST

# ---------------------------------------------------------------------------
# Inlined from motivation/attention_mask/attention_saliency.py
# (kept self-contained so the client does not need a cross-project import)
# ---------------------------------------------------------------------------
VIEW_SLICES = {
    'observation.images.cam_high':        (slice(4, 12), slice(0, 10)),
    'observation.images.cam_left_wrist':  (slice(0, 4),  slice(0, 5)),
    'observation.images.cam_right_wrist': (slice(0, 4),  slice(5, 10)),
}


def _normalize_map(m, eps=1e-8):
    """Min-max normalise a 2-D array to [0, 1]."""
    lo = float(m.min())
    hi = float(m.max())
    if hi - lo < eps:
        return np.zeros_like(m, dtype=np.float32)
    return ((m - lo) / (hi - lo)).astype(np.float32)


def _overlay_heatmap_on_image(img_rgb, heatmap01, alpha=0.5):
    """Blend a [0,1] heatmap onto an RGB image with JET colormap."""
    h, w = img_rgb.shape[:2]
    heat_resized = cv2.resize(heatmap01.astype(np.float32), (w, h),
                              interpolation=cv2.INTER_LINEAR)
    heat_u8 = (heat_resized * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    out = img_rgb.astype(np.float32) * (1 - alpha) + heat_color.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def save_token_cas_outputs(ret, obs, out_dir):
    cas = ret.get("cas")
    if not cas or "grid" not in cas:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "action.npy", ret["action"])
    np.save(out_dir / "cas_grid.npy", cas["grid"])
    for key in ("scores", "token_indices"):
        if key in cas:
            np.save(out_dir / f"cas_{key}.npy", cas[key])
    score_values = np.asarray(cas.get("scores", []), dtype=np.float32)
    meta = {
        "mode": cas.get("mode"),
        "per_token": cas.get("per_token"),
        "score_type": cas.get("score_type"),
        "n_scores": int(score_values.size),
    }
    if score_values.size:
        meta.update({
            "score_min": float(np.nanmin(score_values)),
            "score_max": float(np.nanmax(score_values)),
            "score_mean": float(np.nanmean(score_values)),
            "score_std": float(np.nanstd(score_values)),
        })
    for key in ("block_size", "n_blocks"):
        if key in cas:
            meta[key] = cas[key]
    with open(out_dir / "cas_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(out_dir / "cas_entries.json", "w", encoding="utf-8") as f:
        json.dump(cas.get("entries", []), f, indent=2)

    grid = np.asarray(cas["grid"], dtype=np.float32)
    for cam_key, (rs, cs) in VIEW_SLICES.items():
        cam_short = cam_key.split(".")[-1]
        img = np.asarray(obs[cam_key])
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        imageio.imwrite(out_dir / f"observation_{cam_short}.png", img)

        view = grid[:, rs, cs]
        valid = np.isfinite(view)
        count = valid.sum(axis=0)
        view2d = np.divide(np.where(valid, view, 0).sum(axis=0),
                           np.maximum(count, 1), where=count > 0)
        view2d[count == 0] = 0
        heat01 = _normalize_map(view2d)
        heat_u8 = (cv2.resize(heat01, (img.shape[1], img.shape[0])) * 255).astype(np.uint8)
        heat_rgb = cv2.cvtColor(cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        imageio.imwrite(out_dir / f"cas_map_{cam_short}.png", heat_rgb)
        imageio.imwrite(out_dir / f"cas_overlay_{cam_short}.png",
                        _overlay_heatmap_on_image(img, heat01))


def token_cas_request(args):
    if not args.get("token_cas", True):
        return {}
    req = {
        "cas_enabled": True,
        "cas_mode": args.get("cas_mode", "gradient"),
        "cas_per_token": True,
        "cas_mean_scope": args.get("cas_mean_scope", "same_camera"),
        "cas_show_progress": args.get("cas_show_progress", True),
    }
    if args.get("cas_max_tokens") is not None:
        req["cas_max_tokens"] = args["cas_max_tokens"]
    return req


# ---------------------------------------------------------------------------
# Original utility functions (unchanged)
# ---------------------------------------------------------------------------

def write_json(data: dict, fpath: Path) -> None:
    """Write data to a JSON file.

    Creates parent directories if they don't exist.

    Args:
        data (dict): The dictionary to write.
        fpath (Path): The path to the output JSON file.
    """
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def print_camera_perturbation_config(args):
    camera_perturbation = args.get("camera_perturbation", {})
    enabled = camera_perturbation.get("enabled", False)
    print("\033[96mCamera Perturbation:\033[0m " + str(enabled))
    if not enabled:
        return

    print(" - C1 Distance: " + str(camera_perturbation.get("c1_distance", False)))
    print(" - C1 Scale Range: " + str(camera_perturbation.get("c1_distance_scale_range", [0.85, 1.0])))
    print(" - C2 Spherical: " + str(camera_perturbation.get("c2_spherical", False)))
    print(" - C3 Orientation: " + str(camera_perturbation.get("c3_orientation", False)))
    print(" - C3 Yaw/Pitch/Roll Deg: " + str([
        camera_perturbation.get("c3_yaw_deg", 5),
        camera_perturbation.get("c3_pitch_deg", 5),
        camera_perturbation.get("c3_roll_deg", 5),
    ]))
    print(" - Anchor Mode: " + str(camera_perturbation.get("anchor_mode", "table_center")))


def add_title_bar(img, text, font_scale=0.8, thickness=2):
    """Add a black title bar with text above the image"""
    h, w, _ = img.shape
    bar_height = 40

    # Create black background bar
    title_bar = np.zeros((bar_height, w, 3), dtype=np.uint8)

    # Calculate text position to center it
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_x = (w - text_w) // 2
    text_y = (bar_height + text_h) // 2 - 5

    cv2.putText(title_bar, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return np.vstack([title_bar, img])


def quaternion_to_euler(quat):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw)
    quat: [rx, ry, rz, rw] format
    Return: [roll, pitch, yaw] (radians)
    """
    # scipy uses [x, y, z, w] format
    rotation = R.from_quat(quat)
    euler = rotation.as_euler('xyz', degrees=False)  # returns [roll, pitch, yaw]
    return euler


def visualize_action_step(action_history, step_idx, window=50):
    """
    Plot dual-arm action curves:
    Subplot 1: Left arm XYZ Position + Gripper
    Subplot 2: Left arm Euler angles (Roll, Pitch, Yaw) - converted from quaternion
    Subplot 3: Right arm XYZ Position + Gripper
    Subplot 4: Right arm Euler angles (Roll, Pitch, Yaw) - converted from quaternion

    Input data format: [left_x, left_y, left_z, left_rx, left_ry, left_rz, left_rw, left_gripper,
                   right_x, right_y, right_z, right_rx, right_ry, right_rz, right_rw, right_gripper]
    Total 16 dimensions
    """
    # Create four subplots, sharing the X-axis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 8), dpi=100, sharex=True)

    # 1. Determine slice range
    start = max(0, step_idx - window)
    end = step_idx + 1

    # 2. Get data subset
    history_subset = np.array(action_history)[start:end]

    # 3. Generate X-axis based on actual data length
    actual_len = len(history_subset)
    x_axis = range(start, start + actual_len)

    if actual_len > 0 and history_subset.shape[1] >= 16:
        # Convert quaternions to Euler angles
        left_euler = []
        right_euler = []

        for action in history_subset:
            # Left arm quaternion to Euler angles
            left_quat = action[3:7]  # [rx, ry, rz, rw]
            left_rpy = quaternion_to_euler(left_quat)
            left_euler.append(left_rpy)

            # Right arm quaternion to Euler angles
            right_quat = action[11:15]  # [rx, ry, rz, rw]
            right_rpy = quaternion_to_euler(right_quat)
            right_euler.append(right_rpy)

        left_euler = np.array(left_euler)
        right_euler = np.array(right_euler)

        # --- Left Arm ---
        # Subplot 1: Left Arm Translation (XYZ) + Gripper
        ax1.plot(x_axis, history_subset[:, 0], label='left_x', color='r', linewidth=1.5)
        ax1.plot(x_axis, history_subset[:, 1], label='left_y', color='g', linewidth=1.5)
        ax1.plot(x_axis, history_subset[:, 2], label='left_z', color='b', linewidth=1.5)
        ax1.plot(x_axis, history_subset[:, 7], label='left_grip', color='orange',
                 linestyle=':', linewidth=2, alpha=0.8)
        ax1.set_ylabel('Position (m)')
        ax1.legend(loc='upper right', fontsize='x-small', ncol=4)
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"Step {step_idx}: Left Arm Position & Gripper")

        # Subplot 2: Left Arm Euler Angles (Roll, Pitch, Yaw)
        ax2.plot(x_axis, left_euler[:, 0], label='left_roll', color='c', linewidth=1.5)
        ax2.plot(x_axis, left_euler[:, 1], label='left_pitch', color='m', linewidth=1.5)
        ax2.plot(x_axis, left_euler[:, 2], label='left_yaw', color='y', linewidth=1.5)
        ax2.set_ylabel('Rotation (rad)')
        ax2.legend(loc='upper right', fontsize='x-small', ncol=3)
        ax2.grid(True, alpha=0.3)
        ax2.set_title("Left Arm Rotation (RPY from Quaternion)")

        # --- Right Arm ---
        # Subplot 3: Right Arm Translation (XYZ) + Gripper
        ax3.plot(x_axis, history_subset[:, 8], label='right_x', color='r', linewidth=1.5, linestyle='--')
        ax3.plot(x_axis, history_subset[:, 9], label='right_y', color='g', linewidth=1.5, linestyle='--')
        ax3.plot(x_axis, history_subset[:, 10], label='right_z', color='b', linewidth=1.5, linestyle='--')
        ax3.plot(x_axis, history_subset[:, 15], label='right_grip', color='orange',
                 linestyle=':', linewidth=2, alpha=0.8)
        ax3.set_ylabel('Position (m)')
        ax3.legend(loc='upper right', fontsize='x-small', ncol=4)
        ax3.grid(True, alpha=0.3)
        ax3.set_title("Right Arm Position & Gripper")

        # Subplot 4: Right Arm Euler Angles (Roll, Pitch, Yaw)
        ax4.plot(x_axis, right_euler[:, 0], label='right_roll', color='c', linewidth=1.5, linestyle='--')
        ax4.plot(x_axis, right_euler[:, 1], label='right_pitch', color='m', linewidth=1.5, linestyle='--')
        ax4.plot(x_axis, right_euler[:, 2], label='right_yaw', color='y', linewidth=1.5, linestyle='--')
        ax4.set_ylabel('Rotation (rad)')
        ax4.legend(loc='upper right', fontsize='x-small', ncol=3)
        ax4.grid(True, alpha=0.3)
        ax4.set_title("Right Arm Rotation (RPY from Quaternion)")

    # Set X-axis display range to maintain sliding window effect
    ax1.set_xlim(max(0, step_idx - window), max(window, step_idx))
    ax3.set_xlabel('Step')
    ax4.set_xlabel('Step')

    plt.tight_layout()
    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.asarray(canvas.buffer_rgba())
    img = img[:, :, :3]

    # Convert to uint8
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    plt.close(fig)
    return img


# ---------------------------------------------------------------------------
# New video writer: saves observations + saliency overlay side-by-side
# ---------------------------------------------------------------------------

def save_comparison_video_attn(real_obs_list, chunk_obs_registry, all_saliency_chunks,
                                imagined_video=None, save_path=None, fps=15):
    """Save a combined episode video with generated video and saliency rows.

    Parameters
    ----------
    real_obs_list : list of dict
        Observation dicts keyed by ``observation.images.cam_high`` etc.
    chunk_obs_registry : list of (obs_idx, chunk_idx, frame_idx)
        Maps every observation index to the (chunk, latent frame) that produced it.
    all_saliency_chunks : list of dict
        Per-chunk saliency data from the server:
        ``{'chunk': int, 'saliency_maps': {target: {'grid': ndarray (T,12,10), ...}}}``
    imagined_video : list of ndarray, optional
        Per-chunk generated video arrays returned by the server.
    save_path : str
    fps : int
    """
    if not real_obs_list:
        return

    include_generated = bool(imagined_video)
    generated_by_chunk_frame = {}
    if include_generated:
        n_imagined = sum(len(video) for video in imagined_video)
        for chunk_idx, video in enumerate(imagined_video):
            for frame_idx in range(len(video)):
                generated_by_chunk_frame[(chunk_idx, frame_idx)] = video[frame_idx]
    else:
        n_imagined = 0

    # --- Build per-observation saliency lookup --------------------------------
    # saliency_by_chunk_frame[(chunk, frame_idx)] -> {target: {'grid_2d': ndarray(12,10)}}
    saliency_by_chunk_frame = {}
    for chunk_data in all_saliency_chunks:
        c = chunk_data['chunk']
        for tgt, payload in chunk_data['saliency_maps'].items():
            grid = payload['grid']  # (T, 12, 10)
            if grid is None:
                continue
            for fi in range(grid.shape[0]):
                key = (c, fi)
                entry = saliency_by_chunk_frame.setdefault(key, {})
                entry[tgt] = {'grid_2d': grid[fi]}

    # frame_saliency[obs_idx] = {target: {'grid_2d': ndarray(12,10)}}
    frame_saliency = {}
    frame_generated = {}
    for (obs_idx, chunk_idx, frame_idx) in chunk_obs_registry:
        key = (chunk_idx, frame_idx)
        if key in saliency_by_chunk_frame:
            frame_saliency[obs_idx] = saliency_by_chunk_frame[key]
        if key in generated_by_chunk_frame:
            frame_generated[obs_idx] = generated_by_chunk_frame[key]

    # --- Render frames --------------------------------------------------------
    n_frames = len(real_obs_list)
    n_with_sal = len(frame_saliency)
    print(f"Saving attn video: {n_frames} real frames, "
          f"{n_imagined} generated frames, "
          f"{n_with_sal} with saliency overlay ...")

    final_frames = []

    for i in range(n_frames):
        obs = real_obs_list[i]
        cam_high = obs["observation.images.cam_high"]
        cam_left = obs["observation.images.cam_left_wrist"]
        cam_right = obs["observation.images.cam_right_wrist"]

        base_h = cam_high.shape[0]

        def resize_h(img, h):
            if img.shape[0] != h:
                w = int(img.shape[1] * h / img.shape[0])
                img = cv2.resize(img, (w, h))
            img = np.ascontiguousarray(img)
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            return img

        def generated_tshape_to_row(img):
            top_h = img.shape[0] // 3
            split_w = img.shape[1] // 2
            gen_left = img[:top_h, :split_w]
            gen_right = img[:top_h, split_w:]
            gen_high = img[top_h:]
            return np.hstack([
                resize_h(gen_high, base_h),
                resize_h(gen_left, base_h),
                resize_h(gen_right, base_h),
            ])

        # --- Top row: original observations (unchanged) ---
        row_real = np.hstack([
            resize_h(cam_high, base_h),
            resize_h(cam_left, base_h),
            resize_h(cam_right, base_h),
        ])
        row_real = np.ascontiguousarray(row_real)
        row_real = add_title_bar(row_real, "Real Observation (High / Left / Right)")

        target_width = row_real.shape[1]

        # --- Middle row: generated / imagined video ---
        if include_generated:
            if i in frame_generated:
                img_frame = frame_generated[i]
                if img_frame.dtype != np.uint8 and img_frame.max() <= 1.0001:
                    img_frame = (img_frame * 255).astype(np.uint8)
                elif img_frame.dtype != np.uint8:
                    img_frame = img_frame.astype(np.uint8)

                row_imagined = generated_tshape_to_row(img_frame)
                if row_imagined.shape[1] != target_width:
                    row_imagined = cv2.resize(row_imagined, (target_width, base_h))
            else:
                row_imagined = np.zeros((base_h, target_width, 3), dtype=np.uint8)
                cv2.putText(row_imagined, "No generated video",
                            (target_width // 2 - 140, max(30, base_h // 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)

            row_imagined = np.ascontiguousarray(row_imagined)
            row_imagined = add_title_bar(row_imagined, "Generated Video Stream")

        # --- Bottom row: saliency overlay on observations ---
        if i in frame_saliency:
            sal_info = frame_saliency[i]
            target_name = list(sal_info.keys())[0]  # first available target
            grid_2d = sal_info[target_name]['grid_2d']  # (12, 10)

            overlays = []
            for cam_name, (rs, cs) in VIEW_SLICES.items():
                cam_img = obs[cam_name]
                view_map = grid_2d[rs, cs]
                overlay = _overlay_heatmap_on_image(cam_img, _normalize_map(view_map))
                overlays.append(resize_h(overlay, base_h))

            row_sal = np.hstack(overlays)
            row_sal = np.ascontiguousarray(row_sal)
            row_sal = add_title_bar(row_sal, f"Attention Saliency ({target_name})")
        else:
            row_sal = np.zeros((base_h, target_width, 3), dtype=np.uint8)
            cv2.putText(row_sal, "No saliency data",
                        (target_width // 2 - 100, max(30, base_h // 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
            row_sal = add_title_bar(row_sal, "Attention Saliency")

        row_sal = np.ascontiguousarray(row_sal)
        rows = [row_real]
        if include_generated:
            rows.append(row_imagined)
        rows.append(row_sal)
        full_frame = np.vstack(rows)
        full_frame = np.ascontiguousarray(full_frame)
        final_frames.append(full_frame)

    imageio.mimsave(save_path, final_frames, fps=fps)
    print(f"Combined video (with saliency) saved to: {save_path}")


# ---------------------------------------------------------------------------
# Original utility functions (unchanged) — continued
# ---------------------------------------------------------------------------

def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e


def get_camera_config(camera_type):
    camera_config_path = os.path.join(robowin_root, "task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    save_root = usr_args["save_root"]
    policy_name = usr_args["policy_name"]
    video_guidance_scale = usr_args["video_guidance_scale"]
    action_guidance_scale = usr_args["action_guidance_scale"]
    instruction_type = 'seen'
    save_dir = None
    video_save_dir = None
    video_size = None

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting
    args["save_root"] = save_root
    for key in ("token_cas", "cas_mode", "cas_mean_scope", "cas_max_tokens",
                "cas_show_progress", "cas_first_chunk_only", "return_video"):
        args[key] = usr_args.get(key)

    # Enable actor segmentation for mask export
    if args.get("data_type") is None:
        args["data_type"] = {}
    args["data_type"]["actor_segmentation"] = True

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))
    print_camera_perturbation_config(args)

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    st_seed = 10000 * (1 + seed)
    suc_nums = []
    test_num = usr_args["test_num"]

    model = WebsocketClientPolicy(port=usr_args['port'])

    st_seed, suc_num = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   test_num=test_num,
                                   video_size=video_size,
                                   instruction_type=instruction_type,
                                   save_visualization=True,
                                   video_guidance_scale=video_guidance_scale,
                                   action_guidance_scale=action_guidance_scale)
    suc_nums.append(suc_num)

    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    print(f"Data has been saved to {file_path}")


def format_obs(observation, prompt):
    return {
        "observation.images.cam_high": observation["observation"]["head_camera"]["rgb"],  # H,W,3
        "observation.images.cam_left_wrist": observation["observation"]["left_camera"]["rgb"],
        "observation.images.cam_right_wrist": observation["observation"]["right_camera"]["rgb"],
        "observation.state": observation["joint_action"]["vector"],
        "task": prompt,
    }


def add_eef_pose(new_pose, init_pose):
    new_pose_R = R.from_quat(new_pose[3:7][None])
    init_pose_R = R.from_quat(init_pose[3:7][None])
    out_rot = (init_pose_R * new_pose_R).as_quat().reshape(-1)
    out_trans = new_pose[:3] + init_pose[:3]
    return np.concatenate([out_trans, out_rot, new_pose[7:8]])


def add_init_pose(new_pose, init_pose):
    left_pose = add_eef_pose(new_pose[:8], init_pose[:8])
    right_pose = add_eef_pose(new_pose[8:], init_pose[8:])
    return np.concatenate([left_pose, right_pose])


# ---------------------------------------------------------------------------
# eval_policy — same as original EXCEPT:
#   1. Tracks chunk_obs_registry during action execution
#   2. Captures attn_saliency from compute_kv_cache responses
#   3. Calls save_comparison_video_attn instead of save_comparison_video
# ---------------------------------------------------------------------------

def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None,
                save_visualization=False,
                video_guidance_scale=5.0,
                action_guidance_scale=5.0):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []

    now_seed = st_seed
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True

    while succ_seed < test_num:
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                print(f"error occurs ! {e}")
                traceback.print_exc()
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)

        # Resolve target actor IDs for mask generation (object + robot)
        try:
            target_actor_ids = resolve_mask_actor_ids(
                TASK_ENV, args["task_name"],
                object_attr=None, mask_target="object_robot",
            )
            print(f"[mask] resolved {len(target_actor_ids)} target actor IDs for task={args['task_name']}")
        except Exception as e:
            print(f"[mask] WARNING: could not resolve target_actor_ids: {e}")
            target_actor_ids = None

        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        instruction = TASK_ENV.instruction_rng.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False

        prompt = TASK_ENV.get_instruction()
        task_cfg = args.get("task_config", "default")
        ep_name = f"{task_cfg}_{now_seed}"
        ret = model.infer(dict(reset=True, prompt=prompt,
                               episode_name=ep_name,
                               save_visualization=save_visualization))

        first = True
        full_obs_list = []
        raw_obs_list = []                           # raw observations (with segmentation labels)
        gen_video_list = []
        full_action_history = []
        all_saliency_chunks = []                    # accumulated server saliency data
        chunk_obs_registry = []                     # (obs_idx, chunk_idx, frame_idx)
        chunk_idx = 0

        initial_obs = TASK_ENV.get_obs()
        raw_obs_list.append(initial_obs)            # keep raw obs for mask extraction
        inint_eef_pose = initial_obs['endpose']['left_endpose'] + \
            [initial_obs['endpose']['left_gripper']] + \
            initial_obs['endpose']['right_endpose'] + \
            [initial_obs['endpose']['right_gripper']]
        inint_eef_pose = np.array(inint_eef_pose, dtype=np.float64)
        initial_formatted_obs = format_obs(initial_obs, prompt)
        full_obs_list.append(initial_formatted_obs)
        # Observation index 0 belongs to chunk 0, latent frame 0
        chunk_obs_registry.append((0, 0, 0))

        first_obs = None
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            if first:
                observation = TASK_ENV.get_obs()
                first_obs = format_obs(observation, prompt)

            action_req = dict(obs=first_obs, prompt=prompt,
                              save_visualization=save_visualization,
                              video_guidance_scale=video_guidance_scale,
                              action_guidance_scale=action_guidance_scale,
                              return_video=args.get("return_video", False))
            action_req.update(token_cas_request(args))
            cas_obs = full_obs_list[-1]
            ret = model.infer(action_req)
            save_token_cas_outputs(
                ret, cas_obs,
                Path(args['save_root']) / ep_name / "cas" / f"chunk_{chunk_idx:03d}",
            )
            if args.get("cas_first_chunk_only", False):
                break
            action = ret['action']
            if 'video' in ret:
                imagined_video = ret['video']
                gen_video_list.append(imagined_video)
            key_frame_list = []

            assert action.shape[2] % 4 == 0
            action_per_frame = action.shape[2] // 4

            start_idx = 1 if first else 0
            for i in range(start_idx, action.shape[1]):
                for j in range(action.shape[2]):
                    raw_action_step = action[:, i, j].flatten()
                    full_action_history.append(raw_action_step)

                    ee_action = action[:, i, j]
                    if action.shape[0] == 14:
                        ee_action = np.concatenate([
                            ee_action[:3],
                            euler2quat(ee_action[3], ee_action[4], ee_action[5]),
                            ee_action[6:10],
                            euler2quat(ee_action[10], ee_action[11], ee_action[12]),
                            ee_action[13:14]
                        ])
                    elif action.shape[0] == 16:
                        ee_action = add_init_pose(ee_action, inint_eef_pose)
                        ee_action = np.concatenate([
                            ee_action[:3],
                            ee_action[3:7] / np.linalg.norm(ee_action[3:7]),
                            ee_action[7:11],
                            ee_action[11:15] / np.linalg.norm(ee_action[11:15]),
                            ee_action[15:16]
                        ])
                    else:
                        raise NotImplementedError
                    TASK_ENV.take_action(ee_action, action_type='ee')

                    if (j + 1) % action_per_frame == 0:
                        raw_obs = TASK_ENV.get_obs()
                        obs = format_obs(raw_obs, prompt)
                        obs_idx = len(full_obs_list)
                        full_obs_list.append(obs)
                        raw_obs_list.append(raw_obs)           # keep raw obs for mask extraction
                        key_frame_list.append(obs)
                        chunk_obs_registry.append(
                            (obs_idx, chunk_idx, i)
                        )

            first = False

            # Capture saliency data returned by the attn-capture server
            ret_kv = model.infer(dict(obs=key_frame_list, compute_kv_cache=True,
                                      imagine=False,
                                      save_visualization=save_visualization,
                                      state=action))
            if 'attn_saliency' in ret_kv:                      # NEW
                all_saliency_chunks.extend(ret_kv['attn_saliency'])  # NEW

            chunk_idx += 1                                      # NEW

            if TASK_ENV.eval_success:
                succ = True
                break

        vis_dir = Path(args['save_root']) / f'stseed-{st_seed}' / 'visualization' / task_name
        vis_dir.mkdir(parents=True, exist_ok=True)
        video_name = f"{TASK_ENV.test_num}_{prompt.replace(' ', '_')}_{succ}.mp4"
        out_img_file = vis_dir / video_name
        save_comparison_video_attn(
            real_obs_list=full_obs_list,
            chunk_obs_registry=chunk_obs_registry,
            all_saliency_chunks=all_saliency_chunks,
            imagined_video=gen_video_list,
            save_path=str(out_img_file),
            fps=15,
        )

        # --- Save masks and compute saliency-mask metrics ---
        demo_dir = Path(args['save_root']) / ep_name
        demo_dir.mkdir(parents=True, exist_ok=True)

        if target_actor_ids is not None and raw_obs_list:
            print(f"[mask] generating masks: {len(raw_obs_list)} raw obs, "
                  f"{len(target_actor_ids)} target actor IDs → {demo_dir}")
            # Generate mask frames from raw observations
            mask_frames_by_camera = {cam: [] for cam in MASK_CAMERA_KEY_MAP}
            seg_found = False
            for raw_obs in raw_obs_list:
                obs_data = raw_obs.get("observation", {})
                for rob_cam in MASK_CAMERA_KEY_MAP:
                    cam_data = obs_data.get(rob_cam, {})
                    seg_labels = cam_data.get("actor_segmentation_labels")
                    if seg_labels is not None:
                        seg_found = True
                        mask = binary_mask_from_labels(seg_labels, target_actor_ids)
                        mask_frames_by_camera[rob_cam].append(mask)

            if not seg_found:
                print("[mask] WARNING: no actor_segmentation_labels found in observations. "
                      "Check that data_type.actor_segmentation=True was passed to setup_demo.")

            # Save mask videos (one per camera)
            for rob_cam, cam_key in MASK_CAMERA_KEY_MAP.items():
                masks = mask_frames_by_camera.get(rob_cam, [])
                if masks:
                    cam_short = cam_key.split(".")[-1]
                    mask_video_path = demo_dir / f"mask_{cam_short}.mp4"
                    try:
                        imageio.mimsave(str(mask_video_path), masks, fps=15)
                        print(f"[mask] saved {len(masks)} frames → {mask_video_path}")
                    except Exception as e:
                        print(f"[mask] WARNING: failed to save mask video {cam_short}: {e}")

            # Compute saliency-mask alignment metrics per chunk
            all_metrics = []
            for chunk_data in all_saliency_chunks:
                chunk_idx_sal = chunk_data['chunk']
                for tgt, payload in chunk_data['saliency_maps'].items():
                    grid = payload.get('grid')
                    if grid is None:
                        continue

                    # Build parallel obs→frame + per-camera mask lists for this chunk
                    cam_keys = list(MASK_CAMERA_KEY_MAP.values())  # ["observation.images.cam_high", ...]
                    chunk_obs_to_frame = []
                    chunk_masks_by_camera = {ck: [] for ck in cam_keys}
                    for (obs_idx, ch_idx, fr_idx) in chunk_obs_registry:
                        if ch_idx != chunk_idx_sal:
                            continue
                        if obs_idx >= len(raw_obs_list):
                            continue
                        raw = raw_obs_list[obs_idx]
                        obs_data = raw.get("observation", {})

                        # Build masks for all three cameras
                        masks_this_obs = {}
                        all_valid = True
                        for rob_cam, cam_key in MASK_CAMERA_KEY_MAP.items():
                            cam_data = obs_data.get(rob_cam, {})
                            seg = cam_data.get("actor_segmentation_labels")
                            if seg is None or target_actor_ids is None:
                                all_valid = False
                                break
                            masks_this_obs[cam_key] = binary_mask_from_labels(seg, target_actor_ids)

                        if not all_valid:
                            continue
                        chunk_obs_to_frame.append((obs_idx, fr_idx))
                        for ck in cam_keys:
                            chunk_masks_by_camera[ck].append(masks_this_obs[ck])

                    if chunk_masks_by_camera and chunk_obs_to_frame:
                        try:
                            metrics = compute_saliency_mask_metrics(
                                grid, chunk_masks_by_camera, chunk_obs_to_frame)
                            metrics['chunk'] = int(chunk_idx_sal)
                            metrics['target'] = tgt
                            all_metrics.append(metrics)
                        except Exception as e:
                            print(f"[metrics] WARNING chunk {chunk_idx_sal}/{tgt}: {e}")

            # Save metrics JSON
            if all_metrics:
                metrics_path = demo_dir / "saliency_metrics.json"
                with open(metrics_path, "w") as f:
                    json.dump(all_metrics, f, indent=2)
                print(f"[metrics] saved → {metrics_path}")

        else:
            print(f"[mask] SKIP mask generation: "
                  f"target_actor_ids={'set' if target_actor_ids else 'None'} "
                  f"raw_obs_list={'empty' if not raw_obs_list else len(raw_obs_list)}")

        # Rename episode directory to include success/failure status
        demo_dir_no_suffix = Path(args['save_root']) / ep_name
        demo_dir_final = Path(args['save_root']) / f"{ep_name}_{succ}"
        if demo_dir_no_suffix.exists():
            try:
                # If final name already exists from a previous run, remove it first
                if demo_dir_final.exists():
                    import shutil
                    shutil.rmtree(demo_dir_final)
                demo_dir_no_suffix.rename(demo_dir_final)
                print(f"[mask] renamed {demo_dir_no_suffix.name} → {demo_dir_final.name}")
            except OSError as e:
                print(f"[mask] WARNING: could not rename directory: {e}")

        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        save_dir = Path(args['save_root']) / f'stseed-{st_seed}' / 'metrics' / task_name
        save_dir.mkdir(parents=True, exist_ok=True)
        out_json_file = save_dir / 'res.json'
        write_json({
            "succ_num": float(TASK_ENV.suc),
            "total_num": float(TASK_ENV.test_num),
            "succ_rate": float(TASK_ENV.suc / TASK_ENV.test_num),
        }, out_json_file)

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | "
            f"\033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => "
            f"\033[95m{round(TASK_ENV.suc / TASK_ENV.test_num * 100, 1)}%\033[0m, "
            f"current seed: \033[90m{now_seed}\033[0m\n"
        )
        now_seed += 1

    return now_seed, TASK_ENV.suc


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="policy/ACT/deploy_policy.yml")
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--task_config", type=str, default="demo_clean")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=29169, help='remote policy socket port.')
    parser.add_argument("--save_root", type=str, default="results/token_cas")
    parser.add_argument("--video_guidance_scale", type=float, default=5.0)
    parser.add_argument("--action_guidance_scale", type=float, default=1.0)
    parser.add_argument("--test_num", type=int, default=1)
    parser.add_argument("--return_video", action="store_true",
                        help="Decode and return generated video frames for visualization.")
    parser.add_argument("--no_token_cas", dest="token_cas", action="store_false")
    parser.add_argument("--cas_mode", choices=["gradient", "block", "zero", "mean"], default="gradient")
    parser.add_argument("--cas_mean_scope", choices=["same_camera", "same_frame", "global"],
                        default="same_camera")
    parser.add_argument("--cas_max_tokens", type=int, default=None)
    parser.add_argument("--no_cas_progress", dest="cas_show_progress", action="store_false")
    parser.add_argument("--cas_first_chunk_only", action="store_true")
    parser.set_defaults(token_cas=True, cas_show_progress=True, cas_first_chunk_only=False)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    for k, v in vars(args).items():
        if k != "overrides" and v is not None:
            config[k] = v
    if args.overrides:
        config.update(parse_override_pairs(args.overrides))
    config["task_config"] = str(config["task_config"]).removesuffix(".yml")
    config.setdefault("policy_name", "ACT")
    config.setdefault("train_config_name", "0")
    config.setdefault("model_name", "0")
    if config.get("ckpt_setting") is None:
        config["ckpt_setting"] = str(config.get("model_name", "0"))
    if not config.get("task_name"):
        raise ValueError("Please set --task_name, e.g. --task_name click_alarmclock")

    return config


if __name__ == "__main__":

    Sapien_TEST()
    # Sapien_TEST() globally sets camera shader to "rt" and enables ray
    # tracing, which breaks actor segmentation.  Revert both before any
    # RoboTwin scene is created.
    import sapien.render
    sapien.render.set_camera_shader_dir("default")
    sapien.render.set_ray_tracing_samples_per_pixel(0)
    print("[mask] camera shader reset to rasterization, RT disabled")

    usr_args = parse_args_and_config()
    main(usr_args)
