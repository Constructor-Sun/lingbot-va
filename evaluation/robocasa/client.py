import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import imageio
import numpy as np
from tqdm import tqdm

CURRENT_FILE = Path(__file__).resolve()
LINGBOT_VA_ROOT = CURRENT_FILE.parents[2]
DEFAULT_ROBOCASA_ROOT = LINGBOT_VA_ROOT.parents[1] / "robocasa"
DEFAULT_TASK_REGISTRY_JSON = CURRENT_FILE.with_name("task_mobility_groups.json")
if str(LINGBOT_VA_ROOT) not in sys.path:
    sys.path.insert(0, str(LINGBOT_VA_ROOT))

try:
    from evaluation.robotwin.websocket_client_policy import WebsocketClientPolicy
    from evaluation.robocasa.adapter import (
        VA_IMAGE_KEYS,
        action_to_robocasa_dict,
        extract_prompt,
        extract_va_obs,
        sanitize_filename,
    )
except ImportError:
    sys.path.insert(0, str(LINGBOT_VA_ROOT / "evaluation" / "robotwin"))
    from websocket_client_policy import WebsocketClientPolicy
    from adapter import (
        VA_IMAGE_KEYS,
        action_to_robocasa_dict,
        extract_prompt,
        extract_va_obs,
        sanitize_filename,
    )


def _ensure_robocasa_importable(robocasa_root):
    robocasa_root = Path(robocasa_root).resolve()
    if str(robocasa_root) not in sys.path:
        sys.path.insert(0, str(robocasa_root))

    from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv

    return RoboCasaGymEnv


def write_json(data, path):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_env_names_from_registry(task_registry_json, task_group):
    registry_path = Path(task_registry_json)
    with registry_path.open("r", encoding="utf-8") as f:
        registry = json.load(f)

    groups = registry.get("groups", {})
    if task_group not in groups:
        available = ", ".join(sorted(groups))
        raise ValueError(
            f"Unknown task group '{task_group}' in {registry_path}. "
            f"Available groups: {available}"
        )

    env_names = groups[task_group]
    if not env_names:
        raise ValueError(
            f"Task group '{task_group}' in {registry_path} is empty."
        )
    return env_names


def resize_to_height(img, height):
    if img.shape[0] == height:
        return img
    width = int(round(img.shape[1] * height / img.shape[0]))
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def save_video(real_obs_list, save_path, fps=20, video_names=None):
    if not real_obs_list:
        print("No observation frames to save.")
        return

    if video_names is None:
        video_names = VA_IMAGE_KEYS

    base_h = real_obs_list[0][video_names[0]].shape[0]
    frames = []
    for obs in real_obs_list:
        row = np.hstack(
            [resize_to_height(obs[name], base_h) for name in video_names]
        ).astype(np.uint8)
        frames.append(row)

    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True, parents=True)
    imageio.mimsave(save_path, frames, fps=fps)
    print(f"Saved video to: {save_path}")


def construct_single_env(env_cls, env_name, env_kwargs, max_retries=5):
    last_error = None
    for retry_i in range(max_retries):
        try:
            return env_cls(env_name=env_name, **env_kwargs)
        except Exception as e:
            last_error = e
            print(f"Construct env failed ({retry_i + 1}/{max_retries}): {e}")
            time.sleep(5)
    raise RuntimeError(f"Failed to construct RoboCasa env {env_name}") from last_error


def step_env(env, action_step):
    action_dict = action_to_robocasa_dict(action_step)
    obs, reward, terminated, truncated, info = env.step(action_dict)
    done = bool(terminated or truncated or info.get("success", False))
    success = bool(info.get("success", reward > 0))
    return obs, done, success


def run_one(
    model,
    env_cls,
    env_name,
    out_dir,
    episode_idx,
    env_kwargs,
    max_steps,
    fps,
):
    env = construct_single_env(env_cls, env_name, env_kwargs)
    obs, _ = env.reset(seed=episode_idx)
    prompt = extract_prompt(obs)
    first_obs = extract_va_obs(obs)

    model.infer(dict(reset=True, prompt=prompt))

    full_obs_list = [first_obs]
    success = False
    first = True
    step_count = 0

    while step_count < max_steps:
        ret = model.infer(dict(obs=first_obs, prompt=prompt))
        action = ret["action"]

        key_frame_list = []
        assert action.shape[2] % 4 == 0
        action_per_frame = action.shape[2] // 4
        start_idx = 1 if first else 0

        for i in range(start_idx, action.shape[1]):
            for j in range(action.shape[2]):
                obs, done, success = step_env(env, action[:, i, j])
                step_count += 1

                if (j + 1) % action_per_frame == 0:
                    va_obs = extract_va_obs(obs)
                    full_obs_list.append(va_obs)
                    key_frame_list.append(va_obs)

                if done or step_count >= max_steps:
                    break

            if done or step_count >= max_steps:
                break

        first = False

        if success or step_count >= max_steps:
            break
        if key_frame_list:
            model.infer(
                dict(
                    obs=key_frame_list,
                    compute_kv_cache=True,
                    imagine=False,
                    state=action,
                )
            )

    video_name = f"{episode_idx}_{sanitize_filename(prompt)}_{success}.mp4"
    video_path = Path(out_dir) / env_name / video_name
    save_video(full_obs_list, video_path, fps=fps)

    env.close()
    return success, step_count, prompt


def run(
    env_names,
    port,
    out_dir,
    test_num,
    robocasa_root,
    split="target",
    camera_heights=256,
    camera_widths=256,
    max_steps=800,
    fps=20,
):
    env_cls = _ensure_robocasa_importable(robocasa_root)
    model = WebsocketClientPolicy(port=port)

    env_kwargs = {
        "split": split,
        "camera_heights": camera_heights,
        "camera_widths": camera_widths,
        "enable_render": True,
    }

    for env_name in tqdm(env_names, total=len(env_names)):
        succ_num = 0.0
        for episode_idx in tqdm(range(test_num), total=test_num):
            success, step_count, prompt = run_one(
                model=model,
                env_cls=env_cls,
                env_name=env_name,
                out_dir=out_dir,
                episode_idx=episode_idx,
                env_kwargs=env_kwargs,
                max_steps=max_steps,
                fps=fps,
            )
            succ_num += float(success)
            total_num = episode_idx + 1.0
            succ_rate = succ_num / total_num
            print(
                f"{env_name} | success rate: {succ_rate:.4f}, "
                f"success num: {succ_num}, total num: {total_num}, "
                f"steps: {step_count}, prompt: {prompt}"
            )
            write_json(
                {
                    "env_name": env_name,
                    "succ_num": succ_num,
                    "total_num": total_num,
                    "succ_rate": succ_rate,
                },
                Path(out_dir) / f"{env_name}.json",
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-names",
        type=str,
        nargs="+",
        help="RoboCasa task names, e.g. PnPCounterToCab.",
    )
    parser.add_argument(
        "--task-registry-json",
        type=str,
        default=str(DEFAULT_TASK_REGISTRY_JSON),
        help="Path to RoboCasa task grouping JSON.",
    )
    parser.add_argument(
        "--task-group",
        type=str,
        default=None,
        help="Task group name inside --task-registry-json. Used when --env-names is omitted.",
    )
    parser.add_argument("--port", type=int, default=23908)
    parser.add_argument("--test-num", type=int, default=50)
    parser.add_argument("--out-dir", type=str, default="outputs/robocasa")
    parser.add_argument("--robocasa-root", type=str, default=str(DEFAULT_ROBOCASA_ROOT))
    parser.add_argument(
        "--split",
        type=str,
        default="target",
        choices=["target", "pretrain", "all"],
    )
    parser.add_argument("--camera-heights", type=int, default=256)
    parser.add_argument("--camera-widths", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--fps", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.env_names:
        task_group = args.task_group
        if task_group is None:
            task_group = "guaranteed_no_base_motion"
        args.env_names = load_env_names_from_registry(
            args.task_registry_json, task_group
        )
        print(
            f"Loaded {len(args.env_names)} RoboCasa tasks from "
            f"{args.task_registry_json} group={task_group}"
        )
    run_kwargs = vars(args).copy()
    run_kwargs.pop("task_registry_json", None)
    run_kwargs.pop("task_group", None)
    run(**run_kwargs)
    print("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    main()
