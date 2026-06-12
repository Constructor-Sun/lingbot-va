import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[3]
_CF_ROOT = REPO_ROOT.parents[1]  # counterfactual root
_EXP_ROOT = _CF_ROOT.parent  # /data1/liu/exp (sibling of counterfactual)
_CHECKPOINT_ROOT = _CF_ROOT / "checkpoints"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_va.modules.utils import (  # noqa: E402
    load_text_encoder,
    load_tokenizer,
    load_vae,
)


DEFAULT_DATASET_ROOT = os.environ.get("ROBOCASA_DATASET_PATH")
if DEFAULT_DATASET_ROOT is None:
    DEFAULT_DATASET_ROOT = _EXP_ROOT / "robocasa" / "datasets" / "training_no_base" / "atomic"
else:
    DEFAULT_DATASET_ROOT = Path(DEFAULT_DATASET_ROOT)

DEFAULT_CHECKPOINT_ROOT = os.environ.get("LINGBOT_CHECKPOINT_ROOT")
if DEFAULT_CHECKPOINT_ROOT is None:
    DEFAULT_CHECKPOINT_ROOT = _CHECKPOINT_ROOT / "lingbot-va-base"
else:
    DEFAULT_CHECKPOINT_ROOT = Path(DEFAULT_CHECKPOINT_ROOT)
DEFAULT_DEVICE = "cuda"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_TARGET_FPS = 10
DEFAULT_MAX_TEXT_LENGTH = 512
CAMERA_PRESET_KEYS = {
    "robocasa": [
        "observation.images.robot0_agentview_left",
        "observation.images.robot0_agentview_right",
        "observation.images.robot0_eye_in_hand",
    ],
    "robotwin": [
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ],
}
DEFAULT_RESOLUTION_BY_PRESET = {
    "robocasa": (256, 256),
    "robotwin": (256, 320),
}


def resolve_frame_size_for_video_key(
    camera_preset: str,
    video_key: str,
    height: int,
    width: int,
) -> tuple[int, int]:
    if camera_preset == "robotwin" and "wrist" in video_key:
        return height // 2, width // 2
    return height, width


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Encode LeRobot videos into Wan2.2 VAE latents for LingBot-VA training."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root like robocasa/datasets/training_no_base/atomic.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
        help="Checkpoint root like checkpoints/lingbot-va-base.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="*",
        default=None,
        help="Optional task names. If omitted, all task folders under dataset-root are processed.",
    )
    parser.add_argument(
        "--camera-preset",
        type=str,
        default="robocasa",
        choices=sorted(CAMERA_PRESET_KEYS),
        help="Named camera-key preset to read from meta/info.json features.",
    )
    parser.add_argument(
        "--video-keys",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional explicit video keys. Overrides --camera-preset when provided, "
            "for example observation.images.cam_high observation.images.cam_left_wrist observation.images.cam_right_wrist."
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_TARGET_FPS,
        help="Target fps for frame sampling before VAE encoding.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Resize height before VAE encoding. Defaults depend on --camera-preset.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Resize width before VAE encoding. Defaults depend on --camera-preset.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="Torch device for VAE/text encoder inference.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=DEFAULT_DTYPE,
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype for VAE/text encoder inference.",
    )
    parser.add_argument(
        "--max-text-length",
        type=int,
        default=DEFAULT_MAX_TEXT_LENGTH,
        help="Tokenizer max sequence length for text embeddings.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing latent .pth files.",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated CUDA device ids. Defaults to all visible GPUs.",
    )
    return parser.parse_args()


def resolve_target_resolution(
    camera_preset: str,
    height: int | None,
    width: int | None,
) -> tuple[int, int]:
    default_height, default_width = DEFAULT_RESOLUTION_BY_PRESET[camera_preset]
    return (
        int(default_height if height is None else height),
        int(default_width if width is None else width),
    )


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def torch_dtype_from_name(name: str):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


class WanLatentEncoder:
    def __init__(
        self,
        checkpoint_root: Path,
        device: str,
        dtype: torch.dtype,
        max_text_length: int,
        height: int,
        width: int,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_text_length = max_text_length
        self.height = height
        self.width = width

        self.vae = load_vae(
            checkpoint_root / "vae",
            torch_dtype=dtype,
            torch_device=self.device,
        )
        self.vae.eval()

        self.text_encoder = load_text_encoder(
            checkpoint_root / "text_encoder",
            torch_dtype=dtype,
            torch_device=self.device,
        )
        self.text_encoder.eval()
        self.tokenizer = load_tokenizer(checkpoint_root / "tokenizer")
        self.text_cache = {}

    @torch.inference_mode()
    def encode_text(self, text: str):
        if text in self.text_cache:
            return self.text_cache[text]

        text_inputs = self.tokenizer(
            [text],
            padding="max_length",
            max_length=self.max_text_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self.device)
        mask = text_inputs.attention_mask.to(self.device)
        seq_len = int(mask.gt(0).sum(dim=1).item())

        prompt_embeds = self.text_encoder(input_ids, mask).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=self.device)
        prompt_embeds = prompt_embeds[0, :seq_len]
        if seq_len < self.max_text_length:
            padding = prompt_embeds.new_zeros(
                (self.max_text_length - seq_len, prompt_embeds.shape[1])
            )
            prompt_embeds = torch.cat([prompt_embeds, padding], dim=0)

        prompt_embeds = prompt_embeds.to(torch.bfloat16).cpu()
        self.text_cache[text] = prompt_embeds
        return prompt_embeds

    @torch.inference_mode()
    def encode_video_frames(
        self,
        frames_uint8: np.ndarray,
        height: int | None = None,
        width: int | None = None,
    ):
        if frames_uint8.ndim != 4 or frames_uint8.shape[-1] != 3:
            raise ValueError(f"Expected frames as [T, H, W, 3], got {frames_uint8.shape}")

        target_height = int(self.height if height is None else height)
        target_width = int(self.width if width is None else width)
        video_height = target_height
        video_width = target_width

        frames = torch.from_numpy(frames_uint8).permute(0, 3, 1, 2).float()
        frames = F.interpolate(
            frames,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
        frames = frames.permute(1, 0, 2, 3).unsqueeze(0)
        frames = frames / 255.0 * 2.0 - 1.0
        frames = frames.to(device=self.device, dtype=self.dtype)

        posterior = self.vae.encode(frames).latent_dist
        mu = posterior.mean
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=mu.device)
        latents_std = torch.tensor(self.vae.config.latents_std, device=mu.device)
        mu = ((mu.float() - latents_mean.view(1, -1, 1, 1, 1)) * (1.0 / latents_std).view(1, -1, 1, 1, 1)).to(mu)
        mu = mu[0]

        latent_num_frames = int(mu.shape[1])
        latent_height = int(mu.shape[2])
        latent_width = int(mu.shape[3])
        latent = mu.permute(1, 2, 3, 0).reshape(-1, mu.shape[0]).to(torch.bfloat16).cpu()
        return {
            "latent": latent,
            "latent_num_frames": latent_num_frames,
            "latent_height": latent_height,
            "latent_width": latent_width,
            "video_height": video_height,
            "video_width": video_width,
        }


def load_video_rgb_frames(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return np.stack(frames, axis=0)


def sample_frame_ids(start_frame: int, end_frame: int, ori_fps: int, target_fps: int):
    if target_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")
    if target_fps > ori_fps:
        raise ValueError(f"target_fps ({target_fps}) cannot exceed ori_fps ({ori_fps})")
    if start_frame < 0 or end_frame <= start_frame:
        raise ValueError(f"Invalid frame segment: {start_frame}:{end_frame}")

    stride = ori_fps / target_fps
    frame_ids = np.arange(start_frame, end_frame, stride, dtype=np.float64)
    frame_ids = np.round(frame_ids).astype(np.int64)
    frame_ids = frame_ids[(frame_ids >= start_frame) & (frame_ids < end_frame)]
    frame_ids = np.unique(frame_ids)
    if frame_ids.size == 0:
        frame_ids = np.asarray([start_frame], dtype=np.int64)

    while frame_ids.size > 1 and (frame_ids.size - 1) % 4 != 0:
        frame_ids = frame_ids[:-1]

    return frame_ids


def resolve_video_keys(info: dict, camera_preset: str, explicit_video_keys: list[str] | None):
    features = info["features"]
    video_keys = (
        list(explicit_video_keys)
        if explicit_video_keys
        else list(CAMERA_PRESET_KEYS[camera_preset])
    )
    missing = [key for key in video_keys if key not in features]
    if missing:
        raise KeyError(
            f"Dataset is missing expected observation video keys: {missing}"
        )
    return video_keys


def iter_task_dirs(dataset_root: Path, tasks):
    if tasks:
        for task in tasks:
            task_dir = dataset_root / task
            if not task_dir.exists():
                raise FileNotFoundError(f"Task directory does not exist: {task_dir}")
            yield task_dir
        return

    for path in sorted(dataset_root.iterdir()):
        if path.is_dir() and (path / "meta" / "info.json").is_file():
            yield path


def build_task_jobs(
    task_dir: Path,
    target_fps: int,
    camera_preset: str,
    explicit_video_keys: list[str] | None,
):
    info = read_json(task_dir / "meta" / "info.json")
    episodes = read_jsonl(task_dir / "meta" / "episodes.jsonl")
    video_keys = resolve_video_keys(info, camera_preset, explicit_video_keys)
    ori_fps = int(info["fps"])
    chunks_size = int(info.get("chunks_size", 1000))

    print(f"[task] {task_dir.name}: {len(episodes)} episodes, {len(video_keys)} video streams")
    jobs = []

    for episode in episodes:
        episode_index = int(episode["episode_index"])
        episode_chunk = episode_index // chunks_size
        action_configs = episode.get("action_config")
        if not action_configs:
            raise KeyError(
                f"{task_dir / 'meta' / 'episodes.jsonl'} missing action_config for episode {episode_index}"
            )

        for cfg in action_configs:
            start_frame = int(cfg["start_frame"])
            end_frame = int(cfg["end_frame"])
            text = cfg["action_text"]
            frame_ids = sample_frame_ids(start_frame, end_frame, ori_fps=ori_fps, target_fps=target_fps)
            jobs.append(
                {
                    "task_dir": str(task_dir),
                    "episode_index": episode_index,
                    "episode_chunk": episode_chunk,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "text": text,
                    "frame_ids": frame_ids.tolist(),
                    "ori_fps": ori_fps,
                    "fps": int(target_fps),
                    "video_keys": list(video_keys),
                    "camera_preset": camera_preset,
                }
            )
    return jobs


def run_job(job: dict, encoder: WanLatentEncoder, overwrite: bool):
    task_dir = Path(job["task_dir"])
    episode_index = int(job["episode_index"])
    episode_chunk = int(job["episode_chunk"])
    start_frame = int(job["start_frame"])
    end_frame = int(job["end_frame"])
    text = job["text"]
    frame_ids = np.asarray(job["frame_ids"], dtype=np.int64)
    ori_fps = int(job["ori_fps"])
    target_fps = int(job["fps"])
    video_keys = job["video_keys"]
    camera_preset = job["camera_preset"]

    text_emb = encoder.encode_text(text)

    for video_key in video_keys:
        video_path = (
            task_dir
            / "videos"
            / f"chunk-{episode_chunk:03d}"
            / video_key
            / f"episode_{episode_index:06d}.mp4"
        )
        if not video_path.exists():
            raise FileNotFoundError(f"Missing video file: {video_path}")

        latent_path = (
            task_dir
            / "latents"
            / f"chunk-{episode_chunk:03d}"
            / video_key
            / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
        )
        if latent_path.exists() and not overwrite:
            continue

        all_frames = load_video_rgb_frames(video_path)
        if int(all_frames.shape[0]) < end_frame:
            raise ValueError(
                f"{video_path} has only {all_frames.shape[0]} frames, cannot access segment {start_frame}:{end_frame}"
            )
        sampled_frames = all_frames[frame_ids]

        target_height, target_width = resolve_frame_size_for_video_key(
            camera_preset,
            video_key,
            encoder.height,
            encoder.width,
        )
        encoded = encoder.encode_video_frames(
            sampled_frames,
            height=target_height,
            width=target_width,
        )
        payload = {
            "latent": encoded["latent"],
            "latent_num_frames": encoded["latent_num_frames"],
            "latent_height": encoded["latent_height"],
            "latent_width": encoded["latent_width"],
            "video_num_frames": int(len(frame_ids)),
            "video_height": encoded["video_height"],
            "video_width": encoded["video_width"],
            "text_emb": text_emb,
            "text": text,
            "frame_ids": torch.as_tensor(frame_ids, dtype=torch.int64),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "fps": int(target_fps),
            "ori_fps": int(ori_fps),
        }
        latent_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, latent_path)


def parse_gpu_ids(gpu_ids_arg: str | None, device_arg: str):
    if gpu_ids_arg:
        gpu_ids = [int(x.strip()) for x in gpu_ids_arg.split(",") if x.strip()]
    elif device_arg.startswith("cuda:"):
        gpu_ids = [int(device_arg.split(":", 1)[1])]
    elif device_arg == "cuda":
        gpu_ids = list(range(torch.cuda.device_count()))
    else:
        raise ValueError(
            f"Unsupported device setting for multi-GPU encode: {device_arg!r}. "
            "Use --device cuda, --device cuda:N, or --gpu-ids."
        )
    if not gpu_ids:
        raise ValueError("No CUDA devices available. Specify --gpu-ids or expose GPUs.")
    return gpu_ids


def shard_jobs(jobs, num_shards: int):
    shards = [[] for _ in range(num_shards)]
    for idx, job in enumerate(jobs):
        shards[idx % num_shards].append(job)
    return shards


def worker_main(
    gpu_id: int,
    jobs: list[dict],
    checkpoint_root: str,
    dtype_name: str,
    max_text_length: int,
    height: int,
    width: int,
    overwrite: bool,
):
    if not jobs:
        return

    encoder = WanLatentEncoder(
        checkpoint_root=Path(checkpoint_root),
        device=f"cuda:{gpu_id}",
        dtype=torch_dtype_from_name(dtype_name),
        max_text_length=max_text_length,
        height=height,
        width=width,
    )
    for job in jobs:
        run_job(job, encoder=encoder, overwrite=overwrite)


def main():
    args = parse_args()
    if not args.dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {args.dataset_root}")
    if not args.checkpoint_root.exists():
        raise FileNotFoundError(f"Checkpoint root does not exist: {args.checkpoint_root}")
    args.height, args.width = resolve_target_resolution(
        args.camera_preset,
        args.height,
        args.width,
    )

    all_jobs = []
    for task_dir in iter_task_dirs(args.dataset_root, args.tasks):
        all_jobs.extend(
            build_task_jobs(
                task_dir=task_dir,
                target_fps=args.fps,
                camera_preset=args.camera_preset,
                explicit_video_keys=args.video_keys,
            )
        )

    if not all_jobs:
        print("[summary] no latent jobs found")
        return

    gpu_ids = parse_gpu_ids(args.gpu_ids, args.device)
    job_shards = shard_jobs(all_jobs, len(gpu_ids))
    print(
        f"[summary] {len(all_jobs)} segment jobs across {len(gpu_ids)} GPUs: "
        + ", ".join(f"cuda:{gpu_id}={len(shard)}" for gpu_id, shard in zip(gpu_ids, job_shards))
    )

    if len(gpu_ids) == 1:
        worker_main(
            gpu_id=gpu_ids[0],
            jobs=job_shards[0],
            checkpoint_root=str(args.checkpoint_root),
            dtype_name=args.dtype,
            max_text_length=args.max_text_length,
            height=args.height,
            width=args.width,
            overwrite=args.overwrite,
        )
        return

    ctx = mp.get_context("spawn")
    processes = []
    for gpu_id, shard in zip(gpu_ids, job_shards):
        proc = ctx.Process(
            target=worker_main,
            args=(
                gpu_id,
                shard,
                str(args.checkpoint_root),
                args.dtype,
                args.max_text_length,
                args.height,
                args.width,
                args.overwrite,
            ),
        )
        proc.start()
        processes.append(proc)

    failed = []
    for proc, gpu_id in zip(processes, gpu_ids):
        proc.join()
        if proc.exitcode != 0:
            failed.append((gpu_id, proc.exitcode))

    if failed:
        raise RuntimeError(f"Multi-GPU encoding failed on workers: {failed}")


if __name__ == "__main__":
    main()
