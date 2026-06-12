#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path

from lerobot.datasets.utils import DEFAULT_DATA_FILE_SIZE_IN_MB, DEFAULT_VIDEO_FILE_SIZE_IN_MB
from lerobot.datasets.v30.convert_dataset_v21_to_v30 import (
    convert_data,
    convert_episodes_metadata,
    convert_info,
    convert_tasks,
    convert_videos,
    validate_local_dataset_version,
)


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
_EXP_ROOT = _CF_ROOT.parent

_SRC = os.environ.get("ROBOCASA_DATASET_PATH")
DEFAULT_SOURCE_ROOT = Path(_SRC) if _SRC else (_EXP_ROOT / "robocasa" / "datasets" / "training_no_base" / "atomic")
_OUT = os.environ.get("ROBOCASA_OUTPUT_DATASET_PATH")
DEFAULT_OUTPUT_ROOT = Path(_OUT) if _OUT else (_EXP_ROOT / "robocasa" / "datasets" / "training_no_base_v30" / "atomic")


def copy_file(src: str, dst: str) -> None:
    shutil.copy2(src, dst)


def copy_dir(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copytree(src, dst, copy_function=copy_file, dirs_exist_ok=True)


def convert_one_task(source_root: Path, target_root: Path, data_mb: int, video_mb: int, overwrite: bool, sidecar_dirname: str) -> None:
    validate_local_dataset_version(source_root)
    if overwrite:
        shutil.rmtree(target_root, ignore_errors=True)
    elif target_root.exists():
        raise FileExistsError(f"Refuse to overwrite existing target dir: {target_root}")
    convert_info(source_root, target_root, data_mb, video_mb)
    convert_tasks(source_root, target_root)
    episodes = convert_data(source_root, target_root, data_mb)
    videos = convert_videos(source_root, target_root, video_mb)
    convert_episodes_metadata(source_root, target_root, episodes, videos)
    copy_dir(source_root / "meta", target_root / "meta" / sidecar_dirname)
    copy_dir(source_root / "latents", target_root / "latents")
    copy_dir(source_root / "extras", target_root / "extras")


def iter_task_names(source_root: Path, tasks: list[str] | None) -> list[str]:
    if tasks:
        return tasks
    return sorted(path.name for path in source_root.iterdir() if (path / "meta" / "info.json").is_file())


def parse_args():
    parser = argparse.ArgumentParser(description="Convert RoboCasa LeRobot v2.1 tasks into a separate v3.0 dataset root.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tasks", nargs="*", default=None, help="Optional task names. Defaults to all tasks under source-root.")
    parser.add_argument("--data-file-size-in-mb", type=int, default=DEFAULT_DATA_FILE_SIZE_IN_MB)
    parser.add_argument("--video-file-size-in-mb", type=int, default=DEFAULT_VIDEO_FILE_SIZE_IN_MB)
    parser.add_argument("--sidecar-dirname", type=str, default="v21_sidecars")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if source_root == output_root:
        raise ValueError("source-root and output-root must differ")
    output_root.mkdir(parents=True, exist_ok=True)
    for task in iter_task_names(source_root, args.tasks):
        src = source_root / task
        dst = output_root / task
        if not (src / "meta" / "info.json").is_file():
            raise FileNotFoundError(f"Missing source task: {src}")
        print(f"[task] {task}")
        convert_one_task(
            source_root=src,
            target_root=dst,
            data_mb=args.data_file_size_in_mb,
            video_mb=args.video_file_size_in_mb,
            overwrite=args.overwrite,
            sidecar_dirname=args.sidecar_dirname,
        )
        print(f"[done] {src} -> {dst}")
