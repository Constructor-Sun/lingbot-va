import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATASET_ROOT = "/data1/liu/exp/robocasa/datasets/v1.0/pretrain/atomic"
DEFAULT_OUTPUT_DIR = Path("~/exp").expanduser()


@dataclass
class EpisodeMotionStats:
    base_dim: int
    num_frames: int = 0
    moving_frame_count: int = 0
    sum_abs_all: float = 0.0
    sum_l2: float = 0.0
    max_abs: float = 0.0
    max_l2: float = 0.0

    def __post_init__(self):
        self.max_abs_per_dim = np.zeros(self.base_dim, dtype=np.float64)

    def update(self, base_motion_chunk, movement_threshold):
        if base_motion_chunk.size == 0:
            return

        abs_chunk = np.abs(base_motion_chunk)
        l2_chunk = np.linalg.norm(base_motion_chunk, axis=1)
        max_abs_per_frame = abs_chunk.max(axis=1)
        moving_mask = max_abs_per_frame > movement_threshold

        self.num_frames += int(base_motion_chunk.shape[0])
        self.moving_frame_count += int(moving_mask.sum())
        self.sum_abs_all += float(abs_chunk.sum())
        self.sum_l2 += float(l2_chunk.sum())
        self.max_abs = max(self.max_abs, float(max_abs_per_frame.max(initial=0.0)))
        self.max_l2 = max(self.max_l2, float(l2_chunk.max(initial=0.0)))
        self.max_abs_per_dim = np.maximum(self.max_abs_per_dim, abs_chunk.max(axis=0))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure actual RoboCasa base motion from LeRobot action sequences and "
            "report which episodes/tasks contain base movement."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(DEFAULT_DATASET_ROOT),
        help=(
            "Root directory like robocasa/datasets/v1.0/pretrain/atomic or "
            "robocasa/datasets/v1.0/target."
        ),
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="*",
        default=None,
        help="Optional task names. If omitted, all tasks under dataset-root are used.",
    )
    parser.add_argument(
        "--movement-threshold",
        type=float,
        default=0.0,
        help=(
            "A frame is counted as base-moving if max(abs(base_motion)) exceeds this threshold."
        ),
    )
    parser.add_argument(
        "--min-moving-frames",
        type=int,
        default=1,
        help="An episode is marked moving only if at least this many frames exceed the threshold.",
    )
    parser.add_argument(
        "--min-moving-frame-ratio",
        type=float,
        default=0.0,
        help="An episode is marked moving only if moving_frame_count / num_frames is at least this ratio.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write the minimal episode CSV and summary JSON reports.",
    )
    return parser.parse_args()


def find_modality_paths(dataset_root, tasks=None):
    dataset_root = Path(dataset_root)
    task_set = set(tasks or [])
    modality_paths = sorted(dataset_root.glob("**/lerobot/meta/modality.json"))

    selected = []
    for modality_path in modality_paths:
        rel_parts = modality_path.relative_to(dataset_root).parts
        if len(rel_parts) < 5:
            raise ValueError(f"Unexpected modality path layout: {modality_path}")
        task_name = rel_parts[0]
        if task_set and task_name not in task_set:
            continue
        selected.append(modality_path)
    return selected


def load_base_motion_slice(modality_path):
    with Path(modality_path).open("r", encoding="utf-8") as f:
        modality = json.load(f)

    action_modality = modality.get("action", {})
    if "base_motion" not in action_modality:
        raise KeyError(f"{modality_path} missing action modality 'base_motion'")

    start = int(action_modality["base_motion"]["start"])
    end = int(action_modality["base_motion"]["end"])
    if not (0 <= start < end):
        raise ValueError(f"Invalid base_motion slice in {modality_path}: {start}:{end}")

    return slice(start, end)


def get_dataset_identifiers(dataset_root, modality_path):
    rel_parts = modality_path.relative_to(dataset_root).parts
    task_name = rel_parts[0]
    collection_id = rel_parts[1]
    dataset_id = f"{task_name}/{collection_id}"
    lerobot_root = modality_path.parents[1]
    return task_name, collection_id, dataset_id, lerobot_root


def analyze_dataset(
    dataset_root,
    modality_paths,
    movement_threshold,
    min_moving_frames,
    min_moving_frame_ratio,
):
    episode_stats = {}

    for modality_path in modality_paths:
        base_slice = load_base_motion_slice(modality_path)
        base_dim = base_slice.stop - base_slice.start
        task_name, collection_id, dataset_id, lerobot_root = get_dataset_identifiers(
            dataset_root, modality_path
        )

        parquet_paths = sorted((lerobot_root / "data").glob("*/*.parquet"))
        if not parquet_paths:
            raise ValueError(f"No parquet files found under {lerobot_root / 'data'}")

        for parquet_path in parquet_paths:
            df = pd.read_parquet(parquet_path, columns=["episode_index", "action"])
            if df.empty:
                continue

            actions = np.vstack(df["action"].to_numpy()).astype(np.float64, copy=False)
            if actions.ndim != 2 or actions.shape[1] < base_slice.stop:
                raise ValueError(
                    f"{parquet_path} action shape must cover base slice {base_slice}, got {actions.shape}"
                )

            base_motion = actions[:, base_slice]
            episode_indices = df["episode_index"].to_numpy(dtype=np.int64, copy=False)
            unique_episode_indices = np.unique(episode_indices)

            for episode_index in unique_episode_indices:
                episode_mask = episode_indices == episode_index
                key = (task_name, collection_id, dataset_id, int(episode_index))
                if key not in episode_stats:
                    episode_stats[key] = EpisodeMotionStats(base_dim=base_dim)
                episode_stats[key].update(
                    base_motion[episode_mask], movement_threshold=movement_threshold
                )

    rows = []
    for (task_name, collection_id, dataset_id, episode_index), stats in sorted(
        episode_stats.items()
    ):
        moving_frame_ratio = (
            stats.moving_frame_count / stats.num_frames if stats.num_frames else 0.0
        )
        is_moving = (
            stats.moving_frame_count >= min_moving_frames
            and moving_frame_ratio >= min_moving_frame_ratio
        )
        row = {
            "task_name": task_name,
            "collection_id": collection_id,
            "dataset_id": dataset_id,
            "episode_index": episode_index,
            "num_frames": stats.num_frames,
            "moving_frame_count": stats.moving_frame_count,
            "moving_frame_ratio": moving_frame_ratio,
            "is_moving": bool(is_moving),
            "max_abs_base_motion": stats.max_abs,
            "mean_abs_base_motion": (
                stats.sum_abs_all / (stats.num_frames * stats.base_dim)
                if stats.num_frames
                else 0.0
            ),
            "max_l2_base_motion": stats.max_l2,
            "mean_l2_base_motion": (
                stats.sum_l2 / stats.num_frames if stats.num_frames else 0.0
            ),
        }
        for dim_idx, dim_max in enumerate(stats.max_abs_per_dim):
            row[f"max_abs_base_motion_dim_{dim_idx}"] = float(dim_max)
        rows.append(row)

    if not rows:
        raise ValueError(f"No episodes found under {dataset_root}")

    episode_df = pd.DataFrame(rows).sort_values(
        by=["task_name", "collection_id", "episode_index"]
    )

    summary = {
        "dataset_root": str(Path(dataset_root).resolve()),
        "num_episodes": int(len(episode_df)),
        "moving_episodes": int(episode_df["is_moving"].sum()),
        "static_episodes": int((~episode_df["is_moving"]).sum()),
        "movement_threshold": movement_threshold,
        "min_moving_frames": min_moving_frames,
        "min_moving_frame_ratio": min_moving_frame_ratio,
        "overall_moving_episode_ratio": float(episode_df["is_moving"].mean()),
        "overall_static_episode_ratio": float((~episode_df["is_moving"]).mean()),
    }

    episode_report_df = episode_df[
        ["task_name", "collection_id", "dataset_id", "episode_index", "is_moving"]
    ].copy()
    return episode_report_df, summary


def print_summary(summary):
    print(f"Dataset root: {summary['dataset_root']}")
    print(
        "Episodes: "
        f"{summary['num_episodes']} total, "
        f"{summary['moving_episodes']} moving, "
        f"{summary['static_episodes']} static"
    )
    print(
        "Thresholds: "
        f"movement>{summary['movement_threshold']}, "
        f"moving_frames>={summary['min_moving_frames']}, "
        f"moving_frame_ratio>={summary['min_moving_frame_ratio']}"
    )
    print(
        "Episode ratios: "
        f"moving={summary['overall_moving_episode_ratio']:.4f}, "
        f"static={summary['overall_static_episode_ratio']:.4f}"
    )

def write_reports(output_dir, episode_df, summary):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_path = output_dir / "robocasa_base_motion_episode_flags.csv"
    summary_path = output_dir / "robocasa_base_motion_summary.json"

    episode_df.to_csv(episode_path, index=False)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nWrote reports:")
    print(f"  {episode_path}")
    print(f"  {summary_path}")


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    modality_paths = find_modality_paths(dataset_root, tasks=args.tasks)
    episode_df, summary = analyze_dataset(
        dataset_root=dataset_root,
        modality_paths=modality_paths,
        movement_threshold=args.movement_threshold,
        min_moving_frames=args.min_moving_frames,
        min_moving_frame_ratio=args.min_moving_frame_ratio,
    )
    print_summary(summary=summary)
    write_reports(
        output_dir=args.output_dir,
        episode_df=episode_df,
        summary=summary,
    )


if __name__ == "__main__":
    main()
