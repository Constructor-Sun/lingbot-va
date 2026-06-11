import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATASET_ROOT = Path("/data1/liu/exp/robocasa/datasets/training_no_base/atomic")
ACTION_SLICE = slice(5, 12)
ACTION_DIM_7 = 7


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Add action_config and convert RoboCasa no-base actions from 12D to 7D "
            "for LingBot-VA training."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root like robocasa/datasets/training_no_base/atomic.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="*",
        default=None,
        help="Optional task names. If omitted, all task folders under dataset-root are processed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print intended updates without writing files.",
    )
    return parser.parse_args()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=4)
        f.write("\n")


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True))
            f.write("\n")


def slice_action_array(value):
    return np.asarray(value)[ACTION_SLICE]


def slice_stat_entry(entry):
    out = dict(entry)
    for key in ("mean", "std", "min", "max", "q01", "q99"):
        if key in out:
            out[key] = np.asarray(out[key], dtype=np.float64)[ACTION_SLICE].tolist()
    return out


def build_action_config(task_text: str, length: int):
    return [
        {
            "start_frame": 0,
            "end_frame": length,
            "action_text": task_text,
        }
    ]


def update_modality(modality: dict):
    modality = json.loads(json.dumps(modality))
    modality["action"] = {
        "end_effector_position": {
            "original_key": "action",
            "start": 0,
            "end": 3,
        },
        "end_effector_rotation": {
            "original_key": "action",
            "start": 3,
            "end": 6,
        },
        "gripper_close": {
            "original_key": "action",
            "start": 6,
            "end": 7,
        },
    }
    return modality


def update_info(info: dict):
    info = json.loads(json.dumps(info))
    info["features"]["action"]["shape"] = [ACTION_DIM_7]
    return info


def update_episodes_jsonl(rows):
    out = []
    for row in rows:
        updated = dict(row)
        tasks = updated.get("tasks", [])
        task_text = tasks[0] if tasks else ""
        length = int(updated["length"])
        updated["action_config"] = build_action_config(task_text, length)
        out.append(updated)
    return out


def update_episodes_stats_jsonl(rows):
    out = []
    for row in rows:
        updated = json.loads(json.dumps(row))
        if "action" in updated.get("stats", {}):
            updated["stats"]["action"] = slice_stat_entry(updated["stats"]["action"])
        out.append(updated)
    return out


def update_stats_json(stats: dict):
    stats = json.loads(json.dumps(stats))
    if "action" in stats:
        stats["action"] = slice_stat_entry(stats["action"])
    return stats


def remap_parquet(parquet_path: Path, dry_run: bool):
    df = pd.read_parquet(parquet_path)
    if "action" not in df.columns:
        raise KeyError(f"{parquet_path} missing action column")

    df["action"] = df["action"].map(slice_action_array)
    if dry_run:
        return
    df.to_parquet(parquet_path, index=False)


def process_task(task_dir: Path, dry_run: bool):
    meta_dir = task_dir / "meta"
    data_dir = task_dir / "data"

    info_path = meta_dir / "info.json"
    modality_path = meta_dir / "modality.json"
    episodes_path = meta_dir / "episodes.jsonl"
    episodes_stats_path = meta_dir / "episodes_stats.jsonl"
    stats_path = meta_dir / "stats.json"

    info = read_json(info_path)
    modality = read_json(modality_path)
    episodes = read_jsonl(episodes_path)
    episodes_stats = read_jsonl(episodes_stats_path)
    stats = read_json(stats_path)

    updated_info = update_info(info)
    updated_modality = update_modality(modality)
    updated_episodes = update_episodes_jsonl(episodes)
    updated_episodes_stats = update_episodes_stats_jsonl(episodes_stats)
    updated_stats = update_stats_json(stats)

    parquet_paths = sorted(data_dir.glob("*/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")

    if dry_run:
        print(f"[dry-run] {task_dir.name}: {len(parquet_paths)} parquet files")
    else:
        write_json(info_path, updated_info)
        write_json(modality_path, updated_modality)
        write_jsonl(episodes_path, updated_episodes)
        write_jsonl(episodes_stats_path, updated_episodes_stats)
        write_json(stats_path, updated_stats)

    for parquet_path in parquet_paths:
        remap_parquet(parquet_path, dry_run=dry_run)

    print(f"[done] {task_dir.name}")


def main():
    args = parse_args()
    dataset_root = args.dataset_root
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    if args.tasks:
        task_dirs = [dataset_root / task for task in args.tasks]
    else:
        task_dirs = sorted(
            path for path in dataset_root.iterdir() if path.is_dir() and (path / "meta").is_dir()
        )

    for task_dir in task_dirs:
        if not task_dir.exists():
            raise FileNotFoundError(f"Task directory does not exist: {task_dir}")
        process_task(task_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
