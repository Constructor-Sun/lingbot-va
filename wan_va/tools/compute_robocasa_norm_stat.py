import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


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

_DATASET = os.environ.get("ROBOCASA_DATASET_PATH")
DEFAULT_DATASET_ROOT = _DATASET if _DATASET else str(_EXP_ROOT / "robocasa" / "datasets" / "training_no_base" / "atomic")
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "va_robocasa_no_base_cfg.py"
)
MODEL_ACTION_DIM = 30
NO_BASE_ACTION_DIM = 7


def find_task_dirs(dataset_root, tasks=None):
    dataset_root = Path(dataset_root)
    task_set = set(tasks or [])
    task_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir())

    selected = []
    for task_dir in task_dirs:
        if task_set and task_dir.name not in task_set:
            continue
        info_path = task_dir / "meta" / "info.json"
        if not info_path.exists():
            continue
        selected.append(task_dir)
    return selected


def load_action_samples(task_dir):
    parquet_paths = sorted((Path(task_dir) / "data").glob("*/*.parquet"))
    if not parquet_paths:
        raise ValueError(f"No parquet action data found under {task_dir / 'data'}")

    action_chunks = []
    for parquet_path in parquet_paths:
        df = pd.read_parquet(parquet_path, columns=["action"])
        if df.empty:
            continue
        chunk = np.vstack(df["action"].to_numpy()).astype(np.float64, copy=False)
        if chunk.shape[1] != NO_BASE_ACTION_DIM:
            raise ValueError(
                f"{parquet_path} action dim must be {NO_BASE_ACTION_DIM}, got {chunk.shape}"
            )
        action_chunks.append(chunk)

    if not action_chunks:
        raise ValueError(f"No action samples loaded from {task_dir / 'data'}")
    return np.concatenate(action_chunks, axis=0)


def compute_norm_stat(task_dirs):
    action_chunks = []
    for task_dir in task_dirs:
        action_chunks.append(load_action_samples(task_dir))

    if not action_chunks:
        raise ValueError("No RoboCasa no-base tasks found.")

    actions = np.concatenate(action_chunks, axis=0)
    q01 = np.quantile(actions, 0.01, axis=0)
    q99 = np.quantile(actions, 0.99, axis=0)
    padding = [0.0] * (MODEL_ACTION_DIM - NO_BASE_ACTION_DIM)
    return q01.tolist() + padding, q99.tolist() + padding


def format_float_list(values, indent="        "):
    chunks = []
    for i in range(0, len(values), 6):
        row = ", ".join(f"{float(v):.10g}" for v in values[i : i + 6])
        chunks.append(f"{indent}{row},")
    return "[\n" + "\n".join(chunks) + "\n    ]"


def replace_norm_stat(config_path, q01, q99):
    config_path = Path(config_path)
    text = config_path.read_text(encoding="utf-8")
    replacement = (
        "va_robocasa_no_base_cfg.norm_stat = {\n"
        "    \"q01\": "
        + format_float_list(q01)
        + ",\n"
        "    \"q99\": "
        + format_float_list(q99)
        + ",\n"
        "}\n"
    )
    pattern = r"va_robocasa_no_base_cfg\.norm_stat = \{.*?\n\}\n"
    new_text, count = re.subn(pattern, replacement, text, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(
            f"Expected one norm_stat block in {config_path}, replaced {count}."
        )
    config_path.write_text(new_text, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--config-path", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="*",
        default=None,
        help="Optional task names. If omitted, all task dirs under dataset-root are used.",
    )
    parser.add_argument(
        "--write-config",
        action="store_true",
        help="Write q01/q99 back into va_robocasa_no_base_cfg.py.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    task_dirs = find_task_dirs(args.dataset_root, args.tasks)
    q01, q99 = compute_norm_stat(task_dirs)

    print(f"Used {len(task_dirs)} RoboCasa no-base task dirs.")
    print('"q01": ' + format_float_list(q01) + ",")
    print('"q99": ' + format_float_list(q99) + ",")

    if args.write_config:
        replace_norm_stat(args.config_path, q01, q99)
        print(f"Updated config: {args.config_path}")


if __name__ == "__main__":
    main()
