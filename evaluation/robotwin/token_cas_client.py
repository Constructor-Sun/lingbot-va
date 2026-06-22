"""One-shot client for LingBot-VA token CAS smoke tests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.robotwin.websocket_client_policy import WebsocketClientPolicy


CAM_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
VIEW_SLICES = {
    "observation.images.cam_high": (slice(4, 12), slice(0, 10)),
    "observation.images.cam_left_wrist": (slice(0, 4), slice(0, 5)),
    "observation.images.cam_right_wrist": (slice(0, 4), slice(5, 10)),
}


def load_obs(example_dir: Path) -> dict:
    return {
        key: np.array(Image.open(example_dir / f"{key}.png").convert("RGB"))
        for key in CAM_KEYS
    }


def save_camera_maps(cas_grid: np.ndarray, obs: dict, output_dir: Path) -> None:
    for cam_key, (rs, cs) in VIEW_SLICES.items():
        cam_short = cam_key.split(".")[-1]
        original = Image.fromarray(obs[cam_key])
        original.save(output_dir / f"observation_{cam_short}.png")

        view = cas_grid[:, rs, cs]
        view = np.nanmean(view, axis=0)
        valid = np.isfinite(view)
        if not valid.any():
            view = np.zeros_like(view, dtype=np.float32)
        else:
            lo, hi = float(view[valid].min()), float(view[valid].max())
            view = np.nan_to_num((view - lo) / max(hi - lo, 1e-8), nan=0.0)
        h, w = obs[cam_key].shape[:2]
        gray = Image.fromarray((view * 255).astype(np.uint8), mode="L")
        gray = gray.resize((w, h), resample=Image.BILINEAR)
        gray.save(output_dir / f"cas_map_{cam_short}.png")

        heat = np.zeros((h, w, 3), dtype=np.float32)
        heat[..., 0] = np.asarray(gray, dtype=np.float32)
        heat[..., 1] = np.asarray(gray, dtype=np.float32) * 0.35
        base = np.asarray(original, dtype=np.float32)
        overlay = np.clip(base * 0.55 + heat * 0.45, 0, 255).astype(np.uint8)
        Image.fromarray(overlay).save(output_dir / f"cas_overlay_{cam_short}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=29169)
    parser.add_argument("--example-dir", type=Path, default=Path("example/robotwin"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/token_cas_smoke"))
    parser.add_argument("--prompt", default="Pick up the object.")
    parser.add_argument("--mode", choices=["zero", "mean"], default="zero")
    parser.add_argument("--mean-scope", choices=["same_camera", "same_frame", "global"], default="same_camera")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--show-progress", action="store_true")
    args = parser.parse_args()

    obs = load_obs(args.example_dir)
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    client.infer({"reset": True, "prompt": args.prompt})
    ret = client.infer(
        {
            "obs": obs,
            "prompt": args.prompt,
            "cas_enabled": True,
            "cas_mode": args.mode,
            "cas_per_token": True,
            "cas_max_tokens": args.max_tokens,
            "cas_mean_scope": args.mean_scope,
            "cas_show_progress": args.show_progress,
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "action.npy", ret["action"])
    np.save(args.output_dir / "cas_grid.npy", ret["cas"]["grid"])
    np.save(args.output_dir / "cas_scores.npy", ret["cas"]["scores"])
    np.save(args.output_dir / "cas_token_indices.npy", ret["cas"]["token_indices"])
    save_camera_maps(ret["cas"]["grid"], obs, args.output_dir)
    with open(args.output_dir / "cas_entries.json", "w", encoding="utf-8") as f:
        json.dump(ret["cas"]["entries"], f, indent=2)
    print(f"Saved CAS outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
