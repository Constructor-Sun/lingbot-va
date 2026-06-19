"""Server entrypoint for attention-saliency capture without modifying LingBot-VA.

Usage::

    python -m torch.distributed.run \\
        --nproc_per_node=1 --master_port 29721 --tee 3 \\
        wan_server_attn_capture.py \\
        --config-name robotwin_i2av
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[4]  # counterfactual root
VA_ROOT = REPO_ROOT / "external" / "lingbot-va"
WAN_VA_ROOT = VA_ROOT / "wan_va"
VA_SERVER_PATH = WAN_VA_ROOT / "wan_va_server.py"
ROBUST_WAM_SRC = REPO_ROOT / "robust_wam" / "src"

for path in (VA_ROOT, WAN_VA_ROOT, ROBUST_WAM_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# ---------------------------------------------------------------------------
# Load original wan_va_server (importlib — does not touch on-disk source)
# ---------------------------------------------------------------------------
spec = importlib.util.spec_from_file_location("lingbot_va_wan_server", VA_SERVER_PATH)
_va_server = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = _va_server
spec.loader.exec_module(_va_server)

from motivation.attention_mask import attention_saliency  # noqa: E402
from wan_va.utils.Simple_Remote_Infer.deploy.websocket_policy_server import (  # noqa: E402
    WebsocketPolicyServer,
)


# ---------------------------------------------------------------------------
# Patch VA_Server
# ---------------------------------------------------------------------------
def _install(server, targets, last_n_layers, capture_mode, capture_stride):
    if getattr(server, "_attn_installed", False):
        return
    if last_n_layers == 0:
        last_n_layers = len(server.transformer.blocks)  # 0 → all layers
    if capture_stride <= 0 and capture_mode == "all_steps":
        n = server.job_config.num_inference_steps + server.job_config.action_num_inference_steps
        capture_stride = max(1, n // 2)
    attention_saliency.install(
        server.transformer, targets=targets, last_n_layers=last_n_layers,
        capture_mode=capture_mode, capture_stride=max(int(capture_stride), 1),
    )
    server._attn_installed = True
    server._attn_targets = targets
    server._attn_out_dir = None
    server._attn_chunk = 0
    server._attn_prompt = ""
    server._attn_frames_by_frame = {}
    server._attn_pending_chunks = []

    # Clear VAE streaming cache before single-frame encodes so that every
    # T=1 chunk is a "first call".  The Wan encoder's downsample3d time_conv
    # (kernel=3, stride=2) requires ≥3 temporal frames on non-first streaming
    # calls (1 cached + ≥2 new), which single-frame calls cannot satisfy.
    # Multi-frame calls (T≥2) keep the cache so streaming works as designed.
    _orig_encode_obs = server._encode_obs

    def _patched_encode_obs(obs):
        images = obs['obs']
        if not isinstance(images, list):
            images = [images]
        if len(images) == 1:
            server.streaming_vae.clear_cache()
            if hasattr(server, 'streaming_vae_half'):
                server.streaming_vae_half.clear_cache()
        return _orig_encode_obs(obs)

    server._encode_obs = _patched_encode_obs


def _patch(attn_targets, attn_last_layers, capture_mode, capture_stride):
    if getattr(_va_server.VA_Server, "_attn_patched", False):
        return

    orig_init = _va_server.VA_Server.__init__
    orig_infer = _va_server.VA_Server.infer

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        _install(self, attn_targets, attn_last_layers, capture_mode, capture_stride)

    def patched_infer(self, obs):
        if obs.get("reset", False):
            _finish_episode(self)
            attention_saliency.pop_captured()
            result = orig_infer(self, obs)
            # Use client-supplied episode_name, fall back to auto-generated exp_name
            ep_name = obs.get("episode_name", "")
            if ep_name:
                base = os.path.join(self.job_config.save_root, ep_name)
                self.exp_save_root = base
                os.makedirs(base, exist_ok=True)
            self._attn_out_dir = self.exp_save_root
            self._attn_chunk = 0
            self._attn_prompt = obs.get("prompt", "")
            self._attn_frames_by_frame = {}
            self._attn_pending_chunks = []
            os.makedirs(self._attn_out_dir, exist_ok=True)
            return result
        if obs.get("compute_kv_cache", False):
            result = orig_infer(self, obs)
            chunk_data = _render_pending_chunks(self, obs)
            if chunk_data:
                result['attn_saliency'] = chunk_data
            return result
        # Normal _infer call
        action, latents = self._infer(obs, frame_st_id=self.frame_st_id)
        result = {"action": action}
        if obs.get("return_video", False):
            if not hasattr(self, "video_processor"):
                self.video_processor = _va_server.VideoProcessor(vae_scale_factor=1)
            if self.enable_offload:
                self.vae = self.vae.to(self.device).to(self.dtype)
            with torch.no_grad():
                result["video"] = self.decode_one_video(latents.detach(), "np")[0]
        captured = attention_saliency.pop_captured()
        if captured and self._attn_out_dir is not None:
            _record_chunk(self, captured, obs)
        return result

    _va_server.VA_Server.__init__ = patched_init
    _va_server.VA_Server.infer = patched_infer
    _va_server.VA_Server._attn_patched = True


def _record_chunk(server, captured, obs):
    """Save per-chunk saliency arrays and queue rendering until observations arrive."""
    saliency_maps = attention_saliency.build_saliency_maps(
        captured, targets=server._attn_targets,
    )
    chunk = server._attn_chunk
    server._attn_chunk += 1
    out = server._attn_out_dir

    # Save arrays immediately. PNG/video rendering is deferred until the
    # following compute_kv_cache request brings the moving observation frames.
    chunk_dir = os.path.join(out, "chunks", f"chunk{chunk:03d}")
    os.makedirs(chunk_dir, exist_ok=True)
    for target in server._attn_targets:
        payload = saliency_maps.get(target, {})
        grid = payload.get("grid")
        if grid is not None:
            np.save(os.path.join(chunk_dir, f"saliency_{target}.npy"), grid)

    server._attn_pending_chunks.append({
        "chunk": chunk,
        "chunk_dir": chunk_dir,
        "saliency_maps": saliency_maps,
        "infer_obs": obs.get("obs", obs),
    })

    _va_server.logger.info(f"[attn-capture] chunk {chunk} saliency queued → {chunk_dir}")


def _split_obs_by_latent_frame(obs_list, infer_obs, state, n_frames, chunk):
    """Map client key-frame observations onto latent f0/f1 slots."""
    if not isinstance(obs_list, list):
        obs_list = [obs_list]

    groups = [[] for _ in range(n_frames)]
    obs_per_latent = None
    if state is not None and hasattr(state, "shape") and len(state.shape) >= 3:
        action_steps = int(state.shape[2])
        obs_per_latent = 4 if action_steps >= 4 else 1

    # The original client skips action frame 0 on the first policy chunk, so
    # compute_kv_cache only sends observations for f1. Use the infer request's
    # initial observation as f0 to keep both videos moving on the same timeline.
    if chunk == 0 and obs_per_latent and len(obs_list) == (n_frames - 1) * obs_per_latent:
        if n_frames > 0 and infer_obs is not None:
            groups[0] = [infer_obs]
        if n_frames > 1:
            groups[1] = list(obs_list)
        return groups

    if n_frames > 0 and obs_list:
        if len(obs_list) >= n_frames:
            for frame_idx, split in enumerate(np.array_split(np.array(obs_list, dtype=object), n_frames)):
                groups[frame_idx] = list(split)
        else:
            for frame_idx, item in enumerate(obs_list):
                groups[frame_idx] = [item]

    return groups


def _render_pending_chunks(server, obs):
    """Render queued saliency chunks using the moving observations from the client.

    Returns a list of per-chunk saliency data that can be sent back to the client.
    """
    pending = getattr(server, "_attn_pending_chunks", [])
    if not pending:
        return []

    obs_payload = obs.get("obs", [])
    state = obs.get("state")
    rendered_any = False
    collected = []  # per-chunk saliency data for client response

    while pending:
        item = pending.pop(0)
        saliency_maps = item["saliency_maps"]
        available = [t for t in server._attn_targets
                     if saliency_maps.get(t, {}).get("grid") is not None]
        if not available:
            continue

        n_frames = int(saliency_maps[available[0]]["grid"].shape[0])
        obs_groups = _split_obs_by_latent_frame(
            obs_payload, item.get("infer_obs"), state, n_frames, item["chunk"],
        )
        sal_dir = item["chunk_dir"]
        os.makedirs(sal_dir, exist_ok=True)

        ref_obs = None
        for frame_idx, obs_group in enumerate(obs_groups):
            if not obs_group:
                continue
            last_frame = None
            for obs_dict in obs_group:
                if ref_obs is None:
                    ref_obs = obs_dict
                pil = attention_saliency._render_viz_frame(
                    obs_dict, saliency_maps, server._attn_prompt or "inference",
                    frame_idx, available,
                )
                frame_arr = np.array(pil)
                server._attn_frames_by_frame.setdefault(frame_idx, []).append(frame_arr)
                last_frame = pil
                rendered_any = True
            if last_frame is not None:
                last_frame.save(os.path.join(sal_dir, f"saliency_overview_f{frame_idx}.png"))

        # Save standalone saliency heatmap views (no video overlay) for each target
        _save_saliency_heatmap_views(sal_dir, saliency_maps, available, ref_obs)

        # Collect raw saliency data to send back to the client.
        # Only include the lightweight per-target grids (numpy arrays,
        # msgpack_numpy handles them transparently), not the rendered images.
        client_payload = {}
        for tgt in available:
            client_payload[tgt] = {
                'grid': saliency_maps[tgt]['grid'].copy(),
                'layer_indices': saliency_maps[tgt].get('layer_indices', []),
            }
        collected.append({
            'chunk': item['chunk'],
            'saliency_maps': client_payload,
        })

        _va_server.logger.info(
            f"[attn-capture] chunk {item['chunk']} rendered with observations → {item['chunk_dir']}"
        )

    if rendered_any:
        _write_episode_videos(server)
        _cleanup_pt_files(server)

    return collected


def _cleanup_pt_files(server):
    """Remove intermediate .pt files (latents, actions, obs_data) after episode rendering."""
    exp_root = getattr(server, "exp_save_root", None)
    if not exp_root or not os.path.isdir(exp_root):
        return
    import glob
    removed = 0
    for pt_file in glob.glob(os.path.join(exp_root, "*.pt")):
        try:
            os.remove(pt_file)
            removed += 1
        except OSError:
            pass
    if removed:
        _va_server.logger.info(f"[attn-capture] cleaned up {removed} .pt files from {exp_root}")


def _save_saliency_heatmap_views(sal_dir, saliency_maps, available, ref_obs=None):
    """Save standalone saliency heatmap images (no video overlay) for each camera view.

    For each target and each latent frame, renders the three camera-view sub-regions
    of the saliency grid as pure JET-colormap heatmaps, resized to match the original
    camera image dimensions.
    """
    if not available:
        return

    # Determine target size per camera from reference observation
    cam_sizes = {}
    if ref_obs:
        for cam_name in attention_saliency.VIEW_SLICES:
            cam_img = ref_obs.get(cam_name)
            if cam_img is not None:
                cam_sizes[cam_name] = (cam_img.shape[1], cam_img.shape[0])  # (W, H)

    for tgt in available:
        grid = saliency_maps[tgt].get("grid")
        if grid is None:
            continue
        n_frames = grid.shape[0]
        for fi in range(n_frames):
            for cam_name, (rs, cs) in attention_saliency.VIEW_SLICES.items():
                view_map = grid[fi][rs, cs]
                norm = attention_saliency._normalize_map(view_map)
                # Resize to match camera image dimensions (or keep raw if unavailable)
                if cam_name in cam_sizes:
                    tw, th = cam_sizes[cam_name]
                    norm = cv2.resize(norm.astype(np.float32), (tw, th),
                                      interpolation=cv2.INTER_LINEAR)
                heat_u8 = (norm * 255).astype(np.uint8)
                heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
                heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
                cam_short = cam_name.split(".")[-1]
                out_path = os.path.join(
                    sal_dir, f"saliency_{tgt}_{cam_short}_f{fi}.png")
                Image.fromarray(heat_color).save(out_path)
                _va_server.logger.info(
                    f"[attn-capture] heatmap saved → {out_path}")


def _write_episode_videos(server):
    """Write current accumulated f0/f1 frames as episode mp4 files."""
    frames_by_frame = getattr(server, "_attn_frames_by_frame", {})
    if not frames_by_frame:
        return
    out = server._attn_out_dir
    # Guard: the directory may have been renamed by the client (to include
    # success/failure) by the time the _next_ reset triggers _finish_episode.
    if out and os.path.isdir(out):
        import imageio
        for frame_idx, frames in sorted(frames_by_frame.items()):
            frame_video_path = os.path.join(out, f"episode_attention_f{frame_idx}.mp4")
            imageio.mimsave(frame_video_path, frames, fps=10)
            _va_server.logger.info(
                f"[attn-capture] episode f{frame_idx} video saved ({len(frames)} frames) → {frame_video_path}"
            )


def _finish_episode(server):
    """Export accumulated frames as episode mp4 on reset and clean up .pt files."""
    _write_episode_videos(server)
    server._attn_frames_by_frame = {}
    _cleanup_pt_files(server)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def run_server(args):
    config = _va_server.VA_CONFIGS[args.config_name]
    port = config.port if args.port is None else args.port
    if args.save_root is not None:
        config.save_root = args.save_root
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    _va_server.init_distributed(world_size, local_rank, rank)
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    _patch(args.attn_targets, args.attn_last_layers,
           args.attn_capture_mode, args.attn_capture_stride)

    server = _va_server.VA_Server(config)
    ws = WebsocketPolicyServer(server, host="0.0.0.0", port=port)
    _va_server.logger.info(f"[attn-capture] serving on ws://0.0.0.0:{port}")
    ws.serve_forever()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", default="robotwin")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--save-root", default=None)
    p.add_argument("--attn-targets", default="action_to_video")
    p.add_argument("--attn-last-layers", type=int, default=0,
                   help="Capture last N layers; 0 = all layers (30).")
    p.add_argument("--attn-capture-mode", default="last_step",
                   choices=["last_step", "all_steps"])
    p.add_argument("--attn-capture-stride", type=int, default=0,
                   help="Stride for all_steps mode; 0 = auto.")
    args = p.parse_args()
    args.attn_targets = attention_saliency.parse_targets(args.attn_targets)
    run_server(args)


if __name__ == "__main__":
    main()
