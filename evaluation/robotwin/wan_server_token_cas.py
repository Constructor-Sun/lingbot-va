"""Non-invasive LingBot-VA token-CAS server entrypoint."""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
VA_ROOT = REPO_ROOT / "external" / "lingbot-va"
WAN_VA_ROOT = VA_ROOT / "wan_va"
ROBUST_WAM_SRC = REPO_ROOT / "robust_wam" / "src"
VA_SERVER_PATH = WAN_VA_ROOT / "wan_va_server.py"

for path in (VA_ROOT, WAN_VA_ROOT, ROBUST_WAM_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

spec = importlib.util.spec_from_file_location("lingbot_va_wan_server", VA_SERVER_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load {VA_SERVER_PATH}")

_va_server = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = _va_server
spec.loader.exec_module(_va_server)

from motivation.CAS import cas_requested, run_token_cas  # noqa: E402
from wan_va.utils.Simple_Remote_Infer.deploy.websocket_policy_server import (  # noqa: E402
    WebsocketPolicyServer,
)


def _patch_va_server_class():
    if getattr(_va_server.VA_Server, "_token_cas_class_patched", False):
        return

    original_infer = _va_server.VA_Server.infer

    def patched_infer(self, obs):
        if obs.get("reset", False) or obs.get("compute_kv_cache", False) or not cas_requested(obs):
            return original_infer(self, obs)
        _va_server.logger.info("################# Infer One Chunk with Token CAS #################")
        return run_token_cas(self, obs)

    _va_server.VA_Server.infer = patched_infer
    _va_server.VA_Server._token_cas_class_patched = True


def run_server(args):
    _patch_va_server_class()
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

    server = _va_server.VA_Server(config)
    _va_server.logger.info(f"[token-cas] serving on ws://0.0.0.0:{port}")
    WebsocketPolicyServer(
        server,
        host="0.0.0.0",
        port=port,
        metadata={"server": "lingbot-va-token-cas", "config_name": args.config_name},
    ).serve_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, default="robotwin")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--save-root", "--save_root", dest="save_root", default=None)
    args = parser.parse_args()
    run_server(args)


if __name__ == "__main__":
    main()
