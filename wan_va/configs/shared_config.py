# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os
from pathlib import Path

import torch
from easydict import EasyDict


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

va_shared_cfg = EasyDict()

va_shared_cfg.host = '0.0.0.0'
va_shared_cfg.port = 29536

va_shared_cfg.param_dtype = torch.bfloat16
va_shared_cfg.save_root = str(_CF_ROOT / "checkpoints")

va_shared_cfg.patch_size = (1, 2, 2)

va_shared_cfg.enable_offload = False
