# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_robotwin_cfg import va_robotwin_cfg
import os
from pathlib import Path


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

va_robotwin_train_cfg = EasyDict(__name__='Config: VA robotwin train')
va_robotwin_train_cfg.update(va_robotwin_cfg)

va_robotwin_train_cfg.dataset_path = os.environ.get(
    "LINGBOT_DATASET_PATH",
    str(_CF_ROOT / "robust_wam" / "data" / "data_w_mask_clean_large_le320"),
)
va_robotwin_train_cfg.empty_emb_path = os.path.join(va_robotwin_train_cfg.dataset_path, 'empty_emb.pt')
va_robotwin_train_cfg.enable_wandb = True
va_robotwin_train_cfg.load_worker = 16
va_robotwin_train_cfg.save_interval = 200
va_robotwin_train_cfg.gc_interval = 50
va_robotwin_train_cfg.cfg_prob = 0.1

# Training parameters
va_robotwin_train_cfg.learning_rate = 1e-5
va_robotwin_train_cfg.beta1 = 0.9
va_robotwin_train_cfg.beta2 = 0.95
va_robotwin_train_cfg.weight_decay = 0.1
va_robotwin_train_cfg.warmup_steps = 10
va_robotwin_train_cfg.batch_size = 1 
va_robotwin_train_cfg.gradient_accumulation_steps = 1
va_robotwin_train_cfg.num_steps = 5000

# Object Future Prediction
va_robotwin_train_cfg.enable_object_pred = False

# Token-level mask-weighted latent reconstruction baseline
va_robotwin_train_cfg.enable_mask_weighted_loss = False
