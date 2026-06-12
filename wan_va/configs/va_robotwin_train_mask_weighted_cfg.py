from copy import deepcopy

from .va_robotwin_train_cfg import va_robotwin_train_cfg


va_robotwin_train_mask_weighted_cfg = deepcopy(va_robotwin_train_cfg)
va_robotwin_train_mask_weighted_cfg.__name__ = (
    'Config: VA robotwin train mask weighted latent loss'
)

va_robotwin_train_mask_weighted_cfg.enable_object_pred = False
va_robotwin_train_mask_weighted_cfg.enable_mask_weighted_loss = True
va_robotwin_train_mask_weighted_cfg.mask_foreground_weight = 10.0
