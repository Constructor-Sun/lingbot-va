from copy import deepcopy

from .va_robotwin_train_cfg import va_robotwin_train_cfg


va_robotwin_train_obj_pred_cfg = deepcopy(va_robotwin_train_cfg)
va_robotwin_train_obj_pred_cfg.__name__ = (
    'Config: VA robotwin train object future prediction'
)

va_robotwin_train_obj_pred_cfg.enable_object_pred = True
va_robotwin_train_obj_pred_cfg.enable_mask_weighted_loss = False
va_robotwin_train_obj_pred_cfg.object_pred_loss_weight = 0.1
va_robotwin_train_obj_pred_cfg.object_pred_k = 2
