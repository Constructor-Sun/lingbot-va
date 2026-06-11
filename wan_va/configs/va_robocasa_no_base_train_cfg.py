from easydict import EasyDict
from .va_robocasa_no_base_cfg import va_robocasa_no_base_cfg
import os

va_robocasa_no_base_train_cfg = EasyDict(__name__='Config: VA robocasa no base train')
va_robocasa_no_base_train_cfg.update(va_robocasa_no_base_cfg)

va_robocasa_no_base_train_cfg.dataset_path = '/data1/liu/exp/robocasa/datasets/training_no_base/atomic'
va_robocasa_no_base_train_cfg.empty_emb_path = os.path.join(
    va_robocasa_no_base_train_cfg.dataset_path, 'empty_emb.pt'
)
va_robocasa_no_base_train_cfg.enable_wandb = True
va_robocasa_no_base_train_cfg.load_worker = 16
va_robocasa_no_base_train_cfg.save_interval = 200
va_robocasa_no_base_train_cfg.gc_interval = 50
va_robocasa_no_base_train_cfg.cfg_prob = 0.1

va_robocasa_no_base_train_cfg.learning_rate = 1e-5
va_robocasa_no_base_train_cfg.beta1 = 0.9
va_robocasa_no_base_train_cfg.beta2 = 0.95
va_robocasa_no_base_train_cfg.weight_decay = 0.1
va_robocasa_no_base_train_cfg.warmup_steps = 10
va_robocasa_no_base_train_cfg.batch_size = 1
va_robocasa_no_base_train_cfg.gradient_accumulation_steps = 8
va_robocasa_no_base_train_cfg.num_steps = 5000

# Object Future Prediction
va_robocasa_no_base_train_cfg.enable_object_pred = False
va_robocasa_no_base_train_cfg.object_pred_loss_weight = 0.1
va_robocasa_no_base_train_cfg.object_pred_k = 2

# Token-level mask-weighted latent reconstruction baseline
va_robocasa_no_base_train_cfg.enable_mask_weighted_loss = False
