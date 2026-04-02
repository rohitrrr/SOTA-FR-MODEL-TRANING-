"""
SCaLE-FR (HYDRA Loss) R50 training on BUPT-CBFace-12 dataset.
Based on scale_fr_r50.py — only dataset fields changed.

Run:
    uv run torchrun --nproc_per_node=NUM_GPUS train_scale_fr.py \\
        --config_file ./configs/scale_fr_r50_bupt.py
"""

from configs.base import config as base_config
from easydict import EasyDict

config = EasyDict(base_config)

# ─── Identity ─────────────────────────────────────────────────────────────────
config.prefix       = "scale_fr-r50-bupt-cbface12"
config.head         = "arcface"

# ─── Backbone ─────────────────────────────────────────────────────────────────
config.model        = "iresnet"
config.depth        = "50"
config.embedding_size = 512
config.input_size   = [112, 112]

# ─── Dataset ──────────────────────────────────────────────────────────────────
# Point to the LMDB you created with:
#   python utils/bupt_cbface_to_lmdb.py \
#       --dataset_dir "/home/vmukti/Downloads/DATASET Experiment/BUPT-CBFace-12" \
#       --destination ./dataset --file_name bupt_cbface
config.train_source = "./dataset/bupt_cbface.lmdb"
config.num_ims      = 500000   # approximate; real length read from LMDB

# ─── Base Training ────────────────────────────────────────────────────────────
config.batch_size   = 128      # per GPU (RTX 3060/3090 12-16GB)
config.lr           = 0.1
config.optimizer    = "sgd"
config.momentum     = 0.9
config.weight_decay = 5e-4
config.epochs       = 20
config.fp16         = True
config.scheduler    = True
config.warmup_epoch = 1

# ─── ArcFace params ────────────────────────────────────────────────────────────
config.margin       = 0.5
config.scale        = 64.0

# ─── Validation ───────────────────────────────────────────────────────────────
config.val_list     = ["lfw"]   # add more if you have them
config.val_source   = "./test_set_package_5"
config.add_flip     = True
config.add_norm     = False

# ─── SCaLE-FR (HYDRA loss) ────────────────────────────────────────────────────
config.scale_fr = EasyDict()

# Queue
config.scale_fr.queue_size          = 16384   # 16K embeddings (adjust up if RAM allows)
config.scale_fr.momentum            = 0.999   # momentum encoder EMA

# Fisher projector
config.scale_fr.proj_dim            = 256
config.scale_fr.proj_refresh_steps  = 1000
config.scale_fr.proj_ema_alpha      = 0.95
config.scale_fr.proj_epsilon        = 1e-4
config.scale_fr.max_classes_for_cov = 500
config.scale_fr.max_samples_per_class = 20

# Loss weights — 3 main hyperparameters to tune
config.scale_fr.lambda_tail         = 0.3
config.scale_fr.lambda_pos          = 0.3
config.scale_fr.tail_margin         = 0.1

# Fixed (don't tune in v1)
config.scale_fr.beta                = 20.0
config.scale_fr.top_m               = 20
config.scale_fr.top_q               = 0.1
config.scale_fr.tau_p               = 0.5
config.scale_fr.ramp_steps          = 5000

# Schedule
config.scale_fr.warmup_epochs       = 5
config.scale_fr.activate_epoch      = 6

# Inference
config.scale_fr.use_projected_inference = True
