"""
SCaLE-FR Configuration: R50 + MS1MV2
=====================================
Subspace-Calibrated Large-scale Extreme-risk Face Recognition

Base: ArcFace R50 on MS1MV2 (reproduce baseline first, then activate SCaLE-FR)

Training schedule:
  Phase 1 (epochs 0-5):   Pure ArcFace warmup, queue filling
  Phase 2 (epoch 6):      Activate Fisher projector + queue
  Phase 3 (epochs 6-8):   Linear ramp-in of tail + positive losses
  Phase 4 (epochs 8-20):  Full SCaLE-FR training
"""

from configs.base import config as base_config
from easydict import EasyDict

config = EasyDict(base_config)

# ─── Identity ─────────────────────────────────────────────────────────────────
config.prefix = "scale_fr-r50-ms1mv2"
config.head = "arcface"  # base classification loss

# ─── Backbone ─────────────────────────────────────────────────────────────────
config.model = "iresnet"
config.depth = "50"
config.embedding_size = 512
config.input_size = [112, 112]

# ─── Dataset ──────────────────────────────────────────────────────────────────
config.train_source = "./dataset/ms1mv2.lmdb"
config.num_ims = 5822653
config.num_classes = 85742

# ─── Base Training ────────────────────────────────────────────────────────────
config.batch_size = 128  # per GPU (RTX 3060 12GB)
config.lr = 0.1
config.optimizer = "sgd"
config.momentum = 0.9
config.weight_decay = 5e-4
config.epochs = 20
config.fp16 = True
config.scheduler = True
config.warmup_epoch = 1

# ─── ArcFace params ──────────────────────────────────────────────────────────
config.margin = 0.5
config.scale = 64.0

# ─── Validation ───────────────────────────────────────────────────────────────
config.val_list = ["lfw"]  # single val set to avoid OOM on 16GB RAM
config.val_source = "./test_set_package_5"
config.add_flip = True
config.add_norm = False

# ─── SCaLE-FR specific ───────────────────────────────────────────────────────
config.scale_fr = EasyDict()

# Queue
config.scale_fr.queue_size = 16384         # 16K embeddings (16GB RAM system)
config.scale_fr.momentum = 0.999          # EMA coefficient for momentum encoder

# Fisher projector
config.scale_fr.proj_dim = 256            # k: identity subspace dimension
config.scale_fr.proj_refresh_steps = 1000 # recompute projector every N steps
config.scale_fr.proj_ema_alpha = 0.95     # EMA blending for projector transition
config.scale_fr.proj_epsilon = 1e-4       # regularization for S_W inversion
config.scale_fr.max_classes_for_cov = 500 # balanced subsampling
config.scale_fr.max_samples_per_class = 20

# Losses (only 3 tunable hyperparameters)
config.scale_fr.lambda_tail = 0.3         # weight for tail ranking loss
config.scale_fr.lambda_pos = 0.3          # weight for hardest positive loss
config.scale_fr.tail_margin = 0.1         # m_t: margin in tail ranking

# Fixed hyperparameters (do NOT tune in v1)
config.scale_fr.beta = 20.0              # smooth-max temperature
config.scale_fr.top_m = 20              # top negatives per anchor
config.scale_fr.top_q = 0.1            # CVaR fraction (10%)
config.scale_fr.tau_p = 0.5            # positive compactness threshold
config.scale_fr.ramp_steps = 5000      # linear ramp-in steps after activation

# Schedule
config.scale_fr.warmup_epochs = 5       # pure ArcFace warmup before SCaLE-FR
config.scale_fr.activate_epoch = 6      # epoch to activate projector + losses

# Inference
config.scale_fr.use_projected_inference = True  # evaluate in Fisher space
