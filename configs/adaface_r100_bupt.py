"""
AdaFace R100 training on BUPT-CBFace-12 dataset.
Based on adaface_r100.py — only train_source changed.

Note: AdaFace uses add_norm=True and add_flip=False (unlike ArcFace).
      margin is 0.4 (not 0.5) as per AdaFace paper.

Run:
    uv run torchrun --nproc_per_node=NUM_GPUS train.py \\
        --config_file ./configs/adaface_r100_bupt.py

Before training, generate the LMDB with:
    uv run python utils/bupt_cbface_to_lmdb.py \\
        --dataset_dir "/home/vmukti/Downloads/DATASET Experiment/BUPT-CBFace-12" \\
        --destination ./dataset \\
        --file_name bupt_cbface \\
        --workers 8
"""

from easydict import EasyDict

config = EasyDict()

config.prefix         = "adaface-r100-bupt-cbface12"
config.head           = "adaface"
config.input_size     = [112, 112]
config.embedding_size = 512
config.depth          = "100"
config.model          = "iresnet"
config.mode           = "ir"

# ─── Dataset ──────────────────────────────────────────────────────────────────
config.train_source   = "./dataset/bupt_cbface.lmdb"
config.num_ims        = 500000    # approximate; real value read from LMDB

# ─── Augmentation ─────────────────────────────────────────────────────────────
config.mask           = None
config.label_map      = None
config.augment        = False
config.rand_erase     = False
config.fixed_size     = None

# ─── Training ─────────────────────────────────────────────────────────────────
config.batch_size     = 256
config.lr             = 0.1
config.momentum       = 0.9
config.weight_decay   = 5e-4
config.epochs         = 20
config.fp16           = True
config.sample_rate    = 1.0
config.reduce_lr      = [8, 12, 15, 18]
config.scheduler      = True
config.warmup_epoch   = 1

# ─── AdaFace margin (different from ArcFace — do not change) ──────────────────
config.margin         = 0.4

# ─── Validation ─────────────────────────────────────────────────────────────
config.val_list       = ["lfw", "cfp_fp", "agedb_30"]
config.val_source     = "./test_set_package_5"
config.add_flip       = False   # AdaFace: flip during feature extraction OFF
config.add_norm       = True    # AdaFace: L2-normalize embeddings ON

# ─── Misc ─────────────────────────────────────────────────────────────────────
config.workers        = 4
config.pin_memory     = True
config.frequency_log  = 100
