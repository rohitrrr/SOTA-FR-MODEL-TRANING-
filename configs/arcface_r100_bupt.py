"""
ArcFace R100 training on BUPT-CBFace-12 dataset.
Based on existing arcface_r100.py — only train_source and num_ims changed.

Run:
    uv run torchrun --nproc_per_node=NUM_GPUS train.py \\
        --config_file ./configs/arcface_r100_bupt.py
"""

from easydict import EasyDict

config = EasyDict()

config.prefix        = "arcface-r100-bupt-cbface12"
config.head          = "arcface"
config.input_size    = [112, 112]
config.embedding_size = 512
config.depth         = "100"
config.model         = "iresnet"
config.mode          = "ir"

# ─── Dataset ──────────────────────────────────────────────────────────────────
# Point to the LMDB you created with:
#   python utils/bupt_cbface_to_lmdb.py \
#       --dataset_dir "/home/vmukti/Downloads/DATASET Experiment/BUPT-CBFace-12" \
#       --destination ./dataset --file_name bupt_cbface
config.train_source  = "./dataset/bupt_cbface.lmdb"
config.num_ims       = 500000   # approximate; loader uses LMDB __len__ internally

# ─── Augmentation ─────────────────────────────────────────────────────────────
config.augment       = False
config.rand_erase    = False
config.mask          = None
config.label_map     = None
config.fixed_size    = None

# ─── Training ─────────────────────────────────────────────────────────────────
config.batch_size    = 128
config.lr            = 0.1
config.momentum      = 0.9
config.weight_decay  = 5e-4
config.epochs        = 20
config.fp16          = True
config.sample_rate   = 1.0
config.reduce_lr     = [8, 12, 15, 18]   # step-decay fallback if scheduler=False
config.scheduler     = True
config.warmup_epoch  = 1

# ─── ArcFace margin ───────────────────────────────────────────────────────────
config.margin        = 0.5
config.scale         = 64.0

# ─── Validation ───────────────────────────────────────────────────────────────
config.val_list      = ["lfw", "cfp_fp", "agedb_30"]
config.val_source    = "./test_set_package_5"
config.add_flip      = True
config.add_norm      = False

# ─── Misc ─────────────────────────────────────────────────────────────────────
config.workers       = 4
config.pin_memory    = True
config.frequency_log = 100
