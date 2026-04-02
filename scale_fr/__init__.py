# SCaLE-FR: Subspace-Calibrated Large-scale Extreme-risk Face Recognition
# NeurIPS 2025 submission target
#
# Components:
#   memory_bank.py       - Momentum encoder + embedding queue (65K-131K)
#   fisher_projector.py  - EMA Fisher projector (S_W, S_B → U_k)
#   losses.py            - TailRankingLoss + HardestPositiveLoss

from .memory_bank import MomentumBank
from .fisher_projector import FisherProjector
from .losses import TailRankingLoss, HardestPositiveLoss, ScaleFRLoss
