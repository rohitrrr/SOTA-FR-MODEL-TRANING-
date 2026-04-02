"""
Momentum Encoder + Embedding Queue for SCaLE-FR
================================================
Provides a large pool of recent embeddings (65K-131K) so that:
  - Tail ranking loss sees gallery-scale hard negatives
  - Fisher projector has enough samples for stable S_W / S_B estimation

Architecture follows MoCo v2 pattern:
  - Momentum encoder: EMA copy of backbone, no gradients
  - Queue: FIFO ring buffer of (embedding, label) pairs
  - Online encoder: the actual trainable backbone

Key design decisions:
  - Queue stores L2-normalized embeddings (same as inference)
  - Labels stored alongside for class-aware Fisher estimation
  - Momentum coefficient α = 0.999 (slow teacher)
  - Queue updated every forward pass with momentum-encoded batch
"""

import torch
import torch.nn as nn
import copy


class MomentumBank(nn.Module):
    """
    Momentum encoder + fixed-size embedding queue.

    Args:
        backbone: The online backbone network (IResNet).
        queue_size: Number of embeddings to store. Default 65536.
        embedding_dim: Dimension of embeddings. Default 512.
        momentum: EMA coefficient for momentum encoder. Default 0.999.
    """

    def __init__(self, backbone, queue_size=65536, embedding_dim=512,
                 momentum=0.999):
        super().__init__()
        self.queue_size = queue_size
        self.embedding_dim = embedding_dim
        self.momentum = momentum

        # Momentum encoder: deep copy of backbone, frozen
        self.encoder_m = copy.deepcopy(backbone)
        for p in self.encoder_m.parameters():
            p.requires_grad = False

        # Queue: ring buffer of normalized embeddings + labels
        # Stored as buffers so they survive .cuda() and state_dict
        self.register_buffer(
            'queue_emb', torch.randn(queue_size, embedding_dim))
        self.register_buffer(
            'queue_lbl', torch.full((queue_size,), -1, dtype=torch.long))
        self.register_buffer(
            'queue_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer(
            'queue_filled', torch.zeros(1, dtype=torch.long))

        # Normalize initial random queue
        self.queue_emb = nn.functional.normalize(self.queue_emb, dim=1)

    @torch.no_grad()
    def update_momentum_encoder(self, backbone):
        """EMA update: θ_m ← α·θ_m + (1-α)·θ_online"""
        for p_m, p_o in zip(self.encoder_m.parameters(),
                            backbone.parameters()):
            p_m.data.mul_(self.momentum).add_(
                p_o.data, alpha=1.0 - self.momentum)

    @torch.no_grad()
    def encode_and_enqueue(self, images, labels):
        """
        Forward through momentum encoder, enqueue results.

        Args:
            images: (B, 3, 112, 112) tensor, already normalized.
            labels: (B,) tensor of class labels.

        Returns:
            emb_m: (B, 512) momentum-encoded embeddings (for reference).
        """
        self.encoder_m.eval()

        # Forward through momentum encoder
        # IResNet returns (logits, bottleneck) in train mode,
        # but we only need the embedding
        with torch.no_grad():
            # Try train-mode signature first (logits, embedding)
            out = self.encoder_m(images)
            if isinstance(out, tuple):
                emb_m = out[1]  # bottleneck_embedding
            else:
                emb_m = out

            emb_m = nn.functional.normalize(emb_m, dim=1)

        # Gather across GPUs if distributed
        if torch.distributed.is_initialized():
            emb_gather = self._gather_across_gpus(emb_m)
            lbl_gather = self._gather_across_gpus(labels)
        else:
            emb_gather = emb_m
            lbl_gather = labels

        batch_size = emb_gather.shape[0]
        ptr = int(self.queue_ptr)

        # Enqueue: wrap around if exceeding queue size
        if ptr + batch_size <= self.queue_size:
            self.queue_emb[ptr:ptr + batch_size] = emb_gather
            self.queue_lbl[ptr:ptr + batch_size] = lbl_gather
        else:
            # Wrap around
            overflow = (ptr + batch_size) - self.queue_size
            first_part = self.queue_size - ptr
            self.queue_emb[ptr:] = emb_gather[:first_part]
            self.queue_lbl[ptr:] = lbl_gather[:first_part]
            self.queue_emb[:overflow] = emb_gather[first_part:]
            self.queue_lbl[:overflow] = lbl_gather[first_part:]

        # Advance pointer
        new_ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = new_ptr
        new_filled = min(int(self.queue_filled.item()) + batch_size,
                         self.queue_size)
        self.queue_filled[0] = new_filled

        return emb_m

    def get_queue(self):
        """
        Returns the valid portion of the queue.

        Returns:
            emb: (N, 512) tensor of queue embeddings.
            lbl: (N,) tensor of queue labels.
        """
        n = int(self.queue_filled.item())
        if n == 0:
            return None, None
        # Return a detached copy so queue isn't in the compute graph
        return self.queue_emb[:n].detach(), self.queue_lbl[:n].detach()

    def get_queue_size_filled(self):
        """How many valid entries are in the queue."""
        return int(self.queue_filled.item())

    @torch.no_grad()
    def _gather_across_gpus(self, tensor):
        """All-gather tensors from all GPUs, concatenated on dim 0."""
        world_size = torch.distributed.get_world_size()
        if world_size == 1:
            return tensor

        # All tensors must have same shape except dim 0
        tensors_gather = [torch.zeros_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(tensors_gather, tensor)
        return torch.cat(tensors_gather, dim=0)

    def state_dict_extra(self):
        """Extra state for checkpointing (queue contents)."""
        return {
            'queue_emb': self.queue_emb,
            'queue_lbl': self.queue_lbl,
            'queue_ptr': self.queue_ptr,
            'queue_filled': self.queue_filled,
        }

    def load_state_dict_extra(self, state):
        """Restore queue from checkpoint."""
        self.queue_emb.copy_(state['queue_emb'])
        self.queue_lbl.copy_(state['queue_lbl'])
        self.queue_ptr.copy_(state['queue_ptr'])
        self.queue_filled.copy_(state['queue_filled'])
