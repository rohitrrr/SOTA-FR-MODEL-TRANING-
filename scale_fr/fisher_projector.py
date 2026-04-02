"""
Fisher Projector for SCaLE-FR
=============================
Computes the identity subspace from within-class (S_W) and between-class (S_B)
covariance, using the regularized Fisher criterion:

    M = (S_W + εI)^{-1/2} · S_B · (S_W + εI)^{-1/2}

Top-k eigenvectors of M define the identity subspace U_k.
Projection: g(x) = Normalize(U_k^T · (S_W + εI)^{-1/2} · f(x))

Safety rails:
  1. Balanced class subsampling for covariance estimation
  2. EMA on covariances (NOT on the projection matrix) — preserves Fisher metric
  3. Eigenvalue clamping for numerical stability
  4. Logging of class counts during refresh

Refreshed every N steps (default 1000), NOT every batch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


class FisherProjector(nn.Module):
    """
    Computes and maintains the Fisher identity subspace.

    The key invariant: the projection matrix P = U_k^T · (S_W + εI)^{-1/2}
    is ALWAYS recomputed exactly from the (EMA-smoothed) covariances.
    We never blend or orthonormalize P directly, because that would
    destroy the whitening/metric structure that defines the Fisher criterion.

    Args:
        embedding_dim: Dimension of input embeddings. Default 512.
        proj_dim: Dimension of projected space (k). Default 256.
        cov_ema_alpha: EMA coefficient for covariance smoothing. Default 0.8.
        epsilon: Regularization for S_W inversion. Default 1e-4.
        max_classes_for_cov: Max classes to sample for covariance. Default 500.
        max_samples_per_class: Max samples per class for balance. Default 20.
    """

    def __init__(self, embedding_dim=512, proj_dim=256, cov_ema_alpha=0.8,
                 epsilon=1e-4, max_classes_for_cov=500,
                 max_samples_per_class=20):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.proj_dim = proj_dim
        self.cov_ema_alpha = cov_ema_alpha
        self.epsilon = epsilon
        self.max_classes = max_classes_for_cov
        self.max_samples_per_class = max_samples_per_class

        # The projection matrix: P = U_k^T · (S_W + εI)^{-1/2}
        # Shape: (proj_dim, embedding_dim)
        # Initialized as truncated identity (passthrough until first refresh)
        self.register_buffer(
            'projection', torch.eye(embedding_dim)[:proj_dim].clone())

        # Track whether projector has been initialized from real data
        self.register_buffer(
            'initialized', torch.tensor(False, dtype=torch.bool))

        # Diagnostics: eigenvalue spectrum of M (for monitoring)
        self.register_buffer(
            'fisher_eigenvalues', torch.zeros(proj_dim))

        # EMA-smoothed covariances — these are what get blended, not P
        self.register_buffer(
            'S_W_ema', torch.zeros(embedding_dim, embedding_dim))
        self.register_buffer(
            'S_B_ema', torch.zeros(embedding_dim, embedding_dim))
        self.register_buffer(
            'cov_update_count', torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def refresh(self, queue_emb, queue_lbl):
        """
        Recompute Fisher projector from queue embeddings.
        Called every N training steps (e.g., 1000).

        Strategy:
          1. Compute fresh S_W, S_B from balanced class subsample of queue
          2. EMA-blend with running covariances (smooth signal, not projection)
          3. Recompute exact Fisher projection P from smoothed covariances

        Args:
            queue_emb: (N, D) tensor of L2-normalized embeddings from queue.
            queue_lbl: (N,) tensor of class labels.

        Returns:
            dict with refresh diagnostics, or None if refresh was skipped.
        """
        if queue_emb is None or queue_lbl is None:
            return None
        if queue_emb.shape[0] < 100:
            return None

        device = queue_emb.device
        D = self.embedding_dim

        # ─── Step 1: Balanced class subsampling ───────────────────────
        unique_labels = queue_lbl.unique()
        n_classes_in_queue = len(unique_labels)
        if n_classes_in_queue < 10:
            logging.warning(
                f"[Fisher] Only {n_classes_in_queue} classes in queue, "
                f"need >=10. Skipping refresh.")
            return None

        # Subsample classes if too many
        if n_classes_in_queue > self.max_classes:
            perm = torch.randperm(n_classes_in_queue,
                                  device=device)[:self.max_classes]
            selected_labels = unique_labels[perm]
        else:
            selected_labels = unique_labels

        # Collect per-class data with balanced sampling
        class_embeddings = {}
        n_skipped_too_few = 0
        for lbl in selected_labels:
            mask = queue_lbl == lbl
            embs = queue_emb[mask]
            n = embs.shape[0]
            if n < 2:
                n_skipped_too_few += 1
                continue
            if n > self.max_samples_per_class:
                idx = torch.randperm(n, device=device)[:self.max_samples_per_class]
                embs = embs[idx]
            class_embeddings[lbl.item()] = embs

        n_classes_used = len(class_embeddings)
        if n_classes_used < 10:
            logging.warning(
                f"[Fisher] Only {n_classes_used} classes with >=2 samples "
                f"(skipped {n_skipped_too_few}). Skipping refresh.")
            return None

        # ─── Step 2: Compute fresh S_W and S_B ───────────────────────
        all_embs = torch.cat(list(class_embeddings.values()), dim=0)
        global_mean = all_embs.mean(dim=0)

        S_W_fresh = torch.zeros(D, D, device=device, dtype=torch.float64)
        S_B_fresh = torch.zeros(D, D, device=device, dtype=torch.float64)
        total_n = 0

        for lbl, embs in class_embeddings.items():
            embs_d = embs.double()
            n_c = embs_d.shape[0]
            mu_c = embs_d.mean(dim=0)

            centered = embs_d - mu_c.unsqueeze(0)
            S_W_fresh += centered.T @ centered

            diff = (mu_c - global_mean.double()).unsqueeze(1)
            S_B_fresh += n_c * (diff @ diff.T)

            total_n += n_c

        S_W_fresh /= total_n
        S_B_fresh /= total_n

        # ─── Step 3: EMA blend covariances (NOT the projection) ──────
        if self.cov_update_count > 0:
            alpha = self.cov_ema_alpha
            S_W_smooth = alpha * self.S_W_ema.double() + (1 - alpha) * S_W_fresh
            S_B_smooth = alpha * self.S_B_ema.double() + (1 - alpha) * S_B_fresh
        else:
            S_W_smooth = S_W_fresh
            S_B_smooth = S_B_fresh

        # Store smoothed covariances
        self.S_W_ema.copy_(S_W_smooth.float())
        self.S_B_ema.copy_(S_B_smooth.float())
        self.cov_update_count += 1

        # ─── Step 4: Compute exact Fisher projection from smoothed covs ──
        eps_I = self.epsilon * torch.eye(D, device=device, dtype=torch.float64)
        S_W_reg = S_W_smooth + eps_I

        # Eigendecompose S_W_reg for stable inverse square root
        eigvals_W, eigvecs_W = torch.linalg.eigh(S_W_reg)
        # Hard clamp eigenvalues (not just additive ε)
        eigvals_W = eigvals_W.clamp(min=self.epsilon)
        inv_sqrt_eigvals = 1.0 / torch.sqrt(eigvals_W)
        S_W_inv_sqrt = eigvecs_W @ torch.diag(inv_sqrt_eigvals) @ eigvecs_W.T

        # Fisher matrix M = S_W^{-1/2} · S_B · S_W^{-1/2}
        M = S_W_inv_sqrt @ S_B_smooth @ S_W_inv_sqrt
        M = (M + M.T) / 2.0  # symmetrize for numerical safety

        # Top-k eigenvectors of M
        eigvals_M, eigvecs_M = torch.linalg.eigh(M)
        idx = torch.argsort(eigvals_M, descending=True)
        eigvals_M = eigvals_M[idx]
        eigvecs_M = eigvecs_M[:, idx]

        U_k = eigvecs_M[:, :self.proj_dim]  # (D, k)

        self.fisher_eigenvalues.copy_(
            eigvals_M[:self.proj_dim].float().clamp(min=0))

        # ─── Step 5: Exact Fisher projection P = U_k^T · S_W^{-1/2} ──
        # No EMA blending, no SVD orthonormalization on P.
        # The metric structure comes entirely from the smoothed covariances.
        P_exact = (U_k.T @ S_W_inv_sqrt).float()  # (k, D)
        self.projection.copy_(P_exact)
        self.initialized.fill_(True)

        # ─── Diagnostics ──────────────────────────────────────────────
        diag = {
            'n_classes_in_queue': int(n_classes_in_queue),
            'n_classes_selected': int(len(selected_labels)),
            'n_classes_used': n_classes_used,
            'n_skipped_too_few': n_skipped_too_few,
            'total_samples_used': total_n,
            'cov_update_count': int(self.cov_update_count.item()),
            'fisher_eig_max': float(eigvals_M[0]),
            'fisher_eig_k': float(eigvals_M[min(self.proj_dim - 1,
                                                  len(eigvals_M) - 1)]),
            'S_W_trace': float(S_W_smooth.trace()),
            'S_B_trace': float(S_B_smooth.trace()),
            'BW_ratio': float(S_B_smooth.trace() / (S_W_smooth.trace() + 1e-10)),
        }

        return diag

    def project(self, embeddings):
        """
        Project embeddings into Fisher identity subspace.

        Args:
            embeddings: (B, D) tensor, L2-normalized.

        Returns:
            projected: (B, k) tensor, L2-normalized in projected space.
        """
        proj = F.linear(embeddings, self.projection)  # (B, k)
        proj = F.normalize(proj, dim=1)
        return proj

    def get_diagnostics(self):
        """Return diagnostic info for logging."""
        ev = self.fisher_eigenvalues
        return {
            'fisher_eig_max': ev[0].item() if self.initialized else 0.0,
            'fisher_eig_min': ev[-1].item() if self.initialized else 0.0,
            'fisher_eig_ratio': (ev[0] / (ev[-1] + 1e-10)).item()
                if self.initialized else 0.0,
            'projector_initialized': self.initialized.item(),
            'cov_updates': self.cov_update_count.item(),
        }
