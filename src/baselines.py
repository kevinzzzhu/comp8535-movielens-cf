"""SVD, MF, and NMF baselines for head-to-head RMSE comparison."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


class ProjectableModel(nn.Module):
    """Interface marker: models with a non-trivial `project_()` are clamped after each opt.step()."""

    def project_(self) -> None:  # pragma: no cover — default no-op
        return None


def svd_baseline(train, n_users: int, n_items: int, rank: int = 20):
    """Truncated SVD on mean-filled rating matrix; returns predict(u,i) callable."""
    u = train.user_idx.numpy()
    i = train.item_idx.numpy()
    r = train.rating.numpy()
    global_mean = float(r.mean())

    R = np.full((n_users, n_items), global_mean, dtype=np.float32)
    R[u, i] = r
    sparse = csr_matrix(R - global_mean)
    U, s, Vt = svds(sparse, k=rank)
    S = np.diag(s)
    approx = U @ S @ Vt + global_mean

    def predict(uu, ii):
        return np.clip(approx[uu, ii], 1.0, 5.0)

    return predict


class MFBias(ProjectableModel):
    """MF with biases. Non-negativity is enforced by projecting embeddings to [0, +inf) after each
    optimizer step — the classical NMF framework of Lee & Seung (2000). Only the factor matrices
    are projected; biases remain unconstrained, matching standard NMF-with-biases practice."""

    def __init__(self, n_users: int, n_items: int, embed_dim: int = 128, non_negative: bool = False):
        super().__init__()
        self.non_negative = non_negative
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        std = 0.01
        nn.init.normal_(self.user_emb.weight, std=std)
        nn.init.normal_(self.item_emb.weight, std=std)
        if non_negative:
            with torch.no_grad():
                self.user_emb.weight.abs_()
                self.item_emb.weight.abs_()
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, u_idx, i_idx):
        p = self.user_emb(u_idx)
        q = self.item_emb(i_idx)
        s = (p * q).sum(dim=-1) + self.user_bias(u_idx).squeeze(-1) + self.item_bias(i_idx).squeeze(-1)
        return torch.sigmoid(s) * 4.0 + 1.0

    def loss(self, pred, target):
        return F.mse_loss(pred, target)

    @torch.no_grad()
    def project_(self) -> None:
        if self.non_negative:
            self.user_emb.weight.clamp_(min=0.0)
            self.item_emb.weight.clamp_(min=0.0)
