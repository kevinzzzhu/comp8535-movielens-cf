"""Matrix factorisation with gated auxiliary fusion and an ordinal-regression head.

Toggle `fusion` and `head` at construction time to run ablations.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusion(nn.Module):
    """p' = (1-g)*p + g*ReLU(W v), g = sigmoid(Wg [p || ReLU(W v)])."""

    def __init__(self, embed_dim: int, feat_dim: int, zero_init_gate: bool = True):
        super().__init__()
        self.proj = nn.Linear(feat_dim, embed_dim, bias=True)
        self.gate = nn.Linear(2 * embed_dim, embed_dim, bias=True)
        if zero_init_gate:
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)

    def forward(self, emb: torch.Tensor, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        aux = F.relu(self.proj(feat))
        g = torch.sigmoid(self.gate(torch.cat([emb, aux], dim=-1)))
        out = (1.0 - g) * emb + g * aux
        return out, g


class AdditiveFusion(nn.Module):
    """Previous baseline-style fusion: p' = p + ReLU(W v). Included for ablation only."""

    def __init__(self, embed_dim: int, feat_dim: int):
        super().__init__()
        self.proj = nn.Linear(feat_dim, embed_dim, bias=True)

    def forward(self, emb: torch.Tensor, feat: torch.Tensor):
        return emb + F.relu(self.proj(feat)), None


class OrdinalHead(nn.Module):
    """Cumulative-link ordinal head with monotone thresholds via softplus."""

    def __init__(self, n_classes: int = 5):
        super().__init__()
        self.n_classes = n_classes
        self.theta1 = nn.Parameter(torch.tensor(-2.0))
        self.deltas = nn.Parameter(torch.zeros(n_classes - 2))  # softplus -> positive gaps

    def thresholds(self) -> torch.Tensor:
        gaps = F.softplus(self.deltas)
        return torch.cat([self.theta1.unsqueeze(0), self.theta1 + torch.cumsum(gaps, dim=0)])

    def class_probs(self, s: torch.Tensor) -> torch.Tensor:
        """Return (B, n_classes) probabilities."""
        thr = self.thresholds()  # (n_classes-1,)
        # P(r <= k) = sigmoid(theta_k - s)
        cum = torch.sigmoid(thr.unsqueeze(0) - s.unsqueeze(-1))
        # P(r = k) = diff of cumulative, padded with 0 and 1
        zeros = torch.zeros_like(cum[..., :1])
        ones = torch.ones_like(cum[..., :1])
        cdf = torch.cat([zeros, cum, ones], dim=-1)
        return cdf[..., 1:] - cdf[..., :-1]

    def expected(self, s: torch.Tensor) -> torch.Tensor:
        probs = self.class_probs(s)
        ks = torch.arange(1, self.n_classes + 1, device=s.device, dtype=s.dtype)
        return (probs * ks).sum(dim=-1)


class CFGatedOrdinal(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        user_feat_dim: int,
        item_feat_dim: int,
        embed_dim: int = 128,
        fusion: str = "gated",  # {"none", "additive", "gated"}
        head: str = "ordinal",  # {"sigmoid", "ordinal"}
        n_classes: int = 5,
    ):
        super().__init__()
        self.fusion_kind = fusion
        self.head_kind = head

        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

        if fusion == "gated":
            self.fuse_u = GatedFusion(embed_dim, user_feat_dim)
            self.fuse_i = GatedFusion(embed_dim, item_feat_dim)
        elif fusion == "additive":
            self.fuse_u = AdditiveFusion(embed_dim, user_feat_dim)
            self.fuse_i = AdditiveFusion(embed_dim, item_feat_dim)
        elif fusion == "none":
            self.fuse_u = None
            self.fuse_i = None
        else:
            raise ValueError(f"unknown fusion: {fusion}")

        if head == "ordinal":
            self.head = OrdinalHead(n_classes=n_classes)
        elif head != "sigmoid":
            raise ValueError(f"unknown head: {head}")

    def forward(self, u_idx, i_idx, user_feat, item_feat):
        p = self.user_emb(u_idx)
        q = self.item_emb(i_idx)
        g_u = g_i = None
        if self.fuse_u is not None:
            vu = user_feat[u_idx]
            vi = item_feat[i_idx]
            p, g_u = self.fuse_u(p, vu)
            q, g_i = self.fuse_i(q, vi)

        s = (p * q).sum(dim=-1) + self.user_bias(u_idx).squeeze(-1) + self.item_bias(i_idx).squeeze(-1)

        if self.head_kind == "sigmoid":
            pred = torch.sigmoid(s) * 4.0 + 1.0
            return {"pred": pred, "score": s, "gate_u": g_u, "gate_i": g_i}

        probs = self.head.class_probs(s)
        pred = self.head.expected(s)
        return {"pred": pred, "score": s, "probs": probs, "gate_u": g_u, "gate_i": g_i}

    def loss(self, out: dict, target: torch.Tensor) -> torch.Tensor:
        if self.head_kind == "sigmoid":
            return F.mse_loss(out["pred"], target)
        # Ordinal NLL; targets are integer ratings 1..5 mapped to 0..4
        cls = (target.round().long() - 1).clamp(0, 4)
        log_p = torch.log(out["probs"] + 1e-12)
        return F.nll_loss(log_p, cls)
