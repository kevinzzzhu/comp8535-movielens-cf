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
    """Additive fusion: p' = p + ReLU(W v). Included for ablation only."""

    def __init__(self, embed_dim: int, feat_dim: int):
        super().__init__()
        self.proj = nn.Linear(feat_dim, embed_dim, bias=True)

    def forward(self, emb: torch.Tensor, feat: torch.Tensor):
        return emb + F.relu(self.proj(feat)), None


class OrdinalHead(nn.Module):
    """Cumulative-link ordinal head with monotone thresholds via softplus.

    Thresholds are stored as (theta1, deltas) where deltas are mapped through softplus to
    enforce theta_k < theta_{k+1}. Default initialisation places all thresholds at -2,
    which puts most probability mass on class 1 at init — poor gradient flow early. Passing
    `train_ratings` initialises the thresholds from the empirical rating distribution so that
    at s=0 the predicted marginal matches the data.

    `link ∈ {"logit", "probit"}` selects the link function. Logit is our default; probit is
    the parameterisation used by OPRFM (Zaman & Jana, 2025) and is included for ablation.
    """

    def __init__(self, n_classes: int = 5, train_ratings: torch.Tensor | None = None,
                 link: str = "logit"):
        super().__init__()
        if link not in ("logit", "probit"):
            raise ValueError(f"unknown link {link!r}; expected 'logit' or 'probit'")
        self.n_classes = n_classes
        self.link = link
        if train_ratings is not None:
            theta1, deltas = self._thresholds_from_ratings(train_ratings, n_classes, link)
            self.theta1 = nn.Parameter(theta1)
            self.deltas = nn.Parameter(deltas)
        else:
            self.theta1 = nn.Parameter(torch.tensor(-2.0))
            self.deltas = nn.Parameter(torch.zeros(n_classes - 2))

    @staticmethod
    def _thresholds_from_ratings(ratings: torch.Tensor, n_classes: int, link: str
                                  ) -> tuple[torch.Tensor, torch.Tensor]:
        """Derive (theta1, deltas_pre_softplus) so cdf(theta_k - 0) matches P(r<=k)."""
        cls = (ratings.round().long() - 1).clamp(0, n_classes - 1)
        counts = torch.bincount(cls, minlength=n_classes).float()
        probs = counts / counts.sum()
        cum = torch.cumsum(probs, dim=0)[:-1]              # P(r<=k) for k=1..K-1
        eps = 1e-3
        cum = cum.clamp(eps, 1.0 - eps)
        if link == "logit":
            thr = torch.logit(cum)                          # theta_k = logit(P(r<=k))
        else:  # probit: theta_k = Phi^{-1}(P(r<=k)) = sqrt(2) * erfinv(2p - 1)
            thr = torch.erfinv(2 * cum - 1) * (2.0 ** 0.5)
        theta1 = thr[0].detach().clone()
        gaps = (thr[1:] - thr[:-1]).clamp(min=1e-3)         # strictly positive gaps
        # invert softplus: deltas_raw such that softplus(deltas_raw) = gaps
        deltas = torch.log(torch.expm1(gaps)).detach().clone()
        return theta1, deltas

    def thresholds(self) -> torch.Tensor:
        gaps = F.softplus(self.deltas)
        return torch.cat([self.theta1.unsqueeze(0), self.theta1 + torch.cumsum(gaps, dim=0)])

    def _cdf(self, x: torch.Tensor) -> torch.Tensor:
        """Cumulative distribution of the chosen link's latent noise."""
        if self.link == "logit":
            return torch.sigmoid(x)
        # Probit: standard normal CDF via erf.
        return 0.5 * (1.0 + torch.erf(x / (2.0 ** 0.5)))

    def class_probs(self, s: torch.Tensor) -> torch.Tensor:
        """Return (B, n_classes) probabilities."""
        thr = self.thresholds()  # (n_classes-1,)
        cum = self._cdf(thr.unsqueeze(0) - s.unsqueeze(-1))
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
        train_ratings: torch.Tensor | None = None,
        ordinal_link: str = "logit",  # {"logit", "probit"} for ordinal head
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
            self.head = OrdinalHead(n_classes=n_classes, train_ratings=train_ratings,
                                    link=ordinal_link)
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
