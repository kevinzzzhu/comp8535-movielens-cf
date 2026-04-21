"""Unit tests for the proposed model and NMF baseline invariants.

Run with: uv run pytest tests/ -q
"""
from __future__ import annotations

import torch

from src.baselines import MFBias
from src.model import CFGatedOrdinal, GatedFusion, OrdinalHead


# ---- Fixtures / small helpers ----------------------------------------------

N_USERS, N_ITEMS = 20, 30
U_FEAT, I_FEAT = 24, 19
EMBED = 8


def _make_proposed(head: str = "ordinal", train_ratings: torch.Tensor | None = None) -> CFGatedOrdinal:
    return CFGatedOrdinal(
        n_users=N_USERS,
        n_items=N_ITEMS,
        user_feat_dim=U_FEAT,
        item_feat_dim=I_FEAT,
        embed_dim=EMBED,
        fusion="gated",
        head=head,
        train_ratings=train_ratings,
    )


def _sample_ratings(n: int = 1000, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    # MovieLens-like skew: more 4s and 3s than 1s
    probs = torch.tensor([0.06, 0.11, 0.27, 0.34, 0.22])
    idx = torch.multinomial(probs, n, replacement=True, generator=g)
    return (idx + 1).float()


# ---- Tests -----------------------------------------------------------------


def test_gated_fusion_output_shape_and_gate_in_unit_interval():
    """Gate should return (batch, embed) outputs with g in (0, 1)."""
    fuse = GatedFusion(embed_dim=EMBED, feat_dim=U_FEAT)
    emb = torch.randn(4, EMBED)
    feat = torch.randn(4, U_FEAT)
    out, g = fuse(emb, feat)
    assert out.shape == (4, EMBED)
    assert g.shape == (4, EMBED)
    assert (g > 0).all() and (g < 1).all()


def test_gated_fusion_zero_init_gate_gives_half():
    """With zero-init gate, sigmoid(0) = 0.5 exactly at step 0."""
    fuse = GatedFusion(embed_dim=EMBED, feat_dim=U_FEAT, zero_init_gate=True)
    emb = torch.randn(8, EMBED)
    feat = torch.randn(8, U_FEAT)
    _, g = fuse(emb, feat)
    assert torch.allclose(g, torch.full_like(g, 0.5), atol=1e-6)


def test_ordinal_probs_sum_to_one_and_thresholds_monotone():
    """class_probs rows must sum to 1; softplus guarantees strictly increasing thresholds."""
    head = OrdinalHead(n_classes=5)
    # Push deltas away from zero to exercise the softplus path
    with torch.no_grad():
        head.deltas.copy_(torch.randn_like(head.deltas))
    s = torch.randn(16)
    probs = head.class_probs(s)
    assert probs.shape == (16, 5)
    row_sums = probs.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    thr = head.thresholds()
    assert (thr[1:] > thr[:-1]).all(), f"non-monotone thresholds: {thr}"


def test_ordinal_init_from_ratings_matches_empirical_marginal():
    """At s=0, predicted P(r=k) should match empirical P(r=k) (Fix #4 invariant)."""
    ratings = _sample_ratings(n=2000)
    head = OrdinalHead(n_classes=5, train_ratings=ratings)
    with torch.no_grad():
        probs = head.class_probs(torch.zeros(1)).squeeze(0)
    cls = (ratings.round().long() - 1).clamp(0, 4)
    emp = torch.bincount(cls, minlength=5).float()
    emp = emp / emp.sum()
    assert torch.allclose(probs, emp, atol=5e-3), f"probs={probs} emp={emp}"


def test_nmf_project_clamps_factor_matrices_nonnegative():
    """MFBias(non_negative=True).project_() must leave user/item factors >= 0."""
    nmf = MFBias(N_USERS, N_ITEMS, embed_dim=EMBED, non_negative=True)
    with torch.no_grad():
        # Inject negatives to simulate a post-step state
        nmf.user_emb.weight.copy_(torch.randn_like(nmf.user_emb.weight))
        nmf.item_emb.weight.copy_(torch.randn_like(nmf.item_emb.weight))
    assert (nmf.user_emb.weight < 0).any(), "test setup: needed some negatives"
    nmf.project_()
    assert (nmf.user_emb.weight >= 0).all()
    assert (nmf.item_emb.weight >= 0).all()


def test_proposed_forward_and_loss_backward():
    """End-to-end: proposed model produces valid probs and a differentiable loss."""
    ratings = _sample_ratings(n=500)
    model = _make_proposed(head="ordinal", train_ratings=ratings)
    user_feat = torch.randn(N_USERS, U_FEAT)
    item_feat = torch.randn(N_ITEMS, I_FEAT)
    u = torch.randint(0, N_USERS, (16,))
    i = torch.randint(0, N_ITEMS, (16,))
    r = torch.randint(1, 6, (16,)).float()
    out = model(u, i, user_feat, item_feat)
    assert {"pred", "score", "probs", "gate_u", "gate_i"} <= set(out.keys())
    row_sums = out["probs"].sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    loss = model.loss(out, r)
    loss.backward()
    # At least one gradient must flow into the embeddings
    assert model.user_emb.weight.grad is not None
    assert model.user_emb.weight.grad.abs().sum() > 0
