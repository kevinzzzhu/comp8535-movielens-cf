"""Calibration reliability diagram for the gated+sigmoid and gated+ordinal heads.

Trains two models on u1 (gated+sigmoid and gated+ordinal). For each test point
we derive a five-class probability distribution P(r=k|...), bucket the
(predicted-probability, observed-indicator) pairs, and plot a reliability
curve along with the Expected Calibration Error (ECE).

The two heads emit different things natively, so we put them on a level
playing field:

  ordinal:  P(r=k|s) is read directly from the cumulative-link head.
  sigmoid:  the head emits a single point estimate y_hat = sigma(s) * 4 + 1
            in [1, 5]. We turn it into a discrete distribution by placing a
            Gaussian kernel of bandwidth sigma=1 (in rating units) at y_hat
            and re-normalising over k in {1, 2, 3, 4, 5}.

Both distributions are then evaluated on the same definition of calibration:
for predictions in bin (p_low, p_high], how often was the corresponding class
the true class? Perfect calibration sits on the diagonal y=x.

Outputs:
    results/<date>_reliability/
        reliability_curves.png   2-panel reliability diagrams (sigmoid | ordinal)
        ece_summary.csv          ECE per head + per-class ECE
        results.json             metadata
        config_snapshot.yaml

Usage:
    PYTHONPATH=. uv run python scripts/run_reliability.py
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.dataset import build_metadata, load_split_with_val
from src.model import CFGatedOrdinal
from src.train import TrainConfig, set_seed, train_model

CLASSES = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
N_BINS = 10
SIGMA_SIGMOID = 1.0  # bandwidth for Gaussian kernel applied to sigmoid yhat


@torch.no_grad()
def _ordinal_class_probs(model, meta, ds) -> np.ndarray:
    """Return (n_test, 5) class probabilities from the ordinal head."""
    model.eval()
    out = model(ds.user_idx, ds.item_idx, meta.user_features, meta.item_features)
    probs = out["probs"].cpu().numpy()  # (n_test, 5)
    # Numerical safety: clip and renormalise.
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


@torch.no_grad()
def _sigmoid_class_probs(model, meta, ds, sigma: float = SIGMA_SIGMOID) -> np.ndarray:
    """Convert sigmoid-head point predictions to a 5-class distribution by
    placing a Gaussian kernel of bandwidth sigma at y_hat and normalising over
    classes {1, ..., 5}."""
    model.eval()
    out = model(ds.user_idx, ds.item_idx, meta.user_features, meta.item_features)
    yhat = out["pred"].cpu().numpy()  # (n_test,) in [1, 5]
    diffs = yhat[:, None] - CLASSES[None, :]  # (n_test, 5)
    logp = -(diffs ** 2) / (2.0 * sigma ** 2)
    # Stable softmax across classes
    logp = logp - logp.max(axis=1, keepdims=True)
    probs = np.exp(logp)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


def _per_class_reliability(probs: np.ndarray, y_true: np.ndarray, n_bins: int = N_BINS):
    """For each class k, build a reliability curve from (predicted p_k, indicator
    [r_true==k]) pairs binned over predicted probability.

    Returns dict per class with:
        bin_centres (n_bins,)
        bin_pred    (n_bins,)  mean predicted prob in this bin
        bin_obs     (n_bins,)  mean observed frequency
        bin_count   (n_bins,)  number of points in this bin
        ece         scalar
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: dict = {}
    for k_idx in range(probs.shape[1]):
        p_k = probs[:, k_idx]
        indicator = (y_true == (k_idx + 1)).astype(np.float64)
        bin_pred = np.full(n_bins, np.nan)
        bin_obs = np.full(n_bins, np.nan)
        bin_count = np.zeros(n_bins, dtype=np.int64)
        ece = 0.0
        for b in range(n_bins):
            mask = (p_k > edges[b]) & (p_k <= edges[b + 1])
            if b == 0:
                mask = (p_k >= edges[b]) & (p_k <= edges[b + 1])
            n = int(mask.sum())
            bin_count[b] = n
            if n > 0:
                bin_pred[b] = float(p_k[mask].mean())
                bin_obs[b] = float(indicator[mask].mean())
                ece += (n / len(p_k)) * abs(bin_pred[b] - bin_obs[b])
        out[k_idx + 1] = {
            "bin_centres": (edges[:-1] + edges[1:]) / 2.0,
            "bin_pred": bin_pred,
            "bin_obs": bin_obs,
            "bin_count": bin_count,
            "ece": ece,
        }
    return out


def _pooled_reliability(probs: np.ndarray, y_true: np.ndarray, n_bins: int = N_BINS):
    """Pool all (p_k, indicator) pairs across classes for a single curve."""
    p_flat = probs.flatten()
    ind = np.zeros_like(probs)
    for k_idx in range(probs.shape[1]):
        ind[:, k_idx] = (y_true == (k_idx + 1)).astype(np.float64)
    ind_flat = ind.flatten()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_pred = np.full(n_bins, np.nan)
    bin_obs = np.full(n_bins, np.nan)
    bin_count = np.zeros(n_bins, dtype=np.int64)
    ece = 0.0
    n_total = len(p_flat)
    for b in range(n_bins):
        mask = (p_flat > edges[b]) & (p_flat <= edges[b + 1])
        if b == 0:
            mask = (p_flat >= edges[b]) & (p_flat <= edges[b + 1])
        n = int(mask.sum())
        bin_count[b] = n
        if n > 0:
            bin_pred[b] = float(p_flat[mask].mean())
            bin_obs[b] = float(ind_flat[mask].mean())
            ece += (n / n_total) * abs(bin_pred[b] - bin_obs[b])
    return {
        "bin_pred": bin_pred,
        "bin_obs": bin_obs,
        "bin_count": bin_count,
        "ece": ece,
    }


def _train_one(cfg, meta, train_ds, val_ds, test_ds, head: str, seed: int):
    set_seed(seed)
    tcfg = TrainConfig(
        epochs=cfg["epochs"], batch_size=cfg["batch_size"],
        lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        patience=cfg["patience_ablation"],
        device=cfg["device"], seed=seed,
    )
    model = CFGatedOrdinal(
        n_users=meta.n_users, n_items=meta.n_items,
        user_feat_dim=meta.user_feat_dim, item_feat_dim=meta.item_feat_dim,
        embed_dim=cfg["embed_dim"], fusion="gated", head=head,
        train_ratings=train_ds.rating if head == "ordinal" else None,
    )
    train_model(
        model, train_ds, val_ds, test_ds, tcfg,
        user_features=meta.user_features, item_features=meta.item_features,
        use_features=True, log_gate=True,
    )
    return model


def _plot_reliability(ax, pooled: dict, per_class: dict, title: str, colour: str):
    # Pooled curve as the foreground line
    pred = pooled["bin_pred"]
    obs = pooled["bin_obs"]
    valid = ~np.isnan(pred)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.6, linewidth=1)
    ax.plot(pred[valid], obs[valid], marker="o", linewidth=2.0,
            color=colour, label=f"pooled (ECE = {pooled['ece']:.3f})")
    # Per-class curves as faint lines for diagnostic context
    cmap = plt.get_cmap("tab10")
    for k in [1, 2, 3, 4, 5]:
        pc = per_class[k]
        v = ~np.isnan(pc["bin_pred"])
        if v.sum() < 2:
            continue
        ax.plot(pc["bin_pred"][v], pc["bin_obs"][v],
                marker=".", linewidth=0.8, alpha=0.55,
                color=cmap(k - 1),
                label=f"class {k} (ECE = {pc['ece']:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.85)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma", type=float, default=SIGMA_SIGMOID,
                        help="Gaussian kernel bandwidth for sigmoid -> 5-class distribution")
    args = parser.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    out_root = args.out or Path(f"results/{date.today().isoformat()}_reliability")
    out_root.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out_root / "config_snapshot.yaml")

    set_seed(args.seed)
    data_dir = Path(cfg["data_dir"])
    meta = build_metadata(data_dir)
    train_ds, val_ds, test_ds = load_split_with_val(data_dir, split=cfg["split"])
    y_true = test_ds.rating.numpy().astype(np.int64)

    print("Training gated+sigmoid …")
    t0 = time.perf_counter()
    sig_model = _train_one(cfg, meta, train_ds, val_ds, test_ds, "sigmoid", args.seed)
    sig_t = time.perf_counter() - t0
    print(f"  done in {sig_t:.1f}s")

    print("Training gated+ordinal …")
    t0 = time.perf_counter()
    ord_model = _train_one(cfg, meta, train_ds, val_ds, test_ds, "ordinal", args.seed)
    ord_t = time.perf_counter() - t0
    print(f"  done in {ord_t:.1f}s")

    sig_probs = _sigmoid_class_probs(sig_model, meta, test_ds, sigma=args.sigma)
    ord_probs = _ordinal_class_probs(ord_model, meta, test_ds)

    sig_pool = _pooled_reliability(sig_probs, y_true)
    ord_pool = _pooled_reliability(ord_probs, y_true)
    sig_pc = _per_class_reliability(sig_probs, y_true)
    ord_pc = _per_class_reliability(ord_probs, y_true)

    print("\n=== Expected Calibration Error (ECE) ===")
    print(f"  gated+sigmoid (kernel sigma={args.sigma}):  pooled ECE = {sig_pool['ece']:.4f}")
    print(f"  gated+ordinal (cumulative-link):           pooled ECE = {ord_pool['ece']:.4f}")
    print(f"  delta (sigmoid - ordinal):                 {sig_pool['ece'] - ord_pool['ece']:+.4f}")
    print("\n  Per-class ECE:")
    print(f"  {'class':<6} {'support':>8} {'sigmoid ECE':>12} {'ordinal ECE':>12}")
    for k in [1, 2, 3, 4, 5]:
        support = int((y_true == k).sum())
        print(f"  {k:<6} {support:>8d} {sig_pc[k]['ece']:>12.4f} {ord_pc[k]['ece']:>12.4f}")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    _plot_reliability(axes[0], sig_pool, sig_pc,
                      title=f"gated+sigmoid (Gaussian kernel σ={args.sigma})",
                      colour="#a0aec0")
    _plot_reliability(axes[1], ord_pool, ord_pc,
                      title="gated+ordinal (cumulative-link)",
                      colour="#2c5282")
    fig.suptitle(
        f"Reliability diagram on u1.test "
        f"(n={len(y_true)}, lower ECE is better)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_root / "reliability_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # CSV
    with (out_root / "ece_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["head", "scope", "ece", "support"])
        w.writerow(["sigmoid", "pooled", f"{sig_pool['ece']:.4f}", len(y_true) * 5])
        w.writerow(["ordinal", "pooled", f"{ord_pool['ece']:.4f}", len(y_true) * 5])
        for k in [1, 2, 3, 4, 5]:
            support = int((y_true == k).sum())
            w.writerow(["sigmoid", f"class_{k}", f"{sig_pc[k]['ece']:.4f}", support])
            w.writerow(["ordinal", f"class_{k}", f"{ord_pc[k]['ece']:.4f}", support])

    # Metadata
    meta_out = {
        "seed": args.seed,
        "split": cfg["split"],
        "kernel_sigma": args.sigma,
        "n_test": int(len(y_true)),
        "n_bins": int(N_BINS),
        "sigmoid_train_wall_s": float(sig_t),
        "ordinal_train_wall_s": float(ord_t),
        "ece": {
            "sigmoid_pooled": float(sig_pool["ece"]),
            "ordinal_pooled": float(ord_pool["ece"]),
            "per_class": {
                "sigmoid": {k: float(sig_pc[k]["ece"]) for k in [1, 2, 3, 4, 5]},
                "ordinal": {k: float(ord_pc[k]["ece"]) for k in [1, 2, 3, 4, 5]},
            },
        },
    }
    with (out_root / "results.json").open("w") as f:
        json.dump(meta_out, f, indent=2)
    print(f"\nWrote {out_root}/reliability_curves.png, ece_summary.csv, results.json")


if __name__ == "__main__":
    main()
