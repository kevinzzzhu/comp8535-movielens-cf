"""Week 4 (Tier 2) per-dimension gate analysis.

The gate $g_u$ is computed from $(p_u, a_u)$ alone -- both depend only on the
user index, so $g_u$ is a function of $u$ (not $(u, i)$). Likewise $g_i$ is a
function of $i$. The gate therefore induces a fixed (n_users x d) matrix and
(n_items x d) matrix once training finishes. This script extracts those
matrices for the trained gated+ordinal model and asks three questions:

  (Q1)  How heterogeneous is the gate across the 128 embedding dimensions?
        If g_u were constant ~0.27 over every dim, the per-dim per-user gate
        is a no-op vs. a global learned scalar. If it varies dim-to-dim, the
        per-dimension parameterisation is doing real work.

  (Q2)  How heterogeneous is the gate across users / items? If g_u varies
        substantially user-to-user, the gate is making per-user judgements
        about the relative trustworthiness of the embedding vs. side info.

  (Q3)  Does per-user mean g_u correlate with anything obvious -- user
        activity, occupation? This extends the cold-start analysis from
        bucket-level to per-user.

Outputs:
  results/<date>_gateheatmap/
    gate_per_dim.png        bar chart: 128 dim-means with std error bars
    gate_heatmap_users.png  heatmap of (sampled users x dims), sorted
    gate_heatmap_items.png  same for items, sorted by dominant genre
    gate_per_dim.csv        per-dim mean / std for both g_u and g_i
    results.json            metadata
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
import pandas as pd
import torch
import yaml

from src.dataset import GENRES, build_metadata, load_split_with_val
from src.model import CFGatedOrdinal
from src.train import TrainConfig, set_seed, train_model

N_USER_SAMPLES = 80
N_ITEM_SAMPLES = 80


def _train(cfg, meta, train_ds, val_ds, test_ds, seed: int) -> CFGatedOrdinal:
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
        embed_dim=cfg["embed_dim"], fusion="gated", head="ordinal",
        train_ratings=train_ds.rating,
    )
    train_model(
        model, train_ds, val_ds, test_ds, tcfg,
        user_features=meta.user_features, item_features=meta.item_features,
        use_features=True, log_gate=True,
    )
    return model


@torch.no_grad()
def _gates(model: CFGatedOrdinal, meta) -> tuple[np.ndarray, np.ndarray]:
    """Return (n_users, d) and (n_items, d) gate matrices."""
    model.eval()
    all_u = torch.arange(meta.n_users)
    all_i = torch.arange(meta.n_items)
    p = model.user_emb(all_u)
    q = model.item_emb(all_i)
    _, g_u = model.fuse_u(p, meta.user_features)
    _, g_i = model.fuse_i(q, meta.item_features)
    return g_u.cpu().numpy(), g_i.cpu().numpy()


def _per_dim_summary(G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of each column of G."""
    return G.mean(axis=0), G.std(axis=0)


def _plot_per_dim(g_u: np.ndarray, g_i: np.ndarray, save: Path) -> None:
    mu_u, sd_u = _per_dim_summary(g_u)
    mu_i, sd_i = _per_dim_summary(g_i)
    order_u = np.argsort(mu_u)
    order_i = np.argsort(mu_i)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), sharey=True)

    ax = axes[0]
    x = np.arange(len(mu_u))
    ax.errorbar(x, mu_u[order_u], yerr=sd_u[order_u], fmt=".",
                markersize=3, color="#2c5282", ecolor="#a0aec0", alpha=0.8)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5,
               label="zero-init / equal-weight")
    ax.axhline(mu_u.mean(), color="#d97706", linestyle=":", alpha=0.8,
               label=f"global mean {mu_u.mean():.3f}")
    ax.set_xlabel("embedding dimension (sorted by mean $g_u$)")
    ax.set_ylabel("$g_u$ (gate value)")
    ax.set_title(f"User-side gate: mean per dim across {g_u.shape[0]} users\n"
                 f"range [{mu_u.min():.2f}, {mu_u.max():.2f}], cross-dim std {mu_u.std():.3f}")
    ax.legend(fontsize=8, loc="best")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    ax = axes[1]
    x = np.arange(len(mu_i))
    ax.errorbar(x, mu_i[order_i], yerr=sd_i[order_i], fmt=".",
                markersize=3, color="#742a2a", ecolor="#a0aec0", alpha=0.8)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(mu_i.mean(), color="#d97706", linestyle=":", alpha=0.8,
               label=f"global mean {mu_i.mean():.3f}")
    ax.set_xlabel("embedding dimension (sorted by mean $g_i$)")
    ax.set_title(f"Item-side gate: mean per dim across {g_i.shape[0]} items\n"
                 f"range [{mu_i.min():.2f}, {mu_i.max():.2f}], cross-dim std {mu_i.std():.3f}")
    ax.legend(fontsize=8, loc="best")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap(G: np.ndarray, sample_idx: np.ndarray, sort_key: np.ndarray | None,
                  title: str, save: Path) -> None:
    """Heatmap of G[sample_idx, :] sorted by sort_key (or by mean if None)."""
    sub = G[sample_idx]
    if sort_key is not None:
        order = np.argsort(sort_key[sample_idx])
        sub = sub[order]
    else:
        order = np.argsort(sub.mean(axis=1))
        sub = sub[order]
    # Reorder dims by global mean too, so visually similar dims cluster.
    dim_order = np.argsort(G.mean(axis=0))
    sub = sub[:, dim_order]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    im = ax.imshow(sub, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xlabel("embedding dimension (re-ordered by global mean gate)")
    ax.set_ylabel(f"sampled entity (sorted by per-entity mean gate)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="gate value", pad=0.02)
    fig.tight_layout()
    fig.savefig(save, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    out_root = args.out or Path(f"results/{date.today().isoformat()}_gateheatmap")
    out_root.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out_root / "config_snapshot.yaml")

    set_seed(args.seed)
    data_dir = Path(cfg["data_dir"])
    meta = build_metadata(data_dir)
    train_ds, val_ds, test_ds = load_split_with_val(data_dir, split=cfg["split"])

    print("Training gated+ordinal …")
    t0 = time.perf_counter()
    model = _train(cfg, meta, train_ds, val_ds, test_ds, args.seed)
    train_t = time.perf_counter() - t0
    print(f"  {train_t:.1f}s")

    g_u, g_i = _gates(model, meta)
    print(f"  g_u shape {g_u.shape}, g_i shape {g_i.shape}")

    # Activity per user
    activity = np.zeros(meta.n_users, dtype=np.int64)
    np.add.at(activity, train_ds.user_idx.numpy(), 1)

    # Per-dim summary
    mu_u, sd_u = _per_dim_summary(g_u)
    mu_i, sd_i = _per_dim_summary(g_i)
    print(f"  user-side gate dim-means: range [{mu_u.min():.3f}, {mu_u.max():.3f}], "
          f"cross-dim std {mu_u.std():.3f}")
    print(f"  item-side gate dim-means: range [{mu_i.min():.3f}, {mu_i.max():.3f}], "
          f"cross-dim std {mu_i.std():.3f}")
    print(f"  user gates per-user mean range: [{g_u.mean(axis=1).min():.3f}, "
          f"{g_u.mean(axis=1).max():.3f}]")
    print(f"  item gates per-item mean range: [{g_i.mean(axis=1).min():.3f}, "
          f"{g_i.mean(axis=1).max():.3f}]")

    # Plot per-dim panel
    _plot_per_dim(g_u, g_i, out_root / "gate_per_dim.png")

    # Plot per-user heatmap
    rng = np.random.default_rng(args.seed)
    user_sample = rng.choice(meta.n_users, size=N_USER_SAMPLES, replace=False)
    _plot_heatmap(
        g_u, user_sample, sort_key=activity.astype(float),
        title=f"Per-user gate $g_u$ (sampled users sorted by training activity $|R_u|$)",
        save=out_root / "gate_heatmap_users.png",
    )

    # Item heatmap sorted by dominant-genre index
    item_df = pd.read_csv(data_dir / "u.item", sep="|",
                          names=["item_id", "title", "release", "video", "imdb"] + GENRES,
                          encoding="latin-1").sort_values("item_id").reset_index(drop=True)
    genre_mat = item_df[GENRES].to_numpy(dtype=np.int64)
    dominant = np.argmax(genre_mat[:, 1:], axis=1) + 1
    no_real = genre_mat[:, 1:].sum(axis=1) == 0
    dominant[no_real] = 0
    item_sort_key = np.zeros(meta.n_items, dtype=np.int64)
    item_sort_key[item_df["item_id"].to_numpy() - 1] = dominant
    item_sample = rng.choice(meta.n_items, size=N_ITEM_SAMPLES, replace=False)
    _plot_heatmap(
        g_i, item_sample, sort_key=item_sort_key.astype(float),
        title=f"Per-item gate $g_i$ (sampled items sorted by dominant-genre index)",
        save=out_root / "gate_heatmap_items.png",
    )

    # CSV
    with (out_root / "gate_per_dim.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dim", "mean_g_u", "std_g_u", "mean_g_i", "std_g_i"])
        for d in range(g_u.shape[1]):
            w.writerow([d, f"{mu_u[d]:.4f}", f"{sd_u[d]:.4f}",
                        f"{mu_i[d]:.4f}", f"{sd_i[d]:.4f}"])

    # JSON
    out = {
        "seed": args.seed,
        "split": cfg["split"],
        "train_wall_s": train_t,
        "g_u": {
            "shape": list(g_u.shape),
            "global_mean": float(g_u.mean()),
            "dim_mean_min": float(mu_u.min()),
            "dim_mean_max": float(mu_u.max()),
            "dim_mean_std": float(mu_u.std()),
            "per_user_mean_min": float(g_u.mean(axis=1).min()),
            "per_user_mean_max": float(g_u.mean(axis=1).max()),
            "per_user_mean_std": float(g_u.mean(axis=1).std()),
        },
        "g_i": {
            "shape": list(g_i.shape),
            "global_mean": float(g_i.mean()),
            "dim_mean_min": float(mu_i.min()),
            "dim_mean_max": float(mu_i.max()),
            "dim_mean_std": float(mu_i.std()),
            "per_item_mean_min": float(g_i.mean(axis=1).min()),
            "per_item_mean_max": float(g_i.mean(axis=1).max()),
            "per_item_mean_std": float(g_i.mean(axis=1).std()),
        },
    }
    with (out_root / "results.json").open("w") as f:
        json.dump(out, f, indent=2)

    print(f"\nWrote {out_root}/gate_per_dim.png, gate_heatmap_users.png, "
          f"gate_heatmap_items.png, gate_per_dim.csv, results.json")


if __name__ == "__main__":
    main()
