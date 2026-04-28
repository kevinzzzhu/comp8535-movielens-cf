"""Tier 2 gate interpretability: per-dimension gate distributions on u1.

Trains a single gated+ordinal model and, on the test set, records the
per-prediction gate value g_u[d], g_i[d] for every embedding dimension d.
Aggregates two ways:

  1. Population-level: mean and ±1 std of each dimension across the entire
     test set. Sorting by mean gives a sense of the gate distribution.
     If the bulk of the dimensions sit well below 0.5 (gate closed -> trust
     ID embedding) but a tail of dimensions sits above 0.5 (gate open ->
     trust side-info projection), the model has learned a *selective* fusion
     -- which is the design intent.

  2. Category-stratified: mean gate per dimension within each top-K
     occupation (users) / dominant genre (items). Differences across rows
     show whether the gate's per-dimension behaviour depends on the
     categorical side info. If row-by-row patterns are nearly identical,
     the gate's choice is population-wide; if they differ, the gate is
     conditioning on side info.

Outputs:
    results/<date>_gate_analysis/
        gate_distribution.png    sorted per-dim mean ± std for u and i sides
        gate_strat_users.png     heatmap: rows = occupations, cols = sorted dims
        gate_strat_items.png     heatmap: rows = genres, cols = sorted dims
        gate_perdim_users.csv    per-dim mean / std across population (users)
        gate_perdim_items.csv    same for items
        results.json             metadata

Usage:
    PYTHONPATH=. uv run python scripts/run_gate_analysis.py
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

TOP_OCCUPATIONS = 6
TOP_GENRES = 6


@torch.no_grad()
def _per_pred_gates(model: CFGatedOrdinal, meta, ds) -> tuple[np.ndarray, np.ndarray]:
    """Return (g_u, g_i) of shape (n_test, d) for every test prediction."""
    model.eval()
    out = model(ds.user_idx, ds.item_idx, meta.user_features, meta.item_features)
    g_u = out["gate_u"].cpu().numpy()
    g_i = out["gate_i"].cpu().numpy()
    return g_u, g_i


def _load_user_occupations(data_dir: Path, n_users: int) -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(data_dir / "u.user", sep="|",
                     names=["user_id", "age", "gender", "occupation", "zip"],
                     encoding="latin-1")
    df = df.sort_values("user_id").reset_index(drop=True)
    occ_names = sorted(df["occupation"].unique().tolist())
    occ_to_idx = {n: k for k, n in enumerate(occ_names)}
    labels = np.full(n_users, -1, dtype=np.int64)
    labels[df["user_id"].to_numpy() - 1] = df["occupation"].map(occ_to_idx).to_numpy()
    return labels, occ_names


def _load_item_genres(data_dir: Path, n_items: int) -> tuple[np.ndarray, list[str]]:
    cols = ["item_id", "title", "release", "video_release", "imdb"] + GENRES
    df = pd.read_csv(data_dir / "u.item", sep="|", names=cols, encoding="latin-1")
    df = df.sort_values("item_id").reset_index(drop=True)
    genre_mat = df[GENRES].to_numpy(dtype=np.int64)
    dominant = np.argmax(genre_mat[:, 1:], axis=1) + 1
    no_real_genre = genre_mat[:, 1:].sum(axis=1) == 0
    dominant[no_real_genre] = 0
    labels = np.full(n_items, -1, dtype=np.int64)
    labels[df["item_id"].to_numpy() - 1] = dominant
    return labels, GENRES


def _top_k_categories(labels: np.ndarray, k: int) -> list[int]:
    valid = labels[labels >= 0]
    counts = pd.Series(valid).value_counts()
    return list(counts.head(k).index)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    out_root = args.out or Path(f"results/{date.today().isoformat()}_gate_analysis")
    out_root.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out_root / "config_snapshot.yaml")

    set_seed(args.seed)
    data_dir = Path(cfg["data_dir"])
    meta = build_metadata(data_dir)
    train_ds, val_ds, test_ds = load_split_with_val(data_dir, split=cfg["split"])

    tcfg = TrainConfig(
        epochs=cfg["epochs"], batch_size=cfg["batch_size"],
        lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        patience=cfg["patience_ablation"],
        device=cfg["device"], seed=args.seed,
    )
    print("Training gated+ordinal …")
    t0 = time.perf_counter()
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
    train_wall = time.perf_counter() - t0
    print(f"  done in {train_wall:.1f}s")

    g_u, g_i = _per_pred_gates(model, meta, test_ds)
    n_test, d = g_u.shape
    print(f"  test predictions: {n_test}, embedding dim: {d}")

    # --- Per-dim aggregates across population.
    mu_u = g_u.mean(axis=0); sd_u = g_u.std(axis=0)
    mu_i = g_i.mean(axis=0); sd_i = g_i.std(axis=0)
    sort_u = np.argsort(mu_u)
    sort_i = np.argsort(mu_i)

    print(f"  user gate per-dim: mean range [{mu_u.min():.3f}, {mu_u.max():.3f}], "
          f"population mean {mu_u.mean():.3f}, frac dims > 0.5: "
          f"{(mu_u > 0.5).mean():.3f}")
    print(f"  item gate per-dim: mean range [{mu_i.min():.3f}, {mu_i.max():.3f}], "
          f"population mean {mu_i.mean():.3f}, frac dims > 0.5: "
          f"{(mu_i > 0.5).mean():.3f}")

    # CSVs.
    with (out_root / "gate_perdim_users.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["dim_sorted_idx", "orig_dim", "mean_g_u", "std_g_u"])
        for r, k in enumerate(sort_u):
            w.writerow([r, int(k), f"{mu_u[k]:.4f}", f"{sd_u[k]:.4f}"])
    with (out_root / "gate_perdim_items.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["dim_sorted_idx", "orig_dim", "mean_g_i", "std_g_i"])
        for r, k in enumerate(sort_i):
            w.writerow([r, int(k), f"{mu_i[k]:.4f}", f"{sd_i[k]:.4f}"])

    # --- Distribution figure: sorted per-dim mean ± std for u and i sides.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, mu, sd, sort_idx, title, colour in [
        (axes[0], mu_u, sd_u, sort_u, "user gate $g_u$ per dimension (sorted)", "#2c5282"),
        (axes[1], mu_i, sd_i, sort_i, "item gate $g_i$ per dimension (sorted)", "#742a2a"),
    ]:
        x = np.arange(d)
        m = mu[sort_idx]
        s = sd[sort_idx]
        ax.fill_between(x, m - s, m + s, color=colour, alpha=0.18, label="±1 std")
        ax.plot(x, m, color=colour, linewidth=1.5, label="mean")
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="zero-init (0.5)")
        ax.set_xlabel("embedding dimension (sorted by mean gate)")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("gate value")
    fig.suptitle(f"Per-dimension gate distribution on u1.test "
                 f"(n_test={n_test}, d={d})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_root / "gate_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # --- Category-stratified heatmaps.
    user_occ, occ_names = _load_user_occupations(data_dir, meta.n_users)
    item_genre, genre_names = _load_item_genres(data_dir, meta.n_items)
    top_occ = _top_k_categories(user_occ, TOP_OCCUPATIONS)
    top_gen = _top_k_categories(item_genre, TOP_GENRES)

    # For each test prediction, look up its user's occupation and item's genre.
    test_user = test_ds.user_idx.numpy()
    test_item = test_ds.item_idx.numpy()
    pred_occ = user_occ[test_user]
    pred_gen = item_genre[test_item]

    def _category_strata(labels_pred: np.ndarray, top_idx: list[int],
                         names: list[str], gates: np.ndarray, sort_idx: np.ndarray
                         ) -> tuple[np.ndarray, list[str]]:
        rows = []
        labels = []
        for c in top_idx:
            mask = labels_pred == c
            if mask.sum() == 0:
                continue
            rows.append(gates[mask].mean(axis=0)[sort_idx])
            labels.append(names[c])
        return np.stack(rows), labels

    occ_grid, occ_labels = _category_strata(pred_occ, top_occ, occ_names, g_u, sort_u)
    gen_grid, gen_labels = _category_strata(pred_gen, top_gen, genre_names, g_i, sort_i)

    def _plot_heatmap(grid: np.ndarray, ylabels: list[str], title: str, save: Path) -> None:
        fig, ax = plt.subplots(figsize=(11, 0.55 * len(ylabels) + 1.6))
        im = ax.imshow(grid, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1)
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels)
        ax.set_xlabel("embedding dimension (sorted by population mean gate)")
        ax.set_title(title, fontsize=10)
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("mean gate within category")
        cbar.ax.axhline(y=0.5, color="black", linewidth=0.7)
        fig.tight_layout()
        fig.savefig(save, dpi=180, bbox_inches="tight")
        plt.close(fig)

    _plot_heatmap(occ_grid, occ_labels,
                  f"Per-dim user gate $g_u$ stratified by occupation (top {TOP_OCCUPATIONS})",
                  out_root / "gate_strat_users.png")
    _plot_heatmap(gen_grid, gen_labels,
                  f"Per-dim item gate $g_i$ stratified by dominant genre (top {TOP_GENRES})",
                  out_root / "gate_strat_items.png")

    # --- Heterogeneity score: rank-correlation between any two strata's per-dim
    # ordering. If all categories agree on which dims to open, the gate is
    # population-wide (low heterogeneity). If they disagree, the gate is
    # conditioning on the categorical side info.
    from scipy.stats import spearmanr  # type: ignore
    occ_rho = []
    for i in range(occ_grid.shape[0]):
        for j in range(i + 1, occ_grid.shape[0]):
            r, _ = spearmanr(occ_grid[i], occ_grid[j])
            occ_rho.append(r)
    gen_rho = []
    for i in range(gen_grid.shape[0]):
        for j in range(i + 1, gen_grid.shape[0]):
            r, _ = spearmanr(gen_grid[i], gen_grid[j])
            gen_rho.append(r)
    occ_rho_mean = float(np.mean(occ_rho)) if occ_rho else float("nan")
    gen_rho_mean = float(np.mean(gen_rho)) if gen_rho else float("nan")
    print(f"  pairwise Spearman rho between strata "
          f"(higher -> more population-wide, lower -> more conditional):")
    print(f"    user-side (occupations): mean rho = {occ_rho_mean:.4f}")
    print(f"    item-side (genres):      mean rho = {gen_rho_mean:.4f}")

    meta_out = {
        "seed": args.seed,
        "split": cfg["split"],
        "train_wall_s": train_wall,
        "n_test": int(n_test),
        "embed_dim": int(d),
        "user_gate": {
            "population_mean": float(mu_u.mean()),
            "min_dim_mean": float(mu_u.min()),
            "max_dim_mean": float(mu_u.max()),
            "frac_dims_above_0_5": float((mu_u > 0.5).mean()),
            "occ_pairwise_spearman_mean": occ_rho_mean,
        },
        "item_gate": {
            "population_mean": float(mu_i.mean()),
            "min_dim_mean": float(mu_i.min()),
            "max_dim_mean": float(mu_i.max()),
            "frac_dims_above_0_5": float((mu_i > 0.5).mean()),
            "gen_pairwise_spearman_mean": gen_rho_mean,
        },
    }
    with (out_root / "results.json").open("w") as f:
        json.dump(meta_out, f, indent=2)
    print(f"\nWrote {out_root}/gate_distribution.png, "
          f"gate_strat_users.png, gate_strat_items.png, CSVs, results.json")


if __name__ == "__main__":
    main()
