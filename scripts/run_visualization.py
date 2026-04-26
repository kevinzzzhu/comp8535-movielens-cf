"""Week 4 embedding visualisation: PCA / MDS / IsoMap on learned p' and q'.

Trains a single gated+ordinal model on u1.base, extracts post-fusion
representations p'_u and q'_i (i.e. after passing through the gated fusion
module, not the raw embeddings), projects them to 2D via three manifold methods,
and reports silhouette scores stratified by occupation (users) and dominant
genre (items).

Outputs:
    results/<date>_viz/
        manifold_grid.png       2x3 panel: rows = {users, items}, cols = {PCA, MDS, IsoMap}
        silhouette_scores.csv   per (entity, method) silhouette
        config_snapshot.yaml    config used
        results.json            metadata (seed, n_train_epochs, val_rmse, etc.)

Usage:
    PYTHONPATH=. uv run python scripts/run_visualization.py
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from src.dataset import GENRES, build_metadata, load_split, load_split_with_val
from src.model import CFGatedOrdinal
from src.train import TrainConfig, set_seed, train_model
from src.visualize import project, silhouette


# Top-N labels we keep for coloured visualisation. With 21 occupations or 19
# genres the legend becomes unreadable; the silhouette score also suffers from
# tiny-cluster noise. We restrict to the most common categories and assign all
# other entities to "other" before sampling.
TOP_OCCUPATIONS = 6
TOP_GENRES = 6
SAMPLE_USERS = 200
SAMPLE_ITEMS = 200


def _load_user_occupations(data_dir: Path, n_users: int) -> tuple[np.ndarray, list[str]]:
    """Return per-user occupation as integer label and the list of unique occupation names."""
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
    """Return per-item dominant genre as integer label."""
    cols = ["item_id", "title", "release", "video_release", "imdb"] + GENRES
    df = pd.read_csv(data_dir / "u.item", sep="|", names=cols, encoding="latin-1")
    df = df.sort_values("item_id").reset_index(drop=True)
    genre_mat = df[GENRES].to_numpy(dtype=np.int64)
    # Dominant genre = leftmost-set 1 in row; if all-zero, pick the first ("unknown").
    dominant = np.argmax(genre_mat[:, 1:], axis=1) + 1  # skip "unknown" if anything else set
    no_real_genre = genre_mat[:, 1:].sum(axis=1) == 0
    dominant[no_real_genre] = 0  # "unknown"
    labels = np.full(n_items, -1, dtype=np.int64)
    labels[df["item_id"].to_numpy() - 1] = dominant
    return labels, GENRES


def _restrict_to_top(labels: np.ndarray, names: list[str], top_n: int) -> tuple[np.ndarray, list[str]]:
    """Keep only the top-N most common labels; rest are bucketed as 'rest' (= -2).

    The bucket name avoids the literal string "other" because MovieLens has an
    occupation called "other" and the collision makes the legend ambiguous.
    """
    valid = labels[labels >= 0]
    counts = pd.Series(valid).value_counts()
    top_idx = counts.head(top_n).index.to_numpy()
    new_labels = labels.copy()
    keep_mask = np.isin(labels, top_idx) & (labels >= 0)
    new_labels[~keep_mask] = -2  # bucket
    label_names = [names[i] for i in top_idx] + ["(rest)"]
    remap = {idx: k for k, idx in enumerate(top_idx)}
    out = np.full_like(new_labels, -1)
    for old, new in remap.items():
        out[new_labels == old] = new
    out[new_labels == -2] = top_n
    return out, label_names


def _sample_balanced(labels: np.ndarray, n_per_class: int, rng: np.random.Generator) -> np.ndarray:
    """Sample roughly n_per_class indices from each class (excluding label = -1)."""
    classes = sorted(set(labels.tolist()) - {-1})
    picks = []
    for c in classes:
        idx = np.where(labels == c)[0]
        k = min(n_per_class, len(idx))
        picks.append(rng.choice(idx, size=k, replace=False))
    return np.concatenate(picks)


@torch.no_grad()
def _post_fusion_embeddings(model: CFGatedOrdinal, meta) -> tuple[np.ndarray, np.ndarray]:
    """Extract p'_u and q'_i for all users/items by running fusion on the full population."""
    model.eval()
    all_u = torch.arange(meta.n_users)
    all_i = torch.arange(meta.n_items)
    p = model.user_emb(all_u)
    q = model.item_emb(all_i)
    if model.fuse_u is not None:
        vu = meta.user_features
        vi = meta.item_features
        p_prime, _ = model.fuse_u(p, vu)
        q_prime, _ = model.fuse_i(q, vi)
    else:
        p_prime, q_prime = p, q
    return p_prime.cpu().numpy(), q_prime.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--isomap-neighbors", type=int, default=15)
    args = parser.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    out_root = args.out or Path(f"results/{date.today().isoformat()}_viz")
    out_root.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(args.config, out_root / "config_snapshot.yaml")

    set_seed(args.seed)
    data_dir = Path(cfg["data_dir"])
    meta = build_metadata(data_dir)
    train_ds, val_ds, test_ds = load_split_with_val(data_dir, split=cfg["split"])

    tcfg = TrainConfig(
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        patience=cfg["patience_ablation"],
        device=cfg["device"],
        seed=args.seed,
    )

    print(f"Training gated+ordinal (seed={args.seed}, patience={tcfg.patience}) …")
    t0 = time.perf_counter()
    model = CFGatedOrdinal(
        n_users=meta.n_users, n_items=meta.n_items,
        user_feat_dim=meta.user_feat_dim, item_feat_dim=meta.item_feat_dim,
        embed_dim=cfg["embed_dim"], fusion="gated", head="ordinal",
        train_ratings=train_ds.rating,
    )
    result = train_model(
        model, train_ds, val_ds, test_ds, tcfg,
        user_features=meta.user_features, item_features=meta.item_features,
        use_features=True, log_gate=True,
    )
    train_wall = time.perf_counter() - t0
    print(f"  Best RMSE = {result['best_rmse']:.4f}, train wall = {train_wall:.1f}s")

    # Extract post-fusion embeddings for ALL users/items.
    P, Q = _post_fusion_embeddings(model, meta)
    print(f"  p' shape = {P.shape}, q' shape = {Q.shape}")

    # Build labels.
    user_occ_full, occ_names = _load_user_occupations(data_dir, meta.n_users)
    item_genre_full, genre_names = _load_item_genres(data_dir, meta.n_items)
    user_lbl, user_lbl_names = _restrict_to_top(user_occ_full, occ_names, TOP_OCCUPATIONS)
    item_lbl, item_lbl_names = _restrict_to_top(item_genre_full, genre_names, TOP_GENRES)

    # Sample balanced subsets to keep figures legible.
    rng = np.random.default_rng(args.seed)
    user_subset = _sample_balanced(user_lbl, n_per_class=SAMPLE_USERS // (TOP_OCCUPATIONS + 1), rng=rng)
    item_subset = _sample_balanced(item_lbl, n_per_class=SAMPLE_ITEMS // (TOP_GENRES + 1), rng=rng)
    rng.shuffle(user_subset)
    rng.shuffle(item_subset)
    print(f"  user subset: {len(user_subset)}, item subset: {len(item_subset)}")

    P_sub, q_sub_lbl = P[user_subset], user_lbl[user_subset]
    Q_sub, i_sub_lbl = Q[item_subset], item_lbl[item_subset]

    methods = ["pca", "mds", "isomap"]
    user_proj: dict[str, np.ndarray] = {}
    item_proj: dict[str, np.ndarray] = {}
    sil_rows: list[dict] = []

    for m in methods:
        kw = {}
        if m == "isomap":
            kw["n_neighbors"] = args.isomap_neighbors
        Yp = project(P_sub, method=m, **kw)
        Yi = project(Q_sub, method=m, **kw)
        user_proj[m] = Yp
        item_proj[m] = Yi
        # Silhouette on the FULL (high-D) space using the same labels — this is
        # what really tells us whether the embeddings cluster meaningfully. We
        # also report silhouette on the projected 2D for the figure caption.
        s_user_hd = silhouette(P_sub, q_sub_lbl)
        s_item_hd = silhouette(Q_sub, i_sub_lbl)
        s_user_2d = silhouette(Yp, q_sub_lbl)
        s_item_2d = silhouette(Yi, i_sub_lbl)
        sil_rows.append({
            "entity": "user", "method": m,
            "silhouette_hd": f"{s_user_hd:.4f}",
            "silhouette_2d": f"{s_user_2d:.4f}",
            "n_samples": len(user_subset),
            "n_classes": int(len(set(q_sub_lbl.tolist()))),
        })
        sil_rows.append({
            "entity": "item", "method": m,
            "silhouette_hd": f"{s_item_hd:.4f}",
            "silhouette_2d": f"{s_item_2d:.4f}",
            "n_samples": len(item_subset),
            "n_classes": int(len(set(i_sub_lbl.tolist()))),
        })
        print(f"  {m:>6}  user sil 2D={s_user_2d:.3f}  item sil 2D={s_item_2d:.3f}")

    # 2x3 grid figure.
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.5))
    cmap_u = plt.get_cmap("tab10")
    cmap_i = plt.get_cmap("tab10")

    method_titles = {"pca": "PCA", "mds": "MDS", "isomap": "IsoMap"}
    for col, m in enumerate(methods):
        Yp = user_proj[m]
        Yi = item_proj[m]
        ax = axes[0, col]
        for k, name in enumerate(user_lbl_names):
            mask = q_sub_lbl == k
            if mask.sum() == 0: continue
            ax.scatter(Yp[mask, 0], Yp[mask, 1], s=18, alpha=0.78,
                       color=cmap_u(k), label=name)
        ax.set_title(f"{method_titles[m]}: user embeddings $p'_u$ by occupation")
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.legend(fontsize=7, loc="best", ncol=2, framealpha=0.85)

        ax = axes[1, col]
        for k, name in enumerate(item_lbl_names):
            mask = i_sub_lbl == k
            if mask.sum() == 0: continue
            ax.scatter(Yi[mask, 0], Yi[mask, 1], s=18, alpha=0.78,
                       color=cmap_i(k), label=name)
        ax.set_title(f"{method_titles[m]}: item embeddings $q'_i$ by dominant genre")
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.legend(fontsize=7, loc="best", ncol=2, framealpha=0.85)

    fig.suptitle(
        f"Manifold projections of post-fusion embeddings (gated+ordinal, "
        f"seed={args.seed}, RMSE={result['best_rmse']:.4f})",
        fontsize=11, y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out_root / "manifold_grid.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Silhouette CSV.
    with (out_root / "silhouette_scores.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["entity", "method", "silhouette_hd",
                                           "silhouette_2d", "n_samples", "n_classes"])
        w.writeheader()
        w.writerows(sil_rows)

    # Metadata.
    meta_out = {
        "seed": args.seed,
        "best_rmse": float(result["best_rmse"]),
        "train_wall_s": float(train_wall),
        "n_users": int(meta.n_users),
        "n_items": int(meta.n_items),
        "user_subset_size": int(len(user_subset)),
        "item_subset_size": int(len(item_subset)),
        "user_label_names": user_lbl_names,
        "item_label_names": item_lbl_names,
        "isomap_neighbors": int(args.isomap_neighbors),
    }
    with (out_root / "results.json").open("w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"\nWrote {out_root}/manifold_grid.png, silhouette_scores.csv, results.json")


if __name__ == "__main__":
    main()
