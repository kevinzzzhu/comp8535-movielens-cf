"""Week 4 cold-start / gate-trajectory analysis.

Trains gated+ordinal and additive+ordinal once each, then for every test-set
rating computes:
  - the squared error from each model
  - the per-prediction gate values g_u, g_i (gated model only)

Bucketing predictions by |R_u| (number of training ratings the user has) lets
us answer two questions:

  (Q1)  Does the user-side gate close (g_u -> 0) as |R_u| grows? That would
        mean the model learns to trust the ID embedding when it has signal
        and to lean on side features when it doesn't.

  (Q2)  Does gated fusion's RMSE advantage over additive fusion concentrate
        on cold users (low |R_u|), or is it uniform across the population?

Outputs:
    results/<date>_coldstart/
        coldstart_buckets.csv     per-bucket RMSE/MAE + gate stats for both heads
        gate_vs_users.png         mean g_u (and g_i) vs |R_u|
        rmse_vs_users.png         RMSE-by-bucket bar chart, gated vs additive
        results.json              metadata
        config_snapshot.yaml

Usage:
    PYTHONPATH=. uv run python scripts/run_coldstart.py
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

from src.dataset import build_metadata, load_split
from src.model import CFGatedOrdinal
from src.train import TrainConfig, set_seed, train_model

# Per-user activity buckets. Edges in number of training ratings.
# Chosen on inspection of MovieLens-100K u1.base distribution: min(|R_u|) ~ 19,
# median ~ 50, max ~ 700+. The first bucket captures the genuinely cold tail.
BUCKET_EDGES = [0, 30, 60, 120, 240, 10_000]
BUCKET_LABELS = ["<30", "30–59", "60–119", "120–239", "≥240"]


@torch.no_grad()
def _predict_with_gates(model: CFGatedOrdinal, meta, ds) -> dict:
    """Return per-prediction tensors: pred, gate_u_mean, gate_i_mean (None if no fusion)."""
    model.eval()
    u = ds.user_idx
    i = ds.item_idx
    out = model(u, i, meta.user_features, meta.item_features)
    pred = out["pred"].cpu().numpy()
    gate_u = None
    gate_i = None
    if out.get("gate_u") is not None:
        gate_u = out["gate_u"].mean(dim=-1).cpu().numpy()
    if out.get("gate_i") is not None:
        gate_i = out["gate_i"].mean(dim=-1).cpu().numpy()
    return {"pred": pred, "gate_u_mean": gate_u, "gate_i_mean": gate_i}


def _build_user_activity(train_ds, n_users: int) -> np.ndarray:
    """Return an (n_users,) int array: |R_u| from the training split."""
    activity = np.zeros(n_users, dtype=np.int64)
    u_idx = train_ds.user_idx.numpy()
    np.add.at(activity, u_idx, 1)
    return activity


def _train_one(cfg, meta, train_ds, test_ds, fusion: str, head: str = "ordinal") -> CFGatedOrdinal:
    """Train one CFGatedOrdinal config end-to-end. Returns the trained model."""
    tcfg = TrainConfig(
        epochs=cfg["epochs"], batch_size=cfg["batch_size"],
        lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        patience=cfg["patience_ablation"],
        device=cfg["device"], seed=cfg["seed"],
    )
    model = CFGatedOrdinal(
        n_users=meta.n_users, n_items=meta.n_items,
        user_feat_dim=meta.user_feat_dim, item_feat_dim=meta.item_feat_dim,
        embed_dim=cfg["embed_dim"], fusion=fusion, head=head,
        train_ratings=train_ds.rating if head == "ordinal" else None,
    )
    train_model(
        model, train_ds, test_ds, tcfg,
        user_features=meta.user_features, item_features=meta.item_features,
        use_features=True, log_gate=(fusion == "gated"),
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)
    cfg["seed"] = args.seed

    out_root = args.out or Path(f"results/{date.today().isoformat()}_coldstart")
    out_root.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out_root / "config_snapshot.yaml")

    set_seed(args.seed)
    data_dir = Path(cfg["data_dir"])
    meta = build_metadata(data_dir)
    train_ds, test_ds = load_split(data_dir, split=cfg["split"])
    activity = _build_user_activity(train_ds, meta.n_users)

    # Per-test-rating user activity (i.e. each test row inherits its user's |R_u|).
    test_user_idx = test_ds.user_idx.numpy()
    test_target = test_ds.rating.numpy()
    test_activity = activity[test_user_idx]

    # Bucket assignment for each test row.
    # np.digitize returns 1..len(BUCKET_EDGES)-1; subtract 1 to get 0-indexed bucket.
    bucket_idx = np.digitize(test_activity, BUCKET_EDGES, right=False) - 1
    bucket_idx = np.clip(bucket_idx, 0, len(BUCKET_LABELS) - 1)

    print("=== Bucket sizes ===")
    for k, name in enumerate(BUCKET_LABELS):
        n = int((bucket_idx == k).sum())
        n_users_in_bucket = int(np.unique(test_user_idx[bucket_idx == k]).size)
        print(f"  bucket {k} (|R_u| {name}): {n:5d} test ratings across {n_users_in_bucket} users")

    print("\n=== Training models ===")

    print("Training gated+ordinal …")
    set_seed(args.seed)
    t0 = time.perf_counter()
    gated = _train_one(cfg, meta, train_ds, test_ds, fusion="gated")
    gated_t = time.perf_counter() - t0
    gated_out = _predict_with_gates(gated, meta, test_ds)
    print(f"  done in {gated_t:.1f}s")

    print("Training additive+ordinal …")
    set_seed(args.seed)
    t0 = time.perf_counter()
    additive = _train_one(cfg, meta, train_ds, test_ds, fusion="additive")
    additive_t = time.perf_counter() - t0
    additive_out = _predict_with_gates(additive, meta, test_ds)
    print(f"  done in {additive_t:.1f}s")

    # Per-bucket aggregation.
    rows = []
    print("\n=== Per-bucket metrics ===")
    print(f"  {'bucket':<10} {'n_test':>7} {'n_users':>8} "
          f"{'gated RMSE':>12} {'add. RMSE':>12} {'Δ':>8} "
          f"{'mean g_u':>10} {'mean g_i':>10}")
    for k, name in enumerate(BUCKET_LABELS):
        mask = bucket_idx == k
        if mask.sum() == 0:
            continue
        targets = test_target[mask]
        n_users_in_bucket = int(np.unique(test_user_idx[mask]).size)

        g_pred = gated_out["pred"][mask]
        a_pred = additive_out["pred"][mask]
        g_rmse = float(np.sqrt(np.mean((g_pred - targets) ** 2)))
        g_mae = float(np.mean(np.abs(g_pred - targets)))
        a_rmse = float(np.sqrt(np.mean((a_pred - targets) ** 2)))
        a_mae = float(np.mean(np.abs(a_pred - targets)))

        gu = float(np.mean(gated_out["gate_u_mean"][mask]))
        gi = float(np.mean(gated_out["gate_i_mean"][mask]))

        delta = a_rmse - g_rmse  # positive => gated wins
        print(f"  {name:<10} {int(mask.sum()):>7d} {n_users_in_bucket:>8d} "
              f"{g_rmse:>12.4f} {a_rmse:>12.4f} {delta:>+8.4f} "
              f"{gu:>10.4f} {gi:>10.4f}")

        rows.append({
            "bucket": name,
            "n_test_ratings": int(mask.sum()),
            "n_users": n_users_in_bucket,
            "gated_rmse": f"{g_rmse:.4f}",
            "gated_mae": f"{g_mae:.4f}",
            "additive_rmse": f"{a_rmse:.4f}",
            "additive_mae": f"{a_mae:.4f}",
            "rmse_delta_additive_minus_gated": f"{delta:+.4f}",
            "mean_gate_u": f"{gu:.4f}",
            "mean_gate_i": f"{gi:.4f}",
        })

    # CSV.
    with (out_root / "coldstart_buckets.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Plot 1: mean gate vs |R_u| bucket.
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    xs = np.arange(len(BUCKET_LABELS))
    g_us = [float(r["mean_gate_u"]) for r in rows]
    g_is = [float(r["mean_gate_i"]) for r in rows]
    ax.plot(xs, g_us, marker="o", linewidth=2, label="user gate $g_u$",
            color="#2c5282")
    ax.plot(xs, g_is, marker="s", linewidth=2, label="item gate $g_i$",
            color="#742a2a")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6,
               label="zero-init / equal-weight baseline")
    ax.set_xticks(xs)
    ax.set_xticklabels([r["bucket"] for r in rows])
    ax.set_xlabel("user activity |R_u| (training ratings per user)")
    ax.set_ylabel("mean gate value across test predictions")
    ax.set_title("Gate trajectory by user activity (gated+ordinal)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_root / "gate_vs_users.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: per-bucket RMSE comparison.
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    width = 0.36
    g_rmses = [float(r["gated_rmse"]) for r in rows]
    a_rmses = [float(r["additive_rmse"]) for r in rows]
    ax.bar(xs - width / 2, a_rmses, width, label="additive+ordinal",
           color="#a0aec0")
    ax.bar(xs + width / 2, g_rmses, width, label="gated+ordinal",
           color="#2c5282")
    ax.set_xticks(xs)
    ax.set_xticklabels([r["bucket"] for r in rows])
    ax.set_xlabel("user activity |R_u| (training ratings per user)")
    ax.set_ylabel("test RMSE")
    ax.set_title("RMSE by user activity bucket (3-seed-equivalent single run)")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_root / "rmse_vs_users.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Metadata.
    meta_out = {
        "seed": args.seed,
        "bucket_edges": BUCKET_EDGES,
        "bucket_labels": BUCKET_LABELS,
        "gated_train_wall_s": gated_t,
        "additive_train_wall_s": additive_t,
        "n_test_ratings": int(test_target.size),
        "n_users_total": int(meta.n_users),
        "patience": cfg["patience_ablation"],
        "embed_dim": cfg["embed_dim"],
        "weight_decay": cfg["weight_decay"],
    }
    with (out_root / "results.json").open("w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"\nWrote {out_root}/coldstart_buckets.csv, "
          f"gate_vs_users.png, rmse_vs_users.png, results.json")


if __name__ == "__main__":
    main()
