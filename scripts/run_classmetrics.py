"""Week 4 per-class confusion matrices + per-class F1 / precision / recall.

Trains gated+ordinal and gated+sigmoid once each on u1 and compares per-class
behaviour. Tells us *where* the ordinal head wins (or doesn't): typically on
extreme classes (1 and 5) where the cumulative-link parameterisation captures
the boundary structure better than a sigmoid trained with MSE.

Outputs:
    results/<date>_classmetrics/
        confusion_grid.png      side-by-side 5x5 confusion matrices for both heads
        per_class_f1.png        bar chart of per-class precision / recall / F1
        per_class_metrics.csv   full per-class table for both heads
        results.json            best RMSE, train wall, etc.
        config_snapshot.yaml

Usage:
    PYTHONPATH=. uv run python scripts/run_classmetrics.py
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

CLASSES = [1, 2, 3, 4, 5]


@torch.no_grad()
def _predicted_classes(model: CFGatedOrdinal, meta, ds) -> np.ndarray:
    """Round-and-clip the model's expected rating to {1,2,3,4,5}."""
    model.eval()
    out = model(ds.user_idx, ds.item_idx, meta.user_features, meta.item_features)
    pred = out["pred"].cpu().numpy()
    return np.clip(np.round(pred), 1, 5).astype(np.int64)


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return a 5x5 confusion matrix indexed by class 1..5."""
    cm = np.zeros((5, 5), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t - 1, p - 1] += 1
    return cm


def _per_class_metrics(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precision, recall, F1 per class from a confusion matrix."""
    prec = np.zeros(5, dtype=np.float64)
    rec = np.zeros(5, dtype=np.float64)
    f1 = np.zeros(5, dtype=np.float64)
    for k in range(5):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        prec[k] = tp / max(tp + fp, 1)
        rec[k] = tp / max(tp + fn, 1)
        f1[k] = 2 * prec[k] * rec[k] / max(prec[k] + rec[k], 1e-12)
    return prec, rec, f1


def _train_one(cfg, meta, train_ds, val_ds, test_ds, fusion: str, head: str, seed: int) -> CFGatedOrdinal:
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
        embed_dim=cfg["embed_dim"], fusion=fusion, head=head,
        train_ratings=train_ds.rating if head == "ordinal" else None,
    )
    train_model(
        model, train_ds, val_ds, test_ds, tcfg,
        user_features=meta.user_features, item_features=meta.item_features,
        use_features=True, log_gate=(fusion == "gated"),
    )
    return model


def _plot_confusion(ax, cm: np.ndarray, title: str) -> None:
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(5)); ax.set_xticklabels(CLASSES)
    ax.set_yticks(range(5)); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(title)
    for i in range(5):
        for j in range(5):
            colour = "white" if cm_norm[i, j] > 0.55 else "black"
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                    ha="center", va="center", color=colour, fontsize=8)
    return im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    out_root = args.out or Path(f"results/{date.today().isoformat()}_classmetrics")
    out_root.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out_root / "config_snapshot.yaml")

    set_seed(args.seed)
    data_dir = Path(cfg["data_dir"])
    meta = build_metadata(data_dir)
    train_ds, val_ds, test_ds = load_split_with_val(data_dir, split=cfg["split"])
    y_true = test_ds.rating.numpy().astype(np.int64)

    print("=== Training ===")
    print("gated+ordinal …")
    t0 = time.perf_counter()
    ord_model = _train_one(cfg, meta, train_ds, val_ds, test_ds, "gated", "ordinal", args.seed)
    ord_t = time.perf_counter() - t0
    print(f"  {ord_t:.1f}s")

    print("gated+sigmoid …")
    t0 = time.perf_counter()
    sig_model = _train_one(cfg, meta, train_ds, val_ds, test_ds, "gated", "sigmoid", args.seed)
    sig_t = time.perf_counter() - t0
    print(f"  {sig_t:.1f}s")

    y_pred_ord = _predicted_classes(ord_model, meta, test_ds)
    y_pred_sig = _predicted_classes(sig_model, meta, test_ds)
    cm_ord = _confusion(y_true, y_pred_ord)
    cm_sig = _confusion(y_true, y_pred_sig)
    prec_ord, rec_ord, f1_ord = _per_class_metrics(cm_ord)
    prec_sig, rec_sig, f1_sig = _per_class_metrics(cm_sig)

    print("\n=== Per-class metrics ===")
    print(f"  {'class':<6} {'support':>7} | "
          f"{'P_ord':>6} {'R_ord':>6} {'F1_ord':>7} | "
          f"{'P_sig':>6} {'R_sig':>6} {'F1_sig':>7} | {'ΔF1':>7}")
    rows = []
    for k in range(5):
        support = int(cm_ord[k, :].sum())
        delta_f1 = f1_ord[k] - f1_sig[k]
        print(f"  {CLASSES[k]:<6} {support:>7d} | "
              f"{prec_ord[k]:>6.3f} {rec_ord[k]:>6.3f} {f1_ord[k]:>7.3f} | "
              f"{prec_sig[k]:>6.3f} {rec_sig[k]:>6.3f} {f1_sig[k]:>7.3f} | "
              f"{delta_f1:>+7.3f}")
        rows.append({
            "class": CLASSES[k], "support": support,
            "precision_ordinal": f"{prec_ord[k]:.4f}",
            "recall_ordinal":    f"{rec_ord[k]:.4f}",
            "f1_ordinal":        f"{f1_ord[k]:.4f}",
            "precision_sigmoid": f"{prec_sig[k]:.4f}",
            "recall_sigmoid":    f"{rec_sig[k]:.4f}",
            "f1_sigmoid":        f"{f1_sig[k]:.4f}",
            "delta_f1":          f"{delta_f1:+.4f}",
        })
    macro_f1_ord = float(f1_ord.mean())
    macro_f1_sig = float(f1_sig.mean())
    print(f"\n  macro-F1: ordinal={macro_f1_ord:.4f}, sigmoid={macro_f1_sig:.4f}, "
          f"Δ={macro_f1_ord - macro_f1_sig:+.4f}")

    # CSV
    with (out_root / "per_class_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Confusion matrix figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    im0 = _plot_confusion(axes[0], cm_sig, "gated + sigmoid")
    im1 = _plot_confusion(axes[1], cm_ord, "gated + ordinal (proposed)")
    fig.colorbar(im1, ax=axes, fraction=0.04, pad=0.02,
                 label="row-normalised count (recall per row)")
    fig.suptitle(f"Confusion matrices on u1.test (cell text = absolute count, normalised)",
                 fontsize=11)
    fig.savefig(out_root / "confusion_grid.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Per-class F1 bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(5)
    width = 0.36
    ax.bar(x - width / 2, f1_sig, width, label="gated+sigmoid", color="#a0aec0")
    ax.bar(x + width / 2, f1_ord, width, label="gated+ordinal (proposed)", color="#2c5282")
    for i, k in enumerate(range(5)):
        delta = f1_ord[k] - f1_sig[k]
        ymax = max(f1_ord[k], f1_sig[k])
        ax.text(x[i], ymax + 0.01, f"Δ{delta:+.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_xlabel("rating class")
    ax.set_ylabel("F1 score")
    ax.set_title("Per-class F1 on u1.test")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(f1_ord.max(), f1_sig.max()) * 1.15)
    fig.tight_layout()
    fig.savefig(out_root / "per_class_f1.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Metadata
    meta_out = {
        "seed": args.seed,
        "split": cfg["split"],
        "ordinal_train_wall_s": ord_t,
        "sigmoid_train_wall_s": sig_t,
        "macro_f1_ordinal": macro_f1_ord,
        "macro_f1_sigmoid": macro_f1_sig,
        "macro_f1_delta": macro_f1_ord - macro_f1_sig,
        "support_per_class": [int(cm_ord[k, :].sum()) for k in range(5)],
    }
    with (out_root / "results.json").open("w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"\nWrote {out_root}/confusion_grid.png, per_class_f1.png, "
          f"per_class_metrics.csv, results.json")


if __name__ == "__main__":
    main()
