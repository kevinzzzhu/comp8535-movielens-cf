"""Entry point: run baselines, proposed model, ablations, and visualisation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.baselines import MFBias, svd_baseline
from src.dataset import build_metadata, load_split, load_split_with_val
from src.model import CFGatedOrdinal
from src.train import TrainConfig, evaluate, set_seed, train_model


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def run(cfg_path: Path, log_dir: Path):
    cfg = load_config(cfg_path)
    set_seed(cfg["seed"])

    data_dir = Path(cfg["data_dir"])
    meta = build_metadata(data_dir)
    train_ds, val_ds, test_ds = load_split_with_val(data_dir, split=cfg["split"])

    protocol = cfg.get("protocol", "headline")
    patience_key = f"patience_{protocol}"
    if patience_key not in cfg:
        raise KeyError(f"config missing {patience_key} for protocol={protocol!r}")
    tcfg = TrainConfig(
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        patience=cfg[patience_key],
        device=cfg["device"],
        seed=cfg["seed"],
    )
    print(f"Protocol: {protocol}  (patience={tcfg.patience}, epochs={tcfg.epochs})")
    log_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    # --- SVD (non-learning)
    pred_fn = svd_baseline(train_ds, meta.n_users, meta.n_items, rank=cfg["svd_rank"])
    u = test_ds.user_idx.numpy()
    i = test_ds.item_idx.numpy()
    r = test_ds.rating.numpy()
    svd_pred = pred_fn(u, i)
    svd_rounded = np.clip(np.round(svd_pred), 1.0, 5.0)
    results["svd"] = {
        "rmse": float(np.sqrt(np.mean((svd_pred - r) ** 2))),
        "mae": float(np.mean(np.abs(svd_pred - r))),
        "acc": float(np.mean(svd_rounded == r)),
    }

    # --- MF
    mf = MFBias(meta.n_users, meta.n_items, embed_dim=cfg["embed_dim"], non_negative=False)
    results["mf"] = train_model(mf, train_ds, val_ds, test_ds, tcfg, use_features=False)

    # --- NMF
    nmf = MFBias(meta.n_users, meta.n_items, embed_dim=cfg["embed_dim"], non_negative=True)
    results["nmf"] = train_model(nmf, train_ds, val_ds, test_ds, tcfg, use_features=False)

    # --- Proposed: gated fusion + ordinal head
    proposed = CFGatedOrdinal(
        n_users=meta.n_users,
        n_items=meta.n_items,
        user_feat_dim=meta.user_feat_dim,
        item_feat_dim=meta.item_feat_dim,
        embed_dim=cfg["embed_dim"],
        fusion=cfg["fusion"],
        head=cfg["head"],
        train_ratings=train_ds.rating if cfg["head"] == "ordinal" else None,
    )
    results["proposed"] = train_model(
        proposed, train_ds, val_ds, test_ds, tcfg,
        user_features=meta.user_features,
        item_features=meta.item_features,
        use_features=True,
        log_gate=(cfg["fusion"] == "gated"),
    )

    # --- Ablation: gated fusion + sigmoid head (isolates the fusion contribution
    # under the sigmoid output head).
    if cfg.get("run_gated_sigmoid_ablation", True):
        gated_sigmoid = CFGatedOrdinal(
            n_users=meta.n_users,
            n_items=meta.n_items,
            user_feat_dim=meta.user_feat_dim,
            item_feat_dim=meta.item_feat_dim,
            embed_dim=cfg["embed_dim"],
            fusion="gated",
            head="sigmoid",
        )
        results["gated_sigmoid"] = train_model(
            gated_sigmoid, train_ds, val_ds, test_ds, tcfg,
            user_features=meta.user_features,
            item_features=meta.item_features,
            use_features=True,
            log_gate=True,
        )

    out = log_dir / "results.json"
    with out.open("w") as f:
        json.dump({k: _serialize(v) for k, v in results.items()}, f, indent=2)
    print(f"Wrote {out}")
    print(f"  {'model':<12} {'RMSE':>8} {'MAE':>8} {'Acc':>8} {'NLL':>8}")
    for k, v in results.items():
        # Prefer metrics from the best-RMSE epoch (weights we actually kept) rather than
        # history[-1], which reflects the final epoch — can be heavily overfit when
        # patience disables early stopping (see decisions log 2026-04-20).
        snap = v.get("best_metrics") or (v.get("history", [{}])[-1] if "history" in v else v)
        rmse = v.get("best_rmse", v.get("rmse"))
        mae = snap.get("mae", float("nan"))
        acc = snap.get("acc", float("nan"))
        nll = snap.get("nll", float("nan"))
        print(f"  {k:<12} {rmse:>8.4f} {mae:>8.4f} {acc:>8.4f} {nll:>8.4f}")


def _serialize(x):
    if isinstance(x, dict):
        return {k: _serialize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_serialize(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--log", type=Path, default=Path("log/run"))
    args = parser.parse_args()
    run(args.config, args.log)
