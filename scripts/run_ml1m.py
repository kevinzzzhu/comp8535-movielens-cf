"""Tier 2/3 scale-up: rerun the key ablation cells on MovieLens-1M.

Demonstrates that the proposed gated+ordinal model continues to win on a
$10\\!\\times$-larger benchmark with richer side information. We do not
reproduce the entire 90-run multi-split protocol on ML-1M (canonical CV
splits do not ship with the dataset and one full pass would take ~hours);
instead we run three representative cells under a single deterministic
random 80/10/10 split, with three training seeds:

  none + sigmoid       (MF analogue, no fusion)
  gated + sigmoid      (fusion alone)
  gated + ordinal      (proposed)

Outputs:
    results/<date>_ml1m/
        <fusion>_<head>_seed<seed>/results.json
        ml1m_summary.csv         per-cell mean ± std
        config_snapshot.yaml
        results.json             metadata (split sizes, train wall, etc.)

Usage:
    PYTHONPATH=. uv run python scripts/run_ml1m.py
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from datetime import date
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import yaml

from src.dataset import build_ml1m_metadata, load_ml1m_split_with_val
from src.model import CFGatedOrdinal
from src.train import TrainConfig, set_seed, train_model

CELLS = [
    ("none", "sigmoid"),
    ("gated", "sigmoid"),
    ("gated", "ordinal"),
]


def _serialize(x):
    if isinstance(x, dict):
        return {k: _serialize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_serialize(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    return x


def run_one(cfg: dict, fusion: str, head: str, seed: int, out_dir: Path,
            data_dir: Path) -> dict:
    out_path = out_dir / "results.json"
    if out_path.exists():
        with out_path.open() as f:
            return json.load(f)

    set_seed(seed)
    meta = build_ml1m_metadata(data_dir)
    train_ds, val_ds, test_ds = load_ml1m_split_with_val(
        data_dir, split_seed=cfg.get("ml1m_split_seed", 0),
        test_frac=cfg.get("ml1m_test_frac", 0.1),
        val_frac=cfg.get("ml1m_val_frac", 0.1),
    )

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

    t0 = time.perf_counter()
    result = train_model(
        model, train_ds, val_ds, test_ds, tcfg,
        user_features=meta.user_features, item_features=meta.item_features,
        use_features=True, log_gate=(fusion == "gated"),
    )
    wall_time = time.perf_counter() - t0
    result["wall_time_s"] = wall_time
    result["config"] = {
        "fusion": fusion, "head": head, "seed": seed,
        "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
        "n_users": meta.n_users, "n_items": meta.n_items,
        "user_feat_dim": meta.user_feat_dim, "item_feat_dim": meta.item_feat_dim,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(_serialize(result), f, indent=2)
    return result


def _msd(xs):
    xs = [x for x in xs if x == x]
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return mean(xs), stdev(xs)


def aggregate(rows: list[dict], out_path: Path) -> None:
    by = {}
    for r in rows:
        c = r["config"]
        by.setdefault((c["fusion"], c["head"]), []).append(r)

    fields = ["fusion", "head", "n_seeds",
              "rmse_mean", "rmse_std", "mae_mean", "mae_std",
              "acc_mean", "acc_std", "nll_mean", "nll_std",
              "wall_time_mean_s"]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (fusion, head), runs in sorted(by.items()):
            rmses = [r["best_rmse"] for r in runs]
            maes = [r["best_metrics"].get("mae", float("nan")) for r in runs]
            accs = [r["best_metrics"].get("acc", float("nan")) for r in runs]
            nlls = [r["best_metrics"].get("nll", float("nan")) for r in runs]
            walls = [r["wall_time_s"] for r in runs]
            rmse_m, rmse_s = _msd(rmses)
            mae_m, mae_s = _msd(maes)
            acc_m, acc_s = _msd(accs)
            nll_m, nll_s = _msd(nlls)
            w.writerow({
                "fusion": fusion, "head": head, "n_seeds": len(runs),
                "rmse_mean": f"{rmse_m:.4f}", "rmse_std": f"{rmse_s:.4f}",
                "mae_mean":  f"{mae_m:.4f}",  "mae_std":  f"{mae_s:.4f}",
                "acc_mean":  f"{acc_m:.4f}",  "acc_std":  f"{acc_s:.4f}",
                "nll_mean":  f"{nll_m:.4f}" if nll_m == nll_m else "—",
                "nll_std":   f"{nll_s:.4f}" if nll_s == nll_s else "—",
                "wall_time_mean_s": f"{mean(walls):.1f}",
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--data", type=Path, default=Path("data/ml-1m"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("ml1m_split_seed", 0)
    cfg.setdefault("ml1m_test_frac", 0.1)
    cfg.setdefault("ml1m_val_frac", 0.1)

    out_root = args.out or Path(f"results/{date.today().isoformat()}_ml1m")
    out_root.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out_root / "config_snapshot.yaml")

    runs = [(fusion, head, seed)
            for (fusion, head) in CELLS
            for seed in args.seeds]
    total = len(runs)
    print(f"Running {total} ML-1M configurations: "
          f"{len(CELLS)} cells × {len(args.seeds)} seeds")
    print(f"Protocol: ablation (patience={cfg['patience_ablation']}, "
          f"epochs={cfg['epochs']}). Random 80/10/10 split, "
          f"split_seed={cfg['ml1m_split_seed']}.")
    print(f"Output: {out_root}\n")

    all_results = []
    sweep_t0 = time.perf_counter()
    for idx, (fusion, head, seed) in enumerate(runs, start=1):
        tag = f"{fusion}_{head}_seed{seed}"
        run_dir = out_root / tag
        existed = (run_dir / "results.json").exists()
        if existed:
            print(f"[{idx:>2}/{total}] {tag}  (skip: cached)")
            with (run_dir / "results.json").open() as f:
                all_results.append(json.load(f))
            continue
        print(f"[{idx:>2}/{total}] {tag} …", end="", flush=True)
        result = run_one(cfg, fusion, head, seed, run_dir, args.data)
        m = result["best_metrics"] or {}
        rmse = result["best_rmse"]
        nll = m.get("nll", float("nan"))
        wall = result.get("wall_time_s", 0.0)
        nll_str = f"{nll:.4f}" if nll == nll else "—"
        print(f" RMSE={rmse:.4f}  NLL={nll_str}  ({wall:.1f}s)")
        all_results.append(result)

    sweep_wall = time.perf_counter() - sweep_t0
    print(f"\nTotal sweep time: {sweep_wall:.1f}s ({sweep_wall/60:.1f} min)")
    summary_csv = out_root / "ml1m_summary.csv"
    aggregate(all_results, summary_csv)
    print(f"Aggregated summary: {summary_csv}")


if __name__ == "__main__":
    main()
