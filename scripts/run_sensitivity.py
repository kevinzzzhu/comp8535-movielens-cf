"""Week 3 sensitivity sweep on the winning config (fusion=gated, head=ordinal).

Two axes, varied independently:
    d ∈ {32, 64, 128, 256}     (embedding dimension)
    λ ∈ {1e-6, 1e-5, 1e-4}     (weight decay)

Each axis is swept with the other held at its config-default value. The center
point (d_default, λ_default) is run only once and shared between both axes.

Outputs:
    results/<date>_sensitivity/d<d>_lam<λ>_seed<seed>/results.json
    results/<date>_sensitivity/sensitivity_summary.csv
    results/<date>_sensitivity/config_snapshot.yaml

Usage:
    uv run python scripts/run_sensitivity.py
    PYTHONPATH=. uv run python scripts/run_sensitivity.py --seeds 42 43
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

from src.dataset import build_metadata, load_split, load_split_with_val
from src.model import CFGatedOrdinal
from src.train import TrainConfig, set_seed, train_model

D_VALUES = [32, 64, 128, 256]
LAM_VALUES = [1e-6, 1e-5, 1e-4]


def _serialize(x):
    if isinstance(x, dict):
        return {k: _serialize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_serialize(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    return x


def run_one(cfg: dict, d: int, lam: float, seed: int, out_dir: Path) -> dict:
    """Run a single (d, λ, seed) configuration on gated+ordinal. Returns metrics + wall time."""
    set_seed(seed)
    data_dir = Path(cfg["data_dir"])
    meta = build_metadata(data_dir)
    train_ds, val_ds, test_ds = load_split_with_val(data_dir, split=cfg["split"])

    tcfg = TrainConfig(
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        weight_decay=lam,
        patience=cfg["patience_ablation"],
        device=cfg["device"],
        seed=seed,
    )

    model = CFGatedOrdinal(
        n_users=meta.n_users,
        n_items=meta.n_items,
        user_feat_dim=meta.user_feat_dim,
        item_feat_dim=meta.item_feat_dim,
        embed_dim=d,
        fusion="gated",
        head="ordinal",
        train_ratings=train_ds.rating,
    )

    t0 = time.perf_counter()
    result = train_model(
        model, train_ds, val_ds, test_ds, tcfg,
        user_features=meta.user_features,
        item_features=meta.item_features,
        use_features=True,
        log_gate=True,
    )
    wall_time = time.perf_counter() - t0

    result["wall_time_s"] = wall_time
    result["config"] = {"fusion": "gated", "head": "ordinal",
                        "embed_dim": d, "weight_decay": lam, "seed": seed}

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.json").open("w") as f:
        json.dump(_serialize(result), f, indent=2)
    return result


def aggregate(rows: list[dict], out_path: Path) -> None:
    """Aggregate by (d, λ). Writes CSV with mean ± std per cell."""
    by_cfg: dict[tuple, list[dict]] = {}
    for r in rows:
        c = r["config"]
        key = (c["embed_dim"], c["weight_decay"])
        by_cfg.setdefault(key, []).append(r)

    fields = ["embed_dim", "weight_decay", "n_seeds",
              "rmse_mean", "rmse_std", "mae_mean", "mae_std",
              "acc_mean", "acc_std", "nll_mean", "nll_std",
              "wall_time_mean_s"]

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        # Sort by embed_dim then weight_decay
        for (d, lam), runs in sorted(by_cfg.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            rmses = [r["best_rmse"] for r in runs]
            maes  = [r["best_metrics"].get("mae", float("nan")) for r in runs]
            accs  = [r["best_metrics"].get("acc", float("nan")) for r in runs]
            nlls  = [r["best_metrics"].get("nll", float("nan")) for r in runs]
            wall  = [r["wall_time_s"] for r in runs]

            def _msd(xs):
                xs = [x for x in xs if x == x]
                if not xs:
                    return float("nan"), float("nan")
                if len(xs) == 1:
                    return xs[0], 0.0
                return mean(xs), stdev(xs)

            rmse_m, rmse_s = _msd(rmses)
            mae_m, mae_s = _msd(maes)
            acc_m, acc_s = _msd(accs)
            nll_m, nll_s = _msd(nlls)

            w.writerow({
                "embed_dim": d, "weight_decay": f"{lam:.0e}", "n_seeds": len(runs),
                "rmse_mean": f"{rmse_m:.4f}", "rmse_std": f"{rmse_s:.4f}",
                "mae_mean":  f"{mae_m:.4f}",  "mae_std":  f"{mae_s:.4f}",
                "acc_mean":  f"{acc_m:.4f}",  "acc_std":  f"{acc_s:.4f}",
                "nll_mean":  f"{nll_m:.4f}" if nll_m == nll_m else "—",
                "nll_std":   f"{nll_s:.4f}" if nll_s == nll_s else "—",
                "wall_time_mean_s": f"{mean(wall):.1f}",
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--ds", nargs="+", type=int, default=D_VALUES)
    parser.add_argument("--lams", nargs="+", type=float, default=LAM_VALUES)
    args = parser.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    d_default = cfg["embed_dim"]
    lam_default = cfg["weight_decay"]

    # Build run list: vary d at λ=default, then vary λ at d=default. Deduplicate the centre.
    points: set[tuple[int, float]] = set()
    for d in args.ds:
        points.add((d, lam_default))
    for lam in args.lams:
        points.add((d_default, lam))

    out_root = args.out or Path(f"results/{date.today().isoformat()}_sensitivity")
    out_root.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out_root / "config_snapshot.yaml")

    runs = [(d, lam, seed)
            for (d, lam) in sorted(points)
            for seed in args.seeds]
    total = len(runs)
    print(f"Running {total} configurations: "
          f"{len(points)} (d, λ) cells × {len(args.seeds)} seeds")
    print(f"Defaults: d={d_default}, λ={lam_default:.0e}")
    print(f"Protocol: gated+ordinal, ablation patience={cfg['patience_ablation']}, epochs={cfg['epochs']}")
    print(f"Output: {out_root}\n")

    all_results = []
    sweep_t0 = time.perf_counter()
    for idx, (d, lam, seed) in enumerate(runs, start=1):
        tag = f"d{d}_lam{lam:.0e}_seed{seed}"
        print(f"[{idx:>2}/{total}] {tag} …", end="", flush=True)
        result = run_one(cfg, d, lam, seed, out_root / tag)
        m = result["best_metrics"] or {}
        rmse = result["best_rmse"]
        nll = m.get("nll", float("nan"))
        wall = result["wall_time_s"]
        nll_str = f"{nll:.4f}" if nll == nll else "—"
        print(f" RMSE={rmse:.4f}  NLL={nll_str}  ({wall:.1f}s)")
        all_results.append(result)

    sweep_wall = time.perf_counter() - sweep_t0
    print(f"\nTotal sweep time: {sweep_wall:.1f}s ({sweep_wall/60:.1f} min)")

    summary_csv = out_root / "sensitivity_summary.csv"
    aggregate(all_results, summary_csv)
    print(f"Aggregated summary: {summary_csv}")


if __name__ == "__main__":
    main()
