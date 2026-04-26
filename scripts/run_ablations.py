"""Week 3 ablation sweep: 6 configs (fusion × head) × 3 seeds = 18 runs.

Configurations:
    fusion ∈ {none, additive, gated}
    head   ∈ {sigmoid, ordinal}
    seed   ∈ {42, 43, 44}

Protocol: `ablation` (patience=10) for compute efficiency. Headline numbers are
reproduced separately by `src/main.py` with `protocol: headline` (patience=30).

Outputs:
    results/<date>_ablations/<fusion>_<head>_seed<seed>/results.json
    results/<date>_ablations/ablation_summary.csv     (mean ± std per config)
    results/<date>_ablations/config_snapshot.yaml     (config used for sweep)

Usage:
    uv run python scripts/run_ablations.py
    uv run python scripts/run_ablations.py --out results/my_dir --seeds 42 43
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

FUSIONS = ["none", "additive", "gated"]
HEADS = ["sigmoid", "ordinal"]


def _serialize(x):
    if isinstance(x, dict):
        return {k: _serialize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_serialize(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    return x


def run_one(cfg: dict, fusion: str, head: str, seed: int, out_dir: Path) -> dict:
    """Run a single (fusion, head, seed) configuration. Returns metrics + wall time."""
    set_seed(seed)
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
        seed=seed,
    )

    model = CFGatedOrdinal(
        n_users=meta.n_users,
        n_items=meta.n_items,
        user_feat_dim=meta.user_feat_dim,
        item_feat_dim=meta.item_feat_dim,
        embed_dim=cfg["embed_dim"],
        fusion=fusion,
        head=head,
        train_ratings=train_ds.rating if head == "ordinal" else None,
    )

    t0 = time.perf_counter()
    result = train_model(
        model, train_ds, val_ds, test_ds, tcfg,
        user_features=meta.user_features,
        item_features=meta.item_features,
        use_features=True,
        log_gate=(fusion == "gated"),
    )
    wall_time = time.perf_counter() - t0

    result["wall_time_s"] = wall_time
    result["config"] = {"fusion": fusion, "head": head, "seed": seed}

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.json").open("w") as f:
        json.dump(_serialize(result), f, indent=2)
    return result


def aggregate(rows: list[dict], out_path: Path) -> None:
    """Aggregate per-run metrics into mean ± std per (fusion, head). Writes CSV."""
    by_cfg: dict[tuple, list[dict]] = {}
    for r in rows:
        c = r["config"]
        key = (c["fusion"], c["head"])
        by_cfg.setdefault(key, []).append(r)

    fields = ["fusion", "head", "n_seeds",
              "rmse_mean", "rmse_std", "mae_mean", "mae_std",
              "acc_mean", "acc_std", "nll_mean", "nll_std",
              "wall_time_mean_s"]

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (fusion, head), runs in sorted(by_cfg.items()):
            rmses = [r["best_rmse"] for r in runs]
            maes  = [r["best_metrics"].get("mae", float("nan")) for r in runs]
            accs  = [r["best_metrics"].get("acc", float("nan")) for r in runs]
            nlls  = [r["best_metrics"].get("nll", float("nan")) for r in runs]
            wall  = [r["wall_time_s"] for r in runs]

            def _msd(xs):
                xs = [x for x in xs if x == x]  # drop NaN
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
                "fusion": fusion, "head": head, "n_seeds": len(runs),
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
    parser.add_argument("--out", type=Path, default=None,
                        help="Output dir (default: results/<today>_ablations)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--fusions", nargs="+", default=FUSIONS,
                        choices=FUSIONS)
    parser.add_argument("--heads", nargs="+", default=HEADS, choices=HEADS)
    args = parser.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    out_root = args.out or Path(f"results/{date.today().isoformat()}_ablations")
    out_root.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out_root / "config_snapshot.yaml")

    runs = [(fusion, head, seed)
            for fusion in args.fusions
            for head in args.heads
            for seed in args.seeds]
    total = len(runs)
    print(f"Running {total} configurations: "
          f"{len(args.fusions)} fusions × {len(args.heads)} heads × {len(args.seeds)} seeds")
    print(f"Protocol: ablation (patience={cfg['patience_ablation']}, epochs={cfg['epochs']})")
    print(f"Output: {out_root}\n")

    all_results = []
    sweep_t0 = time.perf_counter()
    for idx, (fusion, head, seed) in enumerate(runs, start=1):
        tag = f"{fusion}_{head}_seed{seed}"
        print(f"[{idx:>2}/{total}] {tag} …", end="", flush=True)
        run_dir = out_root / tag
        result = run_one(cfg, fusion, head, seed, run_dir)
        m = result["best_metrics"] or {}
        rmse = result["best_rmse"]
        mae = m.get("mae", float("nan"))
        acc = m.get("acc", float("nan"))
        nll = m.get("nll", float("nan"))
        wall = result["wall_time_s"]
        nll_str = f"{nll:.4f}" if nll == nll else "—"
        print(f" RMSE={rmse:.4f}  MAE={mae:.4f}  Acc={acc:.4f}  NLL={nll_str}  ({wall:.1f}s)")
        all_results.append(result)

    sweep_wall = time.perf_counter() - sweep_t0
    print(f"\nTotal sweep time: {sweep_wall:.1f}s ({sweep_wall/60:.1f} min)")

    summary_csv = out_root / "ablation_summary.csv"
    aggregate(all_results, summary_csv)
    print(f"Aggregated summary: {summary_csv}")


if __name__ == "__main__":
    main()
