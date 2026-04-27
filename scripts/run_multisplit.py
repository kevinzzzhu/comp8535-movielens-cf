"""Week 4 multi-split CV: ablation matrix across all five MovieLens-100K splits.

Runs the 6-cell ablation matrix on each of {u1, u2, u3, u4, u5}, with three
seeds per cell, giving 90 trained models. The MovieLens-100K ml-100k.zip ships
with five canonical 80/20 splits (u1.base/.test through u5.base/.test); the
across-split mean ± std is the gold-standard reporting protocol for the
dataset.

Aggregation policy (Option B: across-split-of-per-split-means):
  - For each (fusion, head, split): mean ± std over 3 seeds.
  - For each (fusion, head): mean of the five per-split means; std *of those
    five means*. This isolates between-split variance, which is the
    methodologically interesting quantity, from within-split seed noise.

Outputs:
  results/<date>_multisplit/<split>/<fusion>_<head>_seed<seed>/results.json
  results/<date>_multisplit/per_split_summary.csv     (one row per (cell, split))
  results/<date>_multisplit/multisplit_summary.csv    (one row per cell, across splits)
  results/<date>_multisplit/config_snapshot.yaml

Resumes mid-sweep: if a results.json already exists for a given (split, fusion,
head, seed), the run is skipped. Lets you interrupt and re-launch without losing
work.

Usage:
  PYTHONPATH=. uv run python scripts/run_multisplit.py
  PYTHONPATH=. uv run python scripts/run_multisplit.py --splits u1 u2  # subset
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

from src.dataset import build_metadata, load_split_with_val
from src.model import CFGatedOrdinal
from src.train import TrainConfig, set_seed, train_model

SPLITS = ["u1", "u2", "u3", "u4", "u5"]
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


def run_one(cfg: dict, split: str, fusion: str, head: str, seed: int, out_dir: Path) -> dict | None:
    """Run a single (split, fusion, head, seed) cell.

    If `out_dir/results.json` already exists, loads it and returns it (resume).
    Otherwise trains, evaluates, writes the JSON, and returns the result dict.
    """
    out_path = out_dir / "results.json"
    if out_path.exists():
        with out_path.open() as f:
            return json.load(f)

    set_seed(seed)
    data_dir = Path(cfg["data_dir"])
    meta = build_metadata(data_dir)
    train_ds, val_ds, test_ds = load_split_with_val(data_dir, split=split)

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
    result["config"] = {"split": split, "fusion": fusion, "head": head, "seed": seed}

    out_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(_serialize(result), f, indent=2)
    return result


def _msd(xs):
    """Mean and (sample) std of an iterable, ignoring NaN."""
    xs = [x for x in xs if x == x]
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return mean(xs), stdev(xs)


def aggregate_per_split(rows: list[dict], out_path: Path) -> None:
    """One row per (fusion, head, split), reporting mean ± std over seeds."""
    by = {}
    for r in rows:
        c = r["config"]
        key = (c["fusion"], c["head"], c["split"])
        by.setdefault(key, []).append(r)

    fields = ["fusion", "head", "split", "n_seeds",
              "rmse_mean", "rmse_std", "mae_mean", "mae_std",
              "acc_mean", "acc_std", "nll_mean", "nll_std"]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (fusion, head, split), runs in sorted(by.items()):
            rmse_m, rmse_s = _msd([r["best_rmse"] for r in runs])
            mae_m, mae_s = _msd([r["best_metrics"].get("mae", float("nan")) for r in runs])
            acc_m, acc_s = _msd([r["best_metrics"].get("acc", float("nan")) for r in runs])
            nll_m, nll_s = _msd([r["best_metrics"].get("nll", float("nan")) for r in runs])
            w.writerow({
                "fusion": fusion, "head": head, "split": split, "n_seeds": len(runs),
                "rmse_mean": f"{rmse_m:.4f}", "rmse_std": f"{rmse_s:.4f}",
                "mae_mean":  f"{mae_m:.4f}",  "mae_std":  f"{mae_s:.4f}",
                "acc_mean":  f"{acc_m:.4f}",  "acc_std":  f"{acc_s:.4f}",
                "nll_mean":  f"{nll_m:.4f}" if nll_m == nll_m else "—",
                "nll_std":   f"{nll_s:.4f}" if nll_s == nll_s else "—",
            })


def aggregate_across_splits(rows: list[dict], out_path: Path) -> None:
    """One row per (fusion, head), reporting mean ± std OF the per-split means.

    This is the canonical MovieLens-100K reporting: average over splits, std
    of the per-split averages (between-split variance).
    """
    # First: compute per-split means.
    per_split: dict[tuple[str, str, str], dict[str, float]] = {}
    by_split = {}
    for r in rows:
        c = r["config"]
        key = (c["fusion"], c["head"], c["split"])
        by_split.setdefault(key, []).append(r)
    for (fusion, head, split), runs in by_split.items():
        per_split[(fusion, head, split)] = {
            "rmse": _msd([r["best_rmse"] for r in runs])[0],
            "mae":  _msd([r["best_metrics"].get("mae", float("nan")) for r in runs])[0],
            "acc":  _msd([r["best_metrics"].get("acc", float("nan")) for r in runs])[0],
            "nll":  _msd([r["best_metrics"].get("nll", float("nan")) for r in runs])[0],
        }

    # Then aggregate across splits.
    by_cell = {}
    for (fusion, head, split), m in per_split.items():
        by_cell.setdefault((fusion, head), []).append(m)

    fields = ["fusion", "head", "n_splits",
              "rmse_mean", "rmse_std", "mae_mean", "mae_std",
              "acc_mean", "acc_std", "nll_mean", "nll_std",
              "per_split_rmse"]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (fusion, head), per_split_metrics in sorted(by_cell.items()):
            rmses = [m["rmse"] for m in per_split_metrics]
            maes  = [m["mae"]  for m in per_split_metrics]
            accs  = [m["acc"]  for m in per_split_metrics]
            nlls  = [m["nll"]  for m in per_split_metrics]
            rmse_m, rmse_s = _msd(rmses)
            mae_m, mae_s = _msd(maes)
            acc_m, acc_s = _msd(accs)
            nll_m, nll_s = _msd(nlls)
            per_split_str = "[" + ", ".join(f"{r:.4f}" for r in rmses) + "]"
            w.writerow({
                "fusion": fusion, "head": head, "n_splits": len(per_split_metrics),
                "rmse_mean": f"{rmse_m:.4f}", "rmse_std": f"{rmse_s:.4f}",
                "mae_mean":  f"{mae_m:.4f}",  "mae_std":  f"{mae_s:.4f}",
                "acc_mean":  f"{acc_m:.4f}",  "acc_std":  f"{acc_s:.4f}",
                "nll_mean":  f"{nll_m:.4f}" if nll_m == nll_m else "—",
                "nll_std":   f"{nll_s:.4f}" if nll_s == nll_s else "—",
                "per_split_rmse": per_split_str,
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    parser.add_argument("--fusions", nargs="+", default=FUSIONS, choices=FUSIONS)
    parser.add_argument("--heads", nargs="+", default=HEADS, choices=HEADS)
    args = parser.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    out_root = args.out or Path(f"results/{date.today().isoformat()}_multisplit")
    out_root.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out_root / "config_snapshot.yaml")

    runs = [(split, fusion, head, seed)
            for split in args.splits
            for fusion in args.fusions
            for head in args.heads
            for seed in args.seeds]
    total = len(runs)
    print(f"Running {total} configurations: "
          f"{len(args.splits)} splits × {len(args.fusions)} fusions × "
          f"{len(args.heads)} heads × {len(args.seeds)} seeds")
    print(f"Protocol: ablation patience={cfg['patience_ablation']}, "
          f"val-driven early stopping, val_frac=0.10, val_seed=0")
    print(f"Output: {out_root}")
    print(f"Resume: existing results.json files will be loaded, not retrained\n")

    all_results = []
    sweep_t0 = time.perf_counter()
    n_skipped = 0
    for idx, (split, fusion, head, seed) in enumerate(runs, start=1):
        tag = f"{split}/{fusion}_{head}_seed{seed}"
        run_dir = out_root / split / f"{fusion}_{head}_seed{seed}"
        existed = (run_dir / "results.json").exists()
        if existed:
            n_skipped += 1
            print(f"[{idx:>2}/{total}] {tag}  (skip: cached)")
            result = run_one(cfg, split, fusion, head, seed, run_dir)
        else:
            print(f"[{idx:>2}/{total}] {tag} …", end="", flush=True)
            result = run_one(cfg, split, fusion, head, seed, run_dir)
            m = result["best_metrics"] or {}
            rmse = result["best_rmse"]
            wall = result.get("wall_time_s", 0.0)
            nll = m.get("nll", float("nan"))
            nll_str = f"{nll:.4f}" if nll == nll else "—"
            print(f" RMSE={rmse:.4f}  NLL={nll_str}  ({wall:.1f}s)")
        all_results.append(result)

    sweep_wall = time.perf_counter() - sweep_t0
    print(f"\nFinished. New runs: {total - n_skipped}, cached: {n_skipped}.")
    print(f"Wall time: {sweep_wall:.1f}s ({sweep_wall/60:.1f} min)")

    per_split_csv = out_root / "per_split_summary.csv"
    multisplit_csv = out_root / "multisplit_summary.csv"
    aggregate_per_split(all_results, per_split_csv)
    aggregate_across_splits(all_results, multisplit_csv)
    print(f"Wrote {per_split_csv}")
    print(f"Wrote {multisplit_csv}")


if __name__ == "__main__":
    main()
