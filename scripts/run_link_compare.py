"""Tier 2 ablation: logit vs probit cumulative-link function on the ordinal head.

Runs gated+ordinal across u1..u5 with three seeds each, twice: once with the
logit link (default; what we use throughout the paper) and once with the probit
link (the parameterisation used by OPRFM, Zaman & Jana 2025). Reports
across-split mean ± std for both, plus the across-split delta.

The logit numbers are already cached in `results/2026-04-27_multisplit/` and
are reused without retraining; only the probit runs are new (~10 min wall).

Outputs:
    results/<date>_link_compare/<link>/<split>_seed<seed>/results.json
    results/<date>_link_compare/link_compare_summary.csv
    results/<date>_link_compare/results.json
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
LINKS = ["logit", "probit"]
DEFAULT_SEEDS = [42, 43, 44]


def _serialize(x):
    if isinstance(x, dict):
        return {k: _serialize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_serialize(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    return x


def run_one(cfg: dict, link: str, split: str, seed: int, out_dir: Path) -> dict:
    """Train one (link, split, seed) cell. Cached by results.json existence."""
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
        embed_dim=cfg["embed_dim"], fusion="gated", head="ordinal",
        train_ratings=train_ds.rating, ordinal_link=link,
    )
    t0 = time.perf_counter()
    result = train_model(
        model, train_ds, val_ds, test_ds, tcfg,
        user_features=meta.user_features, item_features=meta.item_features,
        use_features=True, log_gate=True,
    )
    result["wall_time_s"] = time.perf_counter() - t0
    result["config"] = {"link": link, "split": split, "seed": seed,
                         "fusion": "gated", "head": "ordinal"}
    out_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(_serialize(result), f, indent=2)
    return result


def _load_logit_from_multisplit(multisplit_dir: Path, splits: list[str], seeds: list[int]
                                 ) -> list[dict]:
    """Load existing gated+ordinal logit results from results/<date>_multisplit/.

    The multi-split sweep already trained gated+ordinal with the default logit
    link on each (split, seed). Loading those JSONs avoids ~10 min of redundant
    compute and guarantees byte-identical comparison.
    """
    out = []
    for split in splits:
        for seed in seeds:
            p = multisplit_dir / split / f"gated_ordinal_seed{seed}" / "results.json"
            if not p.exists():
                raise FileNotFoundError(f"missing logit cache: {p}")
            with p.open() as f:
                r = json.load(f)
            r["config"] = {"link": "logit", "split": split, "seed": seed,
                            "fusion": "gated", "head": "ordinal"}
            out.append(r)
    return out


def _msd(xs):
    xs = [x for x in xs if x == x]
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return mean(xs), stdev(xs)


def aggregate(rows: list[dict], out_path: Path) -> None:
    """Across-split aggregation: per-split mean first, then mean ± std of those."""
    by_split: dict[tuple, list[dict]] = {}
    for r in rows:
        c = r["config"]
        by_split.setdefault((c["link"], c["split"]), []).append(r)
    per_split: dict[tuple, dict[str, float]] = {}
    for (link, split), runs in by_split.items():
        per_split[(link, split)] = {
            "rmse": _msd([r["best_rmse"] for r in runs])[0],
            "mae":  _msd([r["best_metrics"].get("mae", float("nan")) for r in runs])[0],
            "acc":  _msd([r["best_metrics"].get("acc", float("nan")) for r in runs])[0],
            "nll":  _msd([r["best_metrics"].get("nll", float("nan")) for r in runs])[0],
        }
    by_link: dict[str, list[dict[str, float]]] = {}
    for (link, _split), m in per_split.items():
        by_link.setdefault(link, []).append(m)

    fields = ["link", "n_splits",
              "rmse_mean", "rmse_std", "mae_mean", "mae_std",
              "acc_mean", "acc_std", "nll_mean", "nll_std", "per_split_rmse"]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for link in sorted(by_link.keys()):
            ms = by_link[link]
            rmses = [m["rmse"] for m in ms]
            maes  = [m["mae"]  for m in ms]
            accs  = [m["acc"]  for m in ms]
            nlls  = [m["nll"]  for m in ms]
            rmse_m, rmse_s = _msd(rmses)
            mae_m, mae_s = _msd(maes)
            acc_m, acc_s = _msd(accs)
            nll_m, nll_s = _msd(nlls)
            w.writerow({
                "link": link, "n_splits": len(ms),
                "rmse_mean": f"{rmse_m:.4f}", "rmse_std": f"{rmse_s:.4f}",
                "mae_mean":  f"{mae_m:.4f}",  "mae_std":  f"{mae_s:.4f}",
                "acc_mean":  f"{acc_m:.4f}",  "acc_std":  f"{acc_s:.4f}",
                "nll_mean":  f"{nll_m:.4f}", "nll_std":  f"{nll_s:.4f}",
                "per_split_rmse": "[" + ", ".join(f"{r:.4f}" for r in rmses) + "]",
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    parser.add_argument("--multisplit-dir", type=Path,
                        default=Path("results/2026-04-27_multisplit"),
                        help="cached gated+ordinal logit runs to reuse")
    args = parser.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    out_root = args.out or Path(f"results/{date.today().isoformat()}_link_compare")
    out_root.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out_root / "config_snapshot.yaml")

    print("Loading cached logit results from multi-split archive …")
    try:
        logit_rows = _load_logit_from_multisplit(args.multisplit_dir, args.splits, args.seeds)
        print(f"  loaded {len(logit_rows)} logit runs (no retraining)")
    except FileNotFoundError as e:
        print(f"  cache miss ({e}); will retrain logit too")
        logit_rows = []
        for split in args.splits:
            for seed in args.seeds:
                run_dir = out_root / "logit" / f"{split}_seed{seed}"
                logit_rows.append(run_one(cfg, "logit", split, seed, run_dir))

    print("\nTraining probit cells …")
    probit_rows = []
    sweep_t0 = time.perf_counter()
    runs = [(split, seed) for split in args.splits for seed in args.seeds]
    for idx, (split, seed) in enumerate(runs, start=1):
        run_dir = out_root / "probit" / f"{split}_seed{seed}"
        cached = (run_dir / "results.json").exists()
        if cached:
            print(f"[{idx:>2}/{len(runs)}] probit/{split}_seed{seed}  (skip: cached)")
        else:
            print(f"[{idx:>2}/{len(runs)}] probit/{split}_seed{seed} …", end="", flush=True)
        result = run_one(cfg, "probit", split, seed, run_dir)
        if not cached:
            wall = result.get("wall_time_s", 0.0)
            rmse = result["best_rmse"]
            print(f" RMSE={rmse:.4f}  ({wall:.1f}s)")
        probit_rows.append(result)

    sweep_wall = time.perf_counter() - sweep_t0
    print(f"\nTraining wall: {sweep_wall:.1f}s ({sweep_wall/60:.1f} min)")

    all_rows = logit_rows + probit_rows
    summary_csv = out_root / "link_compare_summary.csv"
    aggregate(all_rows, summary_csv)
    print(f"Wrote {summary_csv}")

    # Print compact comparison.
    with summary_csv.open() as f:
        lines = list(csv.DictReader(f))
    if len(lines) == 2:
        l = next(r for r in lines if r["link"] == "logit")
        p = next(r for r in lines if r["link"] == "probit")
        print(f"\n{'metric':<6} {'logit':>20} {'probit':>20} {'Δ probit-logit':>16}")
        for metric, key in [("RMSE", "rmse"), ("MAE", "mae"),
                             ("Acc", "acc"), ("NLL", "nll")]:
            lm = float(l[f"{key}_mean"]); ls = float(l[f"{key}_std"])
            pm = float(p[f"{key}_mean"]); ps = float(p[f"{key}_std"])
            delta = pm - lm
            print(f"{metric:<6} {lm:>10.4f} ± {ls:.4f}   {pm:>10.4f} ± {ps:.4f}   {delta:>+10.4f}")


if __name__ == "__main__":
    main()
