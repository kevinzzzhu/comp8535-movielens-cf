"""Render an ablation/sensitivity sweep directory as paper-grade PNG + Markdown.

Reads `<run_dir>/ablation_summary.csv` (or `sensitivity_summary.csv`) and writes:
    <run_dir>/summary.png   — table figure for the paper
    <run_dir>/summary.md    — markdown table for the report
    <run_dir>/sensitivity_curves.png   — line plots (sensitivity sweeps only)

Usage:
    uv run python scripts/render_ablations.py results/2026-04-26_ablations
    uv run python scripts/render_ablations.py results/2026-04-26_sensitivity
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def _load(csv_path: Path) -> tuple[list[str], list[dict]]:
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return reader.fieldnames or [], rows


def _fmt_cell(mean: str, std: str) -> str:
    """'0.9108' + '0.0026' -> '0.9108 ± 0.0026'."""
    if not mean or mean == "—":
        return "—"
    if not std or std == "—":
        return mean
    return f"{mean} ± {std}"


def _winner_index(rows: list[dict], col: str) -> int | None:
    """Return index of row with lowest col value (NaN ignored). For metrics where lower is better."""
    best = None
    best_val = float("inf")
    for i, r in enumerate(rows):
        try:
            v = float(r.get(col, "nan"))
        except ValueError:
            continue
        if v == v and v < best_val:
            best_val = v
            best = i
    return best


def _winner_index_high(rows: list[dict], col: str) -> int | None:
    """Higher-is-better variant for accuracy."""
    best = None
    best_val = -float("inf")
    for i, r in enumerate(rows):
        try:
            v = float(r.get(col, "nan"))
        except ValueError:
            continue
        if v == v and v > best_val:
            best_val = v
            best = i
    return best


def render_ablation(run_dir: Path) -> None:
    csv_path = run_dir / "ablation_summary.csv"
    fields, rows = _load(csv_path)

    # Build display rows: fusion, head, RMSE±std, MAE±std, Acc±std, NLL±std
    display = []
    for r in rows:
        display.append({
            "Fusion": r["fusion"],
            "Head": r["head"],
            "RMSE": _fmt_cell(r["rmse_mean"], r["rmse_std"]),
            "MAE": _fmt_cell(r["mae_mean"], r["mae_std"]),
            "Acc": _fmt_cell(r["acc_mean"], r["acc_std"]),
            "NLL": _fmt_cell(r["nll_mean"], r["nll_std"]),
        })

    # Best per metric (lower-is-better for RMSE/MAE/NLL; higher-is-better for Acc)
    best_rmse = _winner_index(rows, "rmse_mean")
    best_mae  = _winner_index(rows, "mae_mean")
    best_acc  = _winner_index_high(rows, "acc_mean")
    best_nll  = _winner_index(rows, "nll_mean")

    # PNG
    cols = ["Fusion", "Head", "RMSE", "MAE", "Acc", "NLL"]
    cell_text = [[d[c] for c in cols] for d in display]
    n_rows = len(display)
    fig, ax = plt.subplots(figsize=(10.5, 1.6 + 0.45 * n_rows))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.6)
    for j in range(len(cols)):
        tbl[0, j].set_text_props(weight="bold")
    # Highlight winners per column
    metric_to_col = {"RMSE": (best_rmse, 2), "MAE": (best_mae, 3),
                     "Acc": (best_acc, 4), "NLL": (best_nll, 5)}
    for _, (best_idx, j) in metric_to_col.items():
        if best_idx is not None:
            tbl[best_idx + 1, j].set_facecolor("#dff0d8")
            tbl[best_idx + 1, j].set_text_props(weight="bold")
    title = f"Ablation matrix · MovieLens-100K (u1) · {run_dir.name}"
    ax.set_title(title, pad=14, fontsize=12)
    fig.tight_layout()
    fig.savefig(run_dir / "summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Markdown
    md = [f"# {run_dir.name}", "",
          "| Fusion | Head | RMSE | MAE | Accuracy | NLL |",
          "|---|---|---|---|---|---|"]
    for i, d in enumerate(display):
        cells = [d["Fusion"], d["Head"], d["RMSE"], d["MAE"], d["Acc"], d["NLL"]]
        # Mark winning cells with **bold**
        if i == best_rmse: cells[2] = f"**{cells[2]}**"
        if i == best_mae:  cells[3] = f"**{cells[3]}**"
        if i == best_acc:  cells[4] = f"**{cells[4]}**"
        if i == best_nll:  cells[5] = f"**{cells[5]}**"
        md.append("| " + " | ".join(cells) + " |")
    md += ["",
           "Bold = best per column (lower-is-better for RMSE/MAE/NLL, higher for Acc).",
           "Variance: 3 seeds {42, 43, 44}.",
           "Protocol: ablation (patience=10, max 30 epochs).",
           "",
           "Raw: `ablation_summary.csv` · per-run JSONs: `<fusion>_<head>_seed<N>/results.json`"]
    (run_dir / "summary.md").write_text("\n".join(md) + "\n")
    print(f"Wrote {run_dir}/summary.png and summary.md")


def render_sensitivity(run_dir: Path) -> None:
    csv_path = run_dir / "sensitivity_summary.csv"
    fields, rows = _load(csv_path)

    # Parse rows: split into d-axis sweep (λ at default) and λ-axis sweep (d at default)
    # Default values are inferred as the most common (or from config_snapshot.yaml if present)
    snap_path = run_dir / "config_snapshot.yaml"
    d_default, lam_default = 128, 1e-5
    if snap_path.exists():
        import yaml
        with snap_path.open() as f:
            snap = yaml.safe_load(f)
        d_default = int(snap.get("embed_dim", d_default))
        lam_default = float(snap.get("weight_decay", lam_default))

    def _parse_lam(s: str) -> float:
        return float(s)

    d_axis = [r for r in rows if _parse_lam(r["weight_decay"]) == lam_default]
    lam_axis = [r for r in rows if int(r["embed_dim"]) == d_default]

    d_axis.sort(key=lambda r: int(r["embed_dim"]))
    lam_axis.sort(key=lambda r: float(r["weight_decay"]))

    # ---- PNG table (full grid) ----
    cols = ["d", "λ", "RMSE", "MAE", "Acc", "NLL"]
    cell_text = []
    display_rows = []
    for r in rows:
        display_rows.append({
            "d": r["embed_dim"], "λ": r["weight_decay"],
            "RMSE": _fmt_cell(r["rmse_mean"], r["rmse_std"]),
            "MAE":  _fmt_cell(r["mae_mean"], r["mae_std"]),
            "Acc":  _fmt_cell(r["acc_mean"], r["acc_std"]),
            "NLL":  _fmt_cell(r["nll_mean"], r["nll_std"]),
        })
    cell_text = [[d[c] for c in cols] for d in display_rows]

    best_rmse = _winner_index(rows, "rmse_mean")
    best_mae  = _winner_index(rows, "mae_mean")
    best_acc  = _winner_index_high(rows, "acc_mean")
    best_nll  = _winner_index(rows, "nll_mean")

    fig, ax = plt.subplots(figsize=(10.5, 1.6 + 0.45 * len(rows)))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.6)
    for j in range(len(cols)):
        tbl[0, j].set_text_props(weight="bold")
    for best_idx, j in [(best_rmse, 2), (best_mae, 3), (best_acc, 4), (best_nll, 5)]:
        if best_idx is not None:
            tbl[best_idx + 1, j].set_facecolor("#dff0d8")
            tbl[best_idx + 1, j].set_text_props(weight="bold")
    ax.set_title(f"Sensitivity grid · gated+ordinal · MovieLens-100K (u1) · {run_dir.name}",
                 pad=14, fontsize=12)
    fig.tight_layout()
    fig.savefig(run_dir / "summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ---- Sensitivity curves PNG ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    # d-axis
    ax = axes[0]
    xs = [int(r["embed_dim"]) for r in d_axis]
    ys = [float(r["rmse_mean"]) for r in d_axis]
    es = [float(r["rmse_std"]) for r in d_axis]
    ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, color="#2c5282")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel("Embedding dimension d")
    ax.set_ylabel("Test RMSE")
    ax.set_title(f"Sensitivity to d (λ = {lam_default:.0e})")
    ax.grid(alpha=0.3)

    # λ-axis
    ax = axes[1]
    xs = [float(r["weight_decay"]) for r in lam_axis]
    ys = [float(r["rmse_mean"]) for r in lam_axis]
    es = [float(r["rmse_std"]) for r in lam_axis]
    ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, color="#742a2a")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:.0e}" for x in xs])
    ax.set_xlabel("Weight decay λ")
    ax.set_ylabel("Test RMSE")
    ax.set_title(f"Sensitivity to λ (d = {d_default})")
    ax.grid(alpha=0.3)

    fig.suptitle("Sensitivity curves · gated+ordinal · 3 seeds, error bars = ±1 std",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(run_dir / "sensitivity_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ---- Markdown ----
    md = [f"# {run_dir.name}", "",
          f"Defaults: d={d_default}, λ={lam_default:.0e}",
          "",
          "## Sweep grid", "",
          "| d | λ | RMSE | MAE | Accuracy | NLL |",
          "|---|---|---|---|---|---|"]
    for i, d in enumerate(display_rows):
        cells = [d["d"], d["λ"], d["RMSE"], d["MAE"], d["Acc"], d["NLL"]]
        if i == best_rmse: cells[2] = f"**{cells[2]}**"
        if i == best_mae:  cells[3] = f"**{cells[3]}**"
        if i == best_acc:  cells[4] = f"**{cells[4]}**"
        if i == best_nll:  cells[5] = f"**{cells[5]}**"
        md.append("| " + " | ".join(cells) + " |")
    md += ["",
           "Bold = best per column (RMSE/MAE/NLL min; Acc max).",
           "Variance: 3 seeds {42, 43, 44}.",
           "Protocol: gated+ordinal, ablation patience=10, max 30 epochs.",
           "",
           f"![Sensitivity curves](sensitivity_curves.png)",
           "",
           "Raw: `sensitivity_summary.csv` · per-run JSONs: `d<d>_lam<λ>_seed<N>/results.json`"]
    (run_dir / "summary.md").write_text("\n".join(md) + "\n")
    print(f"Wrote {run_dir}/summary.png, sensitivity_curves.png, and summary.md")


def main(run_dir: Path) -> None:
    if (run_dir / "ablation_summary.csv").exists():
        render_ablation(run_dir)
    elif (run_dir / "sensitivity_summary.csv").exists():
        render_sensitivity(run_dir)
    else:
        raise FileNotFoundError(
            f"{run_dir} contains neither ablation_summary.csv nor sensitivity_summary.csv")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
