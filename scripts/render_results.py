"""Render a results.json run as a PNG snapshot and a markdown summary.

Usage: uv run python scripts/render_results.py results/<run_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def render(run_dir: Path) -> None:
    with (run_dir / "results.json").open() as f:
        results = json.load(f)

    def _fmt(v):
        if v is None or (isinstance(v, float) and v != v):  # NaN check
            return "—"
        return f"{v:.4f}" if isinstance(v, (int, float)) else "—"

    rows = []
    for name in list(results.keys()):
        r = results[name]
        if "history" in r:
            last = r["history"][-1]
            row = {
                "model": name,
                "rmse": _fmt(r.get("best_rmse")),
                "mae": _fmt(last.get("mae")),
                "acc": _fmt(last.get("acc")),
                "nll": _fmt(last.get("nll")),
                "epochs": len(r["history"]),
            }
        else:
            row = {
                "model": name,
                "rmse": _fmt(r.get("rmse")),
                "mae": _fmt(r.get("mae")),
                "acc": _fmt(r.get("acc")),
                "nll": "—",
                "epochs": "—",
            }
        rows.append(row)

    # PNG table
    fig, ax = plt.subplots(figsize=(8.5, 2.2 + 0.3 * max(0, len(rows) - 4)))
    ax.axis("off")
    cols = ["Model", "RMSE", "MAE", "Accuracy", "NLL", "Epochs"]
    cell = [[r["model"], r["rmse"], r["mae"], r["acc"], r["nll"], str(r["epochs"])] for r in rows]
    tbl = ax.table(cellText=cell, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.6)
    # Bold header
    for j in range(len(cols)):
        tbl[0, j].set_text_props(weight="bold")
    # Highlight any row whose model name contains "proposed"
    for r_idx, row in enumerate(rows, start=1):
        if "proposed" in row["model"].lower():
            for j in range(len(cols)):
                tbl[r_idx, j].set_facecolor("#e8f0ff")
    title = f"MovieLens-100K (u1 split) — {run_dir.name}"
    ax.set_title(title, pad=14)
    fig.tight_layout()
    fig.savefig(run_dir / "summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Markdown summary
    md = [f"# {run_dir.name}", "", "| Model | RMSE | MAE | Accuracy | NLL | Epochs |", "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['rmse']} | {r['mae']} | {r['acc']} | {r['nll']} | {r['epochs']} |")
    md += ["", "Config snapshot: `config_snapshot.yaml` · Raw: `results.json`"]
    (run_dir / "summary.md").write_text("\n".join(md) + "\n")
    print(f"Wrote {run_dir}/summary.png and summary.md")


if __name__ == "__main__":
    render(Path(sys.argv[1]))
