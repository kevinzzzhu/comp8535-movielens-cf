# COMP8535 — MovieLens Collaborative Filtering

Group research project for ENGN/COMP8535 Engineering Data Analytics (ANU, Semester 1 2026).

Proposed method: matrix factorisation with **gated auxiliary fusion** and a **cumulative-link ordinal regression head**, with **IsoMap** visualisation of learned embeddings. Evaluated on MovieLens-100K (`u1` split) against SVD, MF, and NMF baselines.

## Setup

```bash
uv sync                          # install pinned deps from uv.lock
uv run python -m src.main        # run baselines + proposed
```

Dataset lives under `data/ml-100k/` (checked in — ~4 MB; GroupLens license permits research use).

## Layout

```
config/       YAML experiment configs
data/         MovieLens-100K raw files
src/          dataset.py, model.py, baselines.py, train.py, visualize.py, main.py
log/          run outputs (json, checkpoints) — gitignored
Fig/          figures for paper — gitignored
paper/        NeurIPS-style report source
tests/        (to add) unit tests for fusion / ordinal head
scripts/      helper shell scripts
```

## Reproduction

Hyperparameters (fixed for fair comparison with published benchmarks):
- `d = 128`, Adam `lr=1e-3`, weight decay `1e-5`
- batch 64, 30 epochs, early-stop patience 10 (ablation) / 30 (headline)
- split: `u1.base` / `u1.test` (single split) or `u1`–`u5` (multi-split CV)
- seeds: {42, 43, 44}; val-split seed 0 (independent of training seed)

### Quick recipes

```bash
# 18-cell ablation matrix on u1                  ~10 min
PYTHONPATH=. uv run python scripts/run_ablations.py

# Sensitivity sweep over d, lambda                ~15 min
PYTHONPATH=. uv run python scripts/run_sensitivity.py

# Full 5-split × 6 cells × 3 seeds = 90 runs      ~60 min
PYTHONPATH=. uv run python scripts/run_multisplit.py

# Manifold visualisation (PCA / MDS / IsoMap)     ~1 min
PYTHONPATH=. uv run python scripts/run_visualization.py

# Cold-start / gate-trajectory analysis           ~2 min
PYTHONPATH=. uv run python scripts/run_coldstart.py

# Per-class confusion matrix and F1               ~2 min
PYTHONPATH=. uv run python scripts/run_classmetrics.py

# Logit vs probit cumulative-link comparison      ~10 min
PYTHONPATH=. uv run python scripts/run_link_compare.py

# Per-dimension gate distribution + stratification ~1 min
PYTHONPATH=. uv run python scripts/run_gate_analysis.py

# MovieLens-1M scale-up (3 cells × 3 seeds)        ~75 min
#  prerequisite: download ml-1m (24 MB) from grouplens.org
curl -L -o /tmp/ml-1m.zip https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip -q /tmp/ml-1m.zip -d data/
PYTHONPATH=. uv run python scripts/run_ml1m.py
```

## Ablations

Toggle `fusion ∈ {none, additive, gated}` and `head ∈ {sigmoid, ordinal}` in `config/config.yaml`. 2×3 = 6 cells.

For probit (instead of logit) cumulative link, pass `ordinal_link="probit"` to `CFGatedOrdinal`.
