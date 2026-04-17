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
- batch 64, 30 epochs, early-stop patience 5
- split: `u1.base` / `u1.test`
- seed: 42 (report mean ± std over 3 seeds in the paper)

## Ablations

Toggle `fusion ∈ {none, additive, gated}` and `head ∈ {sigmoid, ordinal}` in `config/config.yaml`. 2×3 = 6 cells.
