---
created: 2026-05-22
last_edited: 2026-05-22
tags: [comp8535, movielens, collaborative-filtering, reproducibility]
---

# Quantile-Initialised Ordinal Collaborative Filtering

This repository contains the source code for the paper:

> Quantile-Initialised Ordinal Collaborative Filtering with Gated Auxiliary Fusion

The code trains MovieLens collaborative-filtering baselines and the proposed
matrix-factorisation model with gated auxiliary fusion and a cumulative-link
ordinal head. It also regenerates the ablations, calibration plots, cold-start
analysis, embedding visualisations, and MovieLens-1M scale-up used in the paper.

## Repository Layout

```text
config/      Experiment configuration.
paper/       LaTeX paper source and paper figures.
scripts/     Reproduction scripts for each reported experiment.
src/         Dataset loading, baselines, model, training, and visualisation code.
tests/       Unit tests for model components.
```

The following paths are intentionally local-only and are ignored by Git:

```text
data/        Downloaded MovieLens datasets.
results/     Generated experiment outputs.
log/         Scratch run logs.
output/      Local backups and draft artifacts.
tmp/         Temporary files.
```

Planning notes, decision logs, literature-reading notes, and local working files
are also ignored so the repository remains a publishable source code artifact.

## Environment

The project uses Python 3.11+ and `uv`.

```bash
uv sync --group dev
PYTHONPATH=. uv run pytest
```

The lockfile pins the Python dependencies used for the reported experiments.
The default configuration runs on CPU for reproducibility.

## Data

MovieLens data is not committed because the GroupLens license does not permit
redistribution without separate permission. Download the datasets locally before
running the experiments.

```bash
mkdir -p data

# MovieLens-100K, required for all main experiments.
curl -L -o /tmp/ml-100k.zip \
  https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip -q /tmp/ml-100k.zip -d data/

# MovieLens-1M, required only for the scale-up experiment.
curl -L -o /tmp/ml-1m.zip \
  https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip -q /tmp/ml-1m.zip -d data/
```

After extraction, the expected paths are `data/ml-100k/u1.base` and
`data/ml-1m/ratings.dat`.

## Reproducing the Paper

All commands should be run from the repository root.

```bash
# Main single-split baselines and proposed model.
PYTHONPATH=. uv run python -m src.main

# MovieLens-100K five-split ablation matrix.
PYTHONPATH=. uv run python scripts/run_multisplit.py

# Hyperparameter sensitivity curves.
PYTHONPATH=. uv run python scripts/run_sensitivity.py

# Reliability diagrams and expected calibration error.
PYTHONPATH=. uv run python scripts/run_reliability.py

# Per-class F1 and confusion matrices.
PYTHONPATH=. uv run python scripts/run_classmetrics.py

# Embedding visualisation with PCA, MDS, and IsoMap.
PYTHONPATH=. uv run python scripts/run_visualization.py

# Cold-start and gate-behaviour analysis.
PYTHONPATH=. uv run python scripts/run_coldstart.py

# Logit-vs-probit cumulative-link comparison.
PYTHONPATH=. uv run python scripts/run_link_compare.py

# Per-dimension gate analysis.
PYTHONPATH=. uv run python scripts/run_gate_analysis.py

# MovieLens-1M scale-up.
PYTHONPATH=. uv run python scripts/run_ml1m.py
```

Generated outputs are written under timestamped directories in `results/`.
Those outputs are deliberately ignored; rerun the scripts to regenerate them.

## Report Build

The paper source is in `paper/main.tex`.

```bash
cd paper
latexmk -pdf main.tex
```

The checked-in figures under `paper/figures/` are the paper-ready rendered
figures. To replace them, regenerate the corresponding experiments and copy the
new plots into `paper/figures/`.

## Main Configuration

Default hyperparameters live in `config/config.yaml`:

```text
embedding dimension: 128
optimizer: Adam
learning rate: 1e-3
weight decay: 1e-5
training seeds: 42, 43, 44
MovieLens-100K splits: u1-u5
```

The main ablation crosses:

```text
fusion in {none, additive, gated}
head in {sigmoid, ordinal}
```

## Citation

If you use the datasets, cite the MovieLens dataset paper:

```bibtex
@article{harper2015movielens,
  title = {The MovieLens Datasets: History and Context},
  author = {Harper, F Maxwell and Konstan, Joseph A},
  journal = {ACM Transactions on Interactive Intelligent Systems},
  volume = {5},
  number = {4},
  pages = {1--19},
  year = {2015},
  doi = {10.1145/2827872}
}
```
