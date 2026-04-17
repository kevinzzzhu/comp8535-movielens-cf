"""Embedding visualisation: PCA, MDS, IsoMap + silhouette scores."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, Isomap
from sklearn.metrics import silhouette_score


def project(X: np.ndarray, method: str = "isomap", n_neighbors: int = 10):
    if method == "pca":
        return PCA(n_components=2).fit_transform(X)
    if method == "mds":
        return MDS(n_components=2, dissimilarity="euclidean", random_state=0, n_init=1).fit_transform(X)
    if method == "isomap":
        return Isomap(n_components=2, n_neighbors=n_neighbors).fit_transform(X)
    raise ValueError(f"unknown method: {method}")


def silhouette(Y: np.ndarray, labels: np.ndarray) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    return float(silhouette_score(Y, labels))


def scatter(Y: np.ndarray, labels: np.ndarray, title: str, save: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for lab in sorted(set(labels)):
        mask = labels == lab
        ax.scatter(Y[mask, 0], Y[mask, 1], s=24, alpha=0.75, label=str(lab))
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.legend(fontsize=7, loc="best", ncol=2)
    fig.tight_layout()
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save, dpi=160)
    plt.close(fig)
