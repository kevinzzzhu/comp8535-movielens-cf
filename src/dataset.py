"""MovieLens-100K loader with user/item auxiliary features."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


@dataclass
class MovieLensMeta:
    n_users: int
    n_items: int
    user_feat_dim: int
    item_feat_dim: int
    user_features: torch.Tensor  # (n_users, user_feat_dim)
    item_features: torch.Tensor  # (n_items, item_feat_dim)


def _load_users(path: Path) -> pd.DataFrame:
    cols = ["user_id", "age", "gender", "occupation", "zip"]
    return pd.read_csv(path, sep="|", names=cols, encoding="latin-1")


def _load_items(path: Path) -> pd.DataFrame:
    cols = ["item_id", "title", "release", "video_release", "imdb"] + GENRES
    return pd.read_csv(path, sep="|", names=cols, encoding="latin-1")


def _load_ratings(path: Path) -> pd.DataFrame:
    cols = ["user_id", "item_id", "rating", "timestamp"]
    return pd.read_csv(path, sep="\t", names=cols)


def build_metadata(data_dir: Path) -> MovieLensMeta:
    users = _load_users(data_dir / "u.user")
    items = _load_items(data_dir / "u.item")

    n_users = int(users["user_id"].max())
    n_items = int(items["item_id"].max())

    age = users["age"].to_numpy(dtype=np.float32).reshape(-1, 1)
    age = (age - age.mean()) / (age.std() + 1e-8)
    gender = pd.get_dummies(users["gender"]).to_numpy(dtype=np.float32)
    occupation = pd.get_dummies(users["occupation"]).to_numpy(dtype=np.float32)
    user_feat_np = np.concatenate([age, gender, occupation], axis=1)
    user_features = torch.zeros((n_users, user_feat_np.shape[1]), dtype=torch.float32)
    user_features[users["user_id"].to_numpy() - 1] = torch.from_numpy(user_feat_np)

    genre_np = items[GENRES].to_numpy(dtype=np.float32)
    item_features = torch.zeros((n_items, genre_np.shape[1]), dtype=torch.float32)
    item_features[items["item_id"].to_numpy() - 1] = torch.from_numpy(genre_np)

    return MovieLensMeta(
        n_users=n_users,
        n_items=n_items,
        user_feat_dim=user_features.shape[1],
        item_feat_dim=item_features.shape[1],
        user_features=user_features,
        item_features=item_features,
    )


class RatingDataset(Dataset):
    def __init__(self, ratings_path: Path):
        df = _load_ratings(ratings_path)
        self.user_idx = torch.from_numpy(df["user_id"].to_numpy() - 1).long()
        self.item_idx = torch.from_numpy(df["item_id"].to_numpy() - 1).long()
        self.rating = torch.from_numpy(df["rating"].to_numpy(dtype=np.float32))

    def __len__(self) -> int:
        return self.rating.shape[0]

    def __getitem__(self, i: int):
        return self.user_idx[i], self.item_idx[i], self.rating[i]


def load_split(data_dir: Path, split: str = "u1") -> tuple[RatingDataset, RatingDataset]:
    train = RatingDataset(data_dir / f"{split}.base")
    test = RatingDataset(data_dir / f"{split}.test")
    return train, test


class _RatingSubset:
    """Index-based view over a RatingDataset; duck-types as a Dataset.

    Exposes the same `user_idx`, `item_idx`, `rating` tensors that the rest of
    the codebase pulls off `RatingDataset` (e.g. OrdinalHead's
    `_thresholds_from_ratings`), so swapping a subset in for a full split
    requires no further plumbing changes.
    """

    def __init__(self, parent: RatingDataset, idx: np.ndarray):
        idx_t = torch.as_tensor(idx, dtype=torch.long)
        self.user_idx = parent.user_idx[idx_t]
        self.item_idx = parent.item_idx[idx_t]
        self.rating = parent.rating[idx_t]

    def __len__(self) -> int:
        return self.rating.shape[0]

    def __getitem__(self, i: int):
        return self.user_idx[i], self.item_idx[i], self.rating[i]


def load_split_with_val(
    data_dir: Path,
    split: str = "u1",
    val_frac: float = 0.1,
    val_seed: int = 0,
) -> tuple[_RatingSubset, _RatingSubset, RatingDataset]:
    """Load (train, val, test) where val is a `val_frac` slice of `<split>.base`.

    `val_seed` is independent of any training seed, so all training-seed
    replicates share the same val split (otherwise val-split variance would be
    conflated with model-init variance). Default `val_seed=0` is fixed across
    the project.
    """
    train_full = RatingDataset(data_dir / f"{split}.base")
    test = RatingDataset(data_dir / f"{split}.test")
    n = len(train_full)
    n_val = int(round(n * val_frac))
    rng = np.random.default_rng(val_seed)
    perm = rng.permutation(n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train = _RatingSubset(train_full, train_idx)
    val = _RatingSubset(train_full, val_idx)
    return train, val, test
