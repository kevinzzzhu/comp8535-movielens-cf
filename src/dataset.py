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


# -------------------------- MovieLens-1M ---------------------------------

class _RawRatings:
    """Wraps in-memory rating tensors so they expose the RatingDataset surface."""

    def __init__(self, user_idx: torch.Tensor, item_idx: torch.Tensor, rating: torch.Tensor):
        self.user_idx = user_idx
        self.item_idx = item_idx
        self.rating = rating

    def __len__(self) -> int:
        return self.rating.shape[0]

    def __getitem__(self, i: int):
        return self.user_idx[i], self.item_idx[i], self.rating[i]


def _load_ml1m_files(data_dir: Path):
    """Return raw (users_df, items_df, ratings_df) for MovieLens-1M."""
    users = pd.read_csv(
        data_dir / "users.dat", sep="::", engine="python",
        names=["user_id", "gender", "age", "occupation", "zip"], encoding="latin-1",
    )
    items = pd.read_csv(
        data_dir / "movies.dat", sep="::", engine="python",
        names=["item_id", "title", "genres"], encoding="latin-1",
    )
    ratings = pd.read_csv(
        data_dir / "ratings.dat", sep="::", engine="python",
        names=["user_id", "item_id", "rating", "timestamp"], encoding="latin-1",
    )
    return users, items, ratings


def build_ml1m_metadata(data_dir: Path) -> MovieLensMeta:
    """ML-1M side info: gender (one-hot), 7-bucket age (one-hot), 21-class
    occupation (one-hot) for users; multi-hot genre for items (vocabulary
    derived from the data)."""
    users, items, _ = _load_ml1m_files(data_dir)
    n_users = int(users["user_id"].max())
    n_items = int(items["item_id"].max())

    # Age in ML-1M is already bucketed into 7 categories; we one-hot rather
    # than z-score it (no continuous ordering assumed).
    gender = pd.get_dummies(users["gender"]).to_numpy(dtype=np.float32)
    age_oh = pd.get_dummies(users["age"], prefix="age").to_numpy(dtype=np.float32)
    occ = pd.get_dummies(users["occupation"], prefix="occ").to_numpy(dtype=np.float32)
    user_feat_np = np.concatenate([gender, age_oh, occ], axis=1)
    user_features = torch.zeros((n_users, user_feat_np.shape[1]), dtype=torch.float32)
    user_features[users["user_id"].to_numpy() - 1] = torch.from_numpy(user_feat_np)

    # Build genre vocabulary from the data; items.genres is "|"-separated.
    genre_lists = items["genres"].fillna("").str.split("|").tolist()
    vocab: list[str] = sorted({g for row in genre_lists for g in row if g})
    g2i = {g: k for k, g in enumerate(vocab)}
    item_feat_np = np.zeros((len(items), len(vocab)), dtype=np.float32)
    for r, gl in enumerate(genre_lists):
        for g in gl:
            if g:
                item_feat_np[r, g2i[g]] = 1.0
    item_features = torch.zeros((n_items, item_feat_np.shape[1]), dtype=torch.float32)
    item_features[items["item_id"].to_numpy() - 1] = torch.from_numpy(item_feat_np)

    return MovieLensMeta(
        n_users=n_users,
        n_items=n_items,
        user_feat_dim=user_features.shape[1],
        item_feat_dim=item_features.shape[1],
        user_features=user_features,
        item_features=item_features,
    )


def load_ml1m_split_with_val(
    data_dir: Path,
    split_seed: int = 0,
    test_frac: float = 0.1,
    val_frac: float = 0.1,
):
    """ML-1M does not ship canonical CV splits, so we build a deterministic
    random 80/10/10 split (train / val / test) keyed by `split_seed`.

    Both `val` and `test` are 10% of the full ratings; the remaining 80% is
    train. Returns objects that duck-type as `RatingDataset`.
    """
    _, _, ratings = _load_ml1m_files(data_dir)
    user_idx = torch.from_numpy(ratings["user_id"].to_numpy() - 1).long()
    item_idx = torch.from_numpy(ratings["item_id"].to_numpy() - 1).long()
    rating = torch.from_numpy(ratings["rating"].to_numpy(dtype=np.float32))
    n = rating.shape[0]
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test_i = perm[:n_test]
    val_i = perm[n_test:n_test + n_val]
    train_i = perm[n_test + n_val:]

    def _slice(i):
        idx = torch.as_tensor(i, dtype=torch.long)
        return _RawRatings(user_idx[idx], item_idx[idx], rating[idx])

    return _slice(train_i), _slice(val_i), _slice(test_i)
