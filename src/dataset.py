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
