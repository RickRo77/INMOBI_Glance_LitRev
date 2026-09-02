"""
Dataset handling, data preprocessing, and PyTorch MovieLensMAML loader.
"""

import re
import random
import torch
from torch.utils.data import Dataset
import pandas as pd
from config import (
    ZIP_VOCAB, NUM_GENRES, GENRE_TO_IDX,
    SUPPORT_SIZE, QUERY_SIZE_AT_BUILD
)

def _hash_zip(z):
    digits = re.sub(r"\D", "", str(z))
    return int(digits[:5]) % ZIP_VOCAB if digits else 0

def _gender_int(g):
    return 1 if str(g).strip().upper() == "F" else 0

def _multihot(genre_str):
    vec = torch.zeros(NUM_GENRES)
    for g in genre_str.split("|"):
        idx = GENRE_TO_IDX.get(g.strip())
        if idx is not None:
            vec[idx] = 1.0
    return vec

def _make_episode(rows, user_feat, movie_feat):
    g, a, o, z = user_feat
    genres, ratings = [], []
    for mid, rat in rows:
        mf = movie_feat.get(mid)
        if mf is None:
            continue
        genres.append(mf)
        ratings.append(rat)
    n = len(ratings)
    if n == 0:
        return None
    return {
        "gender":         torch.full((n,), g, dtype=torch.long),
        "age":            torch.full((n,), a, dtype=torch.float),
        "occupation":     torch.full((n,), o, dtype=torch.long),
        "zipcode":        torch.full((n,), z, dtype=torch.long),
        "genre_multihot": torch.stack(genres),
        "rating":         torch.tensor(ratings, dtype=torch.float),
    }


class MovieLensMAML(Dataset):
    """
    Builds per-user tasks (support, query) following the MeLU paper.
      1. Query set is fixed at build time (QUERY_SIZE_AT_BUILD items per user)
      2. Users are split 80 / 20 into train_tasks / test_tasks so that the
         20% held-out users are true "new users" never seen during meta-training
         (cold-start evaluation, per paper §4).
    """
    def __init__(self, users_path, movies_path, ratings_path,
                 support_size=SUPPORT_SIZE,
                 query_size=QUERY_SIZE_AT_BUILD,
                 seed=42):
        super().__init__()
        self.support_size = support_size
        self.query_size   = query_size
        rng = random.Random(seed)

        users_df   = pd.read_csv(users_path,   sep="::", engine="python", header=None,
                                  names=["user_id","gender","age","occupation","zip_code"],
                                  encoding="latin-1")
        movies_df  = pd.read_csv(movies_path,  sep="::", engine="python", header=None,
                                  names=["movie_id","title","genres"],
                                  encoding="latin-1")
        ratings_df = pd.read_csv(ratings_path, sep="::", engine="python", header=None,
                                  names=["user_id","movie_id","rating","timestamp"],
                                  encoding="latin-1")

        user_feat = {}
        for _, r in users_df.iterrows():
            user_feat[int(r["user_id"])] = (
                _gender_int(r["gender"]),
                float(r["age"]),
                int(r["occupation"]),
                _hash_zip(r["zip_code"]),
            )

        movie_feat = {}
        for _, r in movies_df.iterrows():
            movie_feat[int(r["movie_id"])] = _multihot(r["genres"])

        self.user_feat  = user_feat
        self.movie_feat = movie_feat
        all_tasks       = []

        for uid, grp in ratings_df.groupby("user_id"):
            if uid not in user_feat:
                continue
            rows = list(zip(grp["movie_id"].tolist(), grp["rating"].tolist()))
            rng.shuffle(rows)

            if len(rows) < support_size + query_size:
                continue
            support_rows = rows[:support_size]
            query_rows   = rows[support_size : support_size + query_size]

            # Sanity check: support and query must never share a movie
            sup_mids = {mid for mid, _ in support_rows}
            qry_mids = {mid for mid, _ in query_rows}
            assert sup_mids.isdisjoint(qry_mids), (
                f"User {uid}: support/query overlap detected!"
            )

            all_tasks.append((int(uid), support_rows, query_rows))

        rng.shuffle(all_tasks)
        cut = int(0.8 * len(all_tasks))
        self.train_tasks = all_tasks[:cut]
        self.test_tasks  = all_tasks[cut:]
        self.tasks       = self.train_tasks   # __getitem__ indexes train by default

        print(f"[Dataset] {len(self.train_tasks)} train users | "
              f"{len(self.test_tasks)} test (cold-start) users | "
              f"support={support_size} | query={query_size} (fixed)")

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        uid, sup_rows, qry_rows = self.tasks[idx]
        uf  = self.user_feat[uid]
        sup = _make_episode(sup_rows, uf, self.movie_feat)
        qry = _make_episode(qry_rows, uf, self.movie_feat)
        return sup, qry


def collate_tasks(batch):
    supports = [b[0] for b in batch if b[0] is not None and b[1] is not None]
    queries  = [b[1] for b in batch if b[0] is not None and b[1] is not None]
    return supports, queries
