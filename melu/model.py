"""
Neural Network Architectures for MeLU Recommender System.
Includes UserEmbedder, MovieEmbedder, RatingMLP, and MovieRecommender.
"""

import torch
import torch.nn as nn
from config import (
    EMBED_DIM, OCC_VOCAB, ZIP_VOCAB, AGE_MAX,
    NUM_GENRES, FUSION_DIM
)

class UserEmbedder(nn.Module):
    """
    Maps user attributes → user vector (θ₁ in the paper).
    Gender + age + occupation + zip_code, matching Figure 2.
    """
    def __init__(self, embed_dim=EMBED_DIM, age_max=AGE_MAX):
        super().__init__()
        self.age_max    = age_max
        self.gender_emb = nn.Embedding(2,         embed_dim)
        self.occ_emb    = nn.Embedding(OCC_VOCAB,  embed_dim)
        self.zip_emb    = nn.Embedding(ZIP_VOCAB,  embed_dim)
        for emb in (self.gender_emb, self.occ_emb, self.zip_emb):
            nn.init.normal_(emb.weight, 0.0, 0.01)

    def forward(self, gender, age, occupation, zipcode):
        g = self.gender_emb(gender)
        a = (age / self.age_max).unsqueeze(1)
        o = self.occ_emb(occupation)
        z = self.zip_emb(zipcode)
        return torch.cat([g, a, o, z], dim=1)   # (B, 97)


class MovieEmbedder(nn.Module):
    """
    Maps movie genre multi-hot → item vector (part of θ₁ in the paper).
    NOTE: The paper also uses year, director, and actor from IMDb. This
    implementation keeps only genre for simplicity on the raw MovieLens-1M
    dataset (no IMDb augmentation). If you add those features, replace this
    Linear projection with a concatenation of several embeddings, matching
    Figure 2 more closely.
    """
    def __init__(self, num_genres=NUM_GENRES, embed_dim=EMBED_DIM):
        super().__init__()
        self.proj = nn.Linear(num_genres, embed_dim, bias=True)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, genre_multihot):
        return self.proj(genre_multihot)         # (B, 32)


class RatingMLP(nn.Module):
    """
    Decision-making / output network (θ₂ in the paper, Eq. 3).
    This is the ONLY part adapted in the inner loop.
    """
    def __init__(self, in_dim=FUSION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256,    128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128,     64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64,      32), nn.ReLU(),
            # ── CHANGE: Output layer is now linear (no sigmoid) ──────────────
            # Original code applied sigmoid(raw) * 4 + 1 to force output into
            # [1, 5]. The paper (§3.1, Eq. 3) says "a linear function might be
            # appropriate" for rating estimation. Sigmoid saturates near 1 and 5,
            # producing near-zero gradients for extreme-rated movies and making
            # it harder for the model to learn from 1-star and 5-star samples.
            # We use a raw linear output here and clamp to [1, 5] only at
            # inference / evaluation time.
            nn.Linear(32, 1),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x).squeeze(1)   # (B,)  unconstrained during training


class MovieRecommender(nn.Module):
    def __init__(self):
        super().__init__()
        self.user_embedder  = UserEmbedder()
        self.movie_embedder = MovieEmbedder()
        self.mlp            = RatingMLP()

    def forward(self, gender, age, occupation, zipcode, genre_multihot):
        user_vec  = self.user_embedder(gender, age, occupation, zipcode)
        movie_vec = self.movie_embedder(genre_multihot)
        fused     = torch.cat([user_vec, movie_vec], dim=1)
        return self.mlp(fused)
