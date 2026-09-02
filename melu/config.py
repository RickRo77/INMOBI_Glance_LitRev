"""
Configuration settings, hyperparameters, and global constants for MeLU recommender system.
"""

import torch

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ══════════════════════════════════════════════════════════════════════════════
#  1. CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western",
]
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}
NUM_GENRES   = 18

EMBED_DIM    = 32
USER_DIM     = EMBED_DIM + 1 + EMBED_DIM + EMBED_DIM   # 97
MOVIE_DIM    = EMBED_DIM                                # 32
FUSION_DIM   = USER_DIM + MOVIE_DIM                     # 129

ZIP_VOCAB    = 3500
OCC_VOCAB    = 21
AGE_MAX      = 56.0

SUPPORT_SIZE         = 20
QUERY_SIZE_AT_BUILD  = 10   # Paper fixes query size = 10 at dataset construction (§4.1)

INNER_LR     = 5e-6   # α  — paper §4.1 for MovieLens
OUTER_LR     = 5e-5   # β  — paper §4.1 for MovieLens
INNER_STEPS  = 5      # paper tests 1–5; 5 is the reported best
META_BATCH   = 32     # paper §4.1
EPOCHS       = 30     # paper §4.1

LOG_INTERVAL = 50

DATA_ROOT   = "/kaggle/input/datasets/rrickyroger/movielens-1m"
OUTPUT_ROOT = "/kaggle/working"

# Demographics Lookups
OCC_LABELS = {
     0: "other / not specified",  1: "academic / educator",   2: "artist",
     3: "clerical / admin",       4: "college / grad student", 5: "customer service",
     6: "doctor / health care",   7: "executive / managerial", 8: "farmer",
     9: "homemaker",             10: "K-12 student",          11: "lawyer",
    12: "programmer",            13: "retired",               14: "sales / marketing",
    15: "scientist",             16: "self-employed",         17: "technician / engineer",
    18: "tradesman / craftsman", 19: "unemployed",            20: "writer",
}

AGE_LABELS = {
    1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44",
    45: "45-49", 50: "50-55", 56: "56+"
}
