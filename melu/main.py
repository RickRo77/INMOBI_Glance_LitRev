"""
Main pipeline execution script for MeLU recommender system.
Orchestrates meta-training, saving, evidence candidate selection, and cold-start evaluation.
"""

import pandas as pd
from config import DATA_ROOT, OUTPUT_ROOT, EPOCHS, INNER_STEPS, INNER_LR, OUTER_LR
from train import meta_train
from utils import save_model
from eval import compute_evidence_candidates, run_melu_evaluation

def main():
    # 1. Meta-Training
    print("🚀 Starting Meta-Training...")
    model, dataset = meta_train()

    # 2. Save Final Global Model
    save_model(model, f"{OUTPUT_ROOT}/recommender_maml_final.pt",
               meta={"epochs": EPOCHS, "inner_steps": INNER_STEPS,
                     "inner_lr": INNER_LR, "outer_lr": OUTER_LR})

    # 3. Load Datasets metadata for pretty-printing
    movies_df    = pd.read_csv(f"{DATA_ROOT}/movies.dat", sep="::", engine="python",
                               header=None, names=["movie_id","title","genres"],
                               encoding="latin-1")
    movie_title  = dict(zip(movies_df["movie_id"], movies_df["title"]))
    movie_genres = dict(zip(movies_df["movie_id"], movies_df["genres"]))
    users_raw    = pd.read_csv(f"{DATA_ROOT}/users.dat", sep="::", engine="python",
                               header=None,
                               names=["user_id","gender","age","occupation","zip_code"],
                               encoding="latin-1")

    # 4. Evidence Candidate Selection (MeLU §3.3)
    print("\n── MeLU §3.3: Evidence candidate selection ──")
    evidence_df = compute_evidence_candidates(
        global_model = model,
        dataset      = dataset,
        movies_df    = movies_df,
        top_k        = 10,
    )

    # 5. Cold-Start Evaluation (MeLU Meta-Test)
    print("\n── MeLU-style per-user personalisation & evaluation (cold-start users) ──")
    run_melu_evaluation(
        global_model  = model,
        dataset        = dataset,
        movie_title    = movie_title,
        movie_genres   = movie_genres,
        users_raw      = users_raw,
        n_users        = 5,
        top_k          = 20,
        finetune_steps = 5,
        finetune_lr    = 1e-2,
    )

if __name__ == "__main__":
    main()
