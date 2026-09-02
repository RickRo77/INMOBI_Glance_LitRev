"""
MeLU evaluation module: candidate evidence selection and per-user cold-start evaluation.
"""

import copy
import random
import torch
import torch.nn.functional as F
import pandas as pd

from config import DEVICE, DATA_ROOT
from dataset import _make_episode
from maml import batch_to_device
from utils import (
    print_user_card, print_support_set,
    print_predictions, print_param_report
)

def compute_evidence_candidates(global_model, dataset, movies_df,
                                 top_k=10, device=DEVICE):
    """
    Implements MeLU §3.3: rank all movies by
        popularity × avg Frobenius-norm of ∇_θ₂ loss over training users.

    Returns a DataFrame of the top_k candidate movies to show new users.
    """
    global_model.eval()

    # Count how many training users rated each movie (popularity proxy)
    movie_count = {}
    for uid, sup_rows, qry_rows in dataset.train_tasks:
        for mid, _ in sup_rows + qry_rows:
            movie_count[mid] = movie_count.get(mid, 0) + 1

    # Accumulate gradient norms per movie across all training users
    movie_grad_norm = {}
    movie_user_count = {}

    n_users = len(dataset.train_tasks)
    print(f"[EvidenceCandidates] Computing gradient norms over {n_users} users...")

    for step, (uid, sup_rows, _) in enumerate(dataset.train_tasks):
        uf = dataset.user_feat[uid]
        ep = _make_episode(sup_rows, uf, dataset.movie_feat)
        if ep is None:
            continue
        ep = batch_to_device(ep, device)

        # We need per-item gradient norms: compute loss for each item separately
        for item_idx in range(ep["rating"].shape[0]):
            single = {k: v[item_idx].unsqueeze(0) for k, v in ep.items()}
            mid = sup_rows[item_idx][0]

            # Zero out any accumulated grads
            global_model.zero_grad()

            pred = global_model(
                single["gender"], single["age"],
                single["occupation"], single["zipcode"],
                single["genre_multihot"])
            loss = F.l1_loss(pred, single["rating"])
            loss.backward()

            # Frobenius norm of gradients across θ₂ (MLP) parameters
            frob = sum(
                p.grad.norm(p="fro").item() ** 2
                for p in global_model.mlp.parameters()
                if p.grad is not None
            ) ** 0.5

            movie_grad_norm[mid]   = movie_grad_norm.get(mid, 0.0) + frob
            movie_user_count[mid]  = movie_user_count.get(mid, 0) + 1

        if (step + 1) % 500 == 0:
            print(f"  processed {step+1}/{n_users} users...")

    # Average gradient norm over users that rated each movie
    scores = {}
    for mid in movie_grad_norm:
        avg_norm   = movie_grad_norm[mid] / movie_user_count[mid]
        popularity = movie_count.get(mid, 0)
        scores[mid] = popularity * avg_norm

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # Build a readable DataFrame
    title_map  = dict(zip(movies_df["movie_id"], movies_df["title"]))
    genres_map = dict(zip(movies_df["movie_id"], movies_df["genres"]))
    rows = []
    for rank, (mid, score) in enumerate(ranked, 1):
        rows.append({
            "rank":           rank,
            "movie_id":       mid,
            "title":          title_map.get(mid, f"movie_{mid}"),
            "genres":         genres_map.get(mid, ""),
            "popularity":     movie_count.get(mid, 0),
            "avg_grad_norm":  movie_grad_norm[mid] / movie_user_count[mid],
            "evidence_score": score,
        })
    df = pd.DataFrame(rows)
    print(f"\n[EvidenceCandidates] Top-{top_k} evidence candidates:")
    print(df[["rank", "title", "genres", "popularity", "evidence_score"]].to_string(index=False))
    return df


def personalize_for_user(global_model, support_batch, finetune_steps=5, lr=1e-2):
    """
    Simulates MeLU's meta-test procedure for ONE new user:
      - deep-copies the GLOBAL model (global_model is never mutated)
      - fine-tunes ONLY the MLP (θ₂) on the support set
      - returns the personalized model + loss trajectory
    """
    local_model = copy.deepcopy(global_model).to(DEVICE)
    local_model.train()

    optimizer = torch.optim.SGD(local_model.mlp.parameters(), lr=lr)

    loss_trace = []
    for _ in range(finetune_steps):
        optimizer.zero_grad()
        pred = local_model(
            support_batch["gender"], support_batch["age"],
            support_batch["occupation"], support_batch["zipcode"],
            support_batch["genre_multihot"],
        )
        loss = F.l1_loss(pred, support_batch["rating"])
        loss.backward()
        optimizer.step()
        loss_trace.append(loss.item())

    local_model.eval()
    return local_model, loss_trace


def param_delta_report(global_model, local_model):
    """Per-layer L2 distance between personalized MLP and global MLP."""
    report = []
    g_params = dict(global_model.mlp.named_parameters())
    l_params = dict(local_model.mlp.named_parameters())
    for name in g_params:
        delta = (l_params[name].detach() - g_params[name].detach()).norm().item()
        gnorm = g_params[name].detach().norm().item()
        report.append((name, delta, gnorm))
    return report


def run_melu_evaluation(global_model, dataset, movie_title, movie_genres, users_raw,
                         n_users=5, top_k=20, finetune_steps=5, finetune_lr=1e-2, seed=99):
    """
    MeLU meta-test evaluation on HELD-OUT new users (test_tasks only).

    For n_users randomly chosen from dataset.test_tasks:
      global θ → clone → fine-tune on support (disjoint from query)
               → evaluate / recommend on query pool → report

    global_model is NEVER modified — only deep copies are fine-tuned.
    """
    global_model.eval()

    eval_pool = dataset.test_tasks
    rng       = random.Random(seed)
    indices   = rng.sample(range(len(eval_pool)), min(n_users, len(eval_pool)))
    all_maes, user_reports = [], []

    for pick_num, idx in enumerate(indices, 1):
        uid, sup_rows, qry_rows = eval_pool[idx]
        user_feat = dataset.user_feat[uid]

        sup_mids = {mid for mid, _ in sup_rows}
        qry_mids = {mid for mid, _ in qry_rows}
        assert sup_mids.isdisjoint(qry_mids), f"User {uid}: support/query overlap!"

        def make_ep(rows):
            g, a, o, z = user_feat
            mids, genre_vecs, ratings = [], [], []
            for mid, rat in rows:
                mf = dataset.movie_feat.get(mid)
                if mf is None:
                    continue
                mids.append(mid)
                genre_vecs.append(mf)
                ratings.append(rat)
            n = len(ratings)
            if n == 0:
                return None, []
            batch = {
                "gender":         torch.full((n,), g, dtype=torch.long),
                "age":            torch.full((n,), a, dtype=torch.float),
                "occupation":     torch.full((n,), o, dtype=torch.long),
                "zipcode":        torch.full((n,), z, dtype=torch.long),
                "genre_multihot": torch.stack(genre_vecs),
                "rating":         torch.tensor(ratings, dtype=torch.float),
            }
            return {k: v.to(DEVICE) for k, v in batch.items()}, mids

        sup_batch, _            = make_ep(sup_rows)
        qry_batch, qry_mids_list = make_ep(qry_rows)
        if sup_batch is None or qry_batch is None:
            continue

        actuals = qry_batch["rating"]

        # Clone global θ, fine-tune on support (MeLU meta-test)
        local_model, loss_trace = personalize_for_user(
            global_model, sup_batch, finetune_steps=finetune_steps, lr=finetune_lr)

        # Evaluate on disjoint query set
        with torch.no_grad():
            preds = local_model(
                qry_batch["gender"], qry_batch["age"], qry_batch["occupation"],
                qry_batch["zipcode"], qry_batch["genre_multihot"])

        mae  = F.l1_loss(preds, actuals).item()
        rmse = F.mse_loss(preds, actuals).item() ** 0.5
        all_maes.append(mae)

        order     = preds.argsort(descending=True).tolist()
        pred_rows = [(qry_mids_list[i], preds[i].item(), actuals[i].item()) for i in order]
        param_report = param_delta_report(global_model, local_model)

        print(f"\n\n{'#'*72}\n  MELU META-TEST  │  USER {pick_num} of {n_users}  (user_id={uid})  [COLD-START]\n{'#'*72}")
        print_user_card(uid, user_feat, users_raw)
        print_support_set(sup_rows, movie_title, movie_genres)
        print_predictions(pred_rows, movie_title, movie_genres, top_k=top_k)
        print_param_report(param_report, finetune_steps, loss_trace)
        print(f"\n  Query-set MAE  (personalized model, paper metric): {mae:.4f}")
        print(f"  Query-set RMSE (secondary):                         {rmse:.4f}")
        print(f"  Support/query disjoint: ✅ "
              f"({len(sup_mids)} support, {len(qry_mids)} query, 0 overlap)")
        print(f"  User is from HELD-OUT test split (true cold-start): ✅")

        user_reports.append({
            "user_id": uid, "mae": mae, "rmse": rmse,
            "loss_trace": loss_trace,
            "param_report": param_report,
            "personalized_model": local_model,
        })

    avg_mae = sum(all_maes) / len(all_maes) if all_maes else float("nan")
    print(f"\n{'═'*72}")
    print(f"  OVERALL │ {len(all_maes)} cold-start users │ avg MAE = {avg_mae:.4f}")
    print(f"{'═'*72}\n")
    return user_reports, avg_mae
