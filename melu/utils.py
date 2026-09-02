"""
Utility functions for saving/loading model weights and pretty-printing evaluation cards.
"""

import torch
from config import DEVICE, OCC_LABELS, AGE_LABELS
from model import MovieRecommender

def save_model(model, path, meta=None):
    """Save GLOBAL model weights (embedders + MLP) for reuse later."""
    payload = {
        "model_state_dict": model.state_dict(),
        "model_class":      "MovieRecommender",
        "meta":             meta or {},
    }
    torch.save(payload, path)
    print(f"✅ Model saved → {path}")


def load_model(path, device=DEVICE):
    """Load a model saved with save_model()."""
    payload = torch.load(path, map_location=device)
    model = MovieRecommender().to(device)
    model.load_state_dict(payload["model_state_dict"])
    print(f"✅ Model loaded from {path}  (meta: {payload.get('meta', {})})")
    return model


def _bar(rating, width=20):
    filled = int(round(max(0.0, min(rating, 5.0)) / 5.0 * width))
    return "█" * filled + "░" * (width - filled)


def print_user_card(uid, user_feat, users_raw):
    g, a, o, z = user_feat
    gender_str = "Female" if g == 1 else "Male"
    age_str    = AGE_LABELS.get(int(a), str(int(a)))
    occ_str    = OCC_LABELS.get(o, f"code {o}")
    row = users_raw[users_raw["user_id"] == uid]
    zip_str = row["zip_code"].values[0] if len(row) else "N/A"
    print(f"\n{'═'*72}\n  USER {uid}\n{'═'*72}")
    print(f"  Gender     : {gender_str}")
    print(f"  Age group  : {age_str}")
    print(f"  Occupation : {occ_str}")
    print(f"  Zip code   : {zip_str}")


def print_support_set(sup_rows, movie_title, movie_genres):
    print(f"\n  ┌─ SUPPORT SET ({len(sup_rows)} movies used to fine-tune this user's model) ──")
    print(f"  │  {'#':>2}  {'Rating':>6}  {'Title':<42}  Genres")
    print(f"  │  {'─'*2}  {'─'*6}  {'─'*42}  {'─'*20}")
    for i, (mid, rat) in enumerate(sup_rows, 1):
        title  = movie_title.get(mid, f"movie_{mid}")[:41]
        genres = movie_genres.get(mid, "")
        print(f"  │  {i:>2}  {rat:>5.1f}★  {title:<42}  {genres}")
    print(f"  └{'─'*70}")


def print_predictions(pred_rows, movie_title, movie_genres, top_k=20):
    top = pred_rows[:top_k]
    print(f"\n  ┌─ TOP {top_k} RECOMMENDATIONS (personalized model, query pool) ──────────")
    print(f"  │  {'#':>2}  {'Pred':>5}  {'Bar':<20}  {'Actual':>6}  {'Title':<38}  Genres")
    print(f"  │  {'─'*2}  {'─'*5}  {'─'*20}  {'─'*6}  {'─'*38}  {'─'*18}")
    for rank, (mid, pred, actual) in enumerate(top, 1):
        title  = movie_title.get(mid, f"movie_{mid}")[:37]
        genres = movie_genres.get(mid, "")[:30]
        bar    = _bar(pred)
        print(f"  │  {rank:>2}  {pred:>4.2f}  {bar}  {actual:>5.1f}★  {title:<38}  {genres}")
    print(f"  └{'─'*70}")


def print_param_report(report, finetune_steps, loss_trace):
    print(f"\n  ┌─ PERSONALIZED PARAMETERS (MLP only, {finetune_steps} fine-tune steps) ──")
    print(f"  │  Loss trajectory: {['%.4f' % l for l in loss_trace]}")
    print(f"  │")
    print(f"  │  {'Layer':<22} {'‖Δθ‖ (moved from global)':>26}  {'‖θ_global‖':>12}")
    print(f"  │  {'─'*22} {'─'*26}  {'─'*12}")
    for name, delta, gnorm in report:
        pct = (delta / gnorm * 100) if gnorm > 0 else 0.0
        print(f"  │  {name:<22} {delta:>20.4f} ({pct:>4.1f}%)  {gnorm:>12.4f}")
    print(f"  └{'─'*70}")
