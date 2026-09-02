"""
Meta-training loop for MeLU recommender system.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import (
    DEVICE, DATA_ROOT, OUTPUT_ROOT, META_BATCH,
    INNER_LR, OUTER_LR, INNER_STEPS, EPOCHS, LOG_INTERVAL
)
from dataset import MovieLensMAML, collate_tasks
from model import MovieRecommender
from maml import batch_to_device, inner_loop, forward_with_params
from utils import save_model

def meta_train():
    dataset = MovieLensMAML(
        users_path   = f"{DATA_ROOT}/users.dat",
        movies_path  = f"{DATA_ROOT}/movies.dat",
        ratings_path = f"{DATA_ROOT}/ratings.dat",
    )

    train_dataset = dataset   # dataset.__getitem__ already indexes train_tasks
    loader = DataLoader(train_dataset, batch_size=META_BATCH, shuffle=True,
                        collate_fn=collate_tasks, num_workers=2)

    model    = MovieRecommender().to(DEVICE)
    meta_opt = torch.optim.Adam(model.parameters(), lr=OUTER_LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n")
    print(f"{'Epoch':>5} {'Step':>6} {'Meta-MAE':>10} {'MAE':>8}  Time/step")
    print("─" * 55)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, batch_count = 0.0, 0
        t0 = time.time()

        for supports, queries in loader:
            if not supports:
                continue
            meta_opt.zero_grad()

            total_loss, valid = torch.tensor(0.0, device=DEVICE), 0

            for sup_raw, qry_raw in zip(supports, queries):
                sup = batch_to_device(sup_raw, DEVICE)
                qry = batch_to_device(qry_raw, DEVICE)

                if sup["rating"].numel() == 0 or qry["rating"].numel() == 0:
                    continue

                # Inner loop: adapt θ₂ on support set
                theta_prime = inner_loop(model, sup, INNER_LR, INNER_STEPS)

                # Query loss using adapted parameters θ'
                pred_q     = forward_with_params(model, theta_prime, qry)
                loss_q     = F.l1_loss(pred_q, qry["rating"])
                total_loss = total_loss + loss_q
                valid     += 1

            if valid == 0:
                continue

            avg_loss = total_loss / valid
            avg_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            meta_opt.step()

            running_loss += avg_loss.item()
            batch_count  += 1

            if batch_count % LOG_INTERVAL == 0:
                ml  = running_loss / batch_count
                ms  = (time.time() - t0) / batch_count * 1000
                print(f"{epoch:>5} {batch_count:>6} {ml:>10.4f} {ml:>8.4f}  {ms:>7.1f}ms")

        if batch_count:
            ml = running_loss / batch_count
            print(f"\n  ▶ Epoch {epoch} │ meta-MAE={ml:.4f}\n")
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "opt_state": meta_opt.state_dict(), "meta_loss": ml,
            }, f"{OUTPUT_ROOT}/ckpt_epoch{epoch}.pt")

    print("✅ Training done.")
    return model, dataset


if __name__ == "__main__":
    model, dataset = meta_train()
    save_model(model, f"{OUTPUT_ROOT}/recommender_maml_final.pt",
               meta={"epochs": EPOCHS, "inner_steps": INNER_STEPS,
                     "inner_lr": INNER_LR, "outer_lr": OUTER_LR})
