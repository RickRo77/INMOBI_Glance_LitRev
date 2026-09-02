"""
MAML core routines for inner-loop gradient adaptation and functional execution.
"""

import torch
import torch.nn.functional as F
from torch.func import functional_call

def batch_to_device(d, device):
    return {k: v.to(device) for k, v in d.items()}

def forward_with_params(model, mlp_params, batch):
    """Forward pass using functional (per-step) MLP params + global embedders."""
    user_vec  = model.user_embedder(
        batch["gender"], batch["age"], batch["occupation"], batch["zipcode"])
    movie_vec = model.movie_embedder(batch["genre_multihot"])
    fused     = torch.cat([user_vec, movie_vec], dim=1)
    pred      = functional_call(model.mlp, mlp_params, (fused,))
    return pred.squeeze(-1)

def inner_loop(model, support, inner_lr, inner_steps):
    """
    Adapt θ₂ (MLP only) on the support set via gradient descent.
    Returns θ′ — the adapted MLP parameters for this user.
    Embedder weights (θ₁) are NOT touched here; they only receive
    gradient updates through the outer (meta) loss.
    """
    local_params = {n: p.clone() for n, p in model.mlp.named_parameters()}
    for _ in range(inner_steps):
        pred  = forward_with_params(model, local_params, support)
        loss  = F.l1_loss(pred, support["rating"])
        grads = torch.autograd.grad(
            loss, list(local_params.values()),
            create_graph=True, allow_unused=True)
        local_params = {
            name: param - inner_lr * (grad if grad is not None else torch.zeros_like(param))
            for (name, param), grad in zip(local_params.items(), grads)
        }
    return local_params
