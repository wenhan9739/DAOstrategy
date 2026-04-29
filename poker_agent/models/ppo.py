"""PPO helpers that reuse BCModel as the actor-critic backbone.

The hero policy and value share the BC Transformer trunk; only the final
heads are effectively trained from scratch for RL (though they are warm-started
from BC).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..env.actions import NUM_ACTIONS
from .bc import BCModel


def load_bc_checkpoint(path: str, map_location: str = "cpu") -> BCModel:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg = ckpt.get("cfg", {})
    model = BCModel(
        d_model=cfg.get("d_model", 192),
        n_layers=cfg.get("n_layers", 4),
        n_heads=cfg.get("n_heads", 4),
        dropout=cfg.get("dropout", 0.1),
    )
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        print(f"load_bc_checkpoint: missing={missing} unexpected={unexpected}")
    return model


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return logits with illegal actions set to -inf (so probability = 0)."""
    neg = torch.finfo(logits.dtype).min
    return logits.masked_fill(~mask, neg)


def sample_action(
    logits: torch.Tensor,
    mask: torch.Tensor,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (action, log_prob, entropy) for a masked categorical."""
    ml = masked_logits(logits, mask)
    dist = torch.distributions.Categorical(logits=ml)
    if deterministic:
        action = ml.argmax(dim=-1)
    else:
        action = dist.sample()
    log_prob = dist.log_prob(action)
    entropy = dist.entropy()
    return action, log_prob, entropy


class ActorCritic(nn.Module):
    """Thin wrapper around BCModel that exposes the RL-friendly API."""

    def __init__(self, bc: BCModel) -> None:
        super().__init__()
        self.bc = bc

    def forward(self, batch: dict) -> dict:
        return self.bc(batch)

    def act(self, batch: dict, mask: torch.Tensor, deterministic: bool = False):
        out = self.bc(batch)
        action, log_prob, entropy = sample_action(out["policy_logits"], mask, deterministic)
        return action, log_prob, entropy, out["value"], out["policy_logits"]

    def evaluate(self, batch: dict, mask: torch.Tensor, action: torch.Tensor):
        out = self.bc(batch)
        ml = masked_logits(out["policy_logits"], mask)
        dist = torch.distributions.Categorical(logits=ml)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, out["value"], out["policy_logits"]
