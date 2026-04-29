"""Behaviour-cloning model for the player pool.

Input per sample:
  ctx_float:  (B, N_CTX_FLOAT)
  ctx_int:    (B, 3)          # [street, actor_pos, num_players]
  board:      (B, 5)          # card indices (52 = unknown/pad)
  hole:       (B, 2)          # acting player's own cards (may be unknown)
  active:     (B, 6)          # per-position active mask (bool)
  hist_actor: (B, HIST_LEN)   # -1 padding else 0..5
  hist_street:(B, HIST_LEN)
  hist_bucket:(B, HIST_LEN)   # -1 padding else 0..NUM_ACTIONS-1
  hist_rfrac: (B, HIST_LEN)   # raise fraction (0 if not raise)

Output:
  policy_logits: (B, NUM_ACTIONS)
  raise_frac_pred: (B,)       # predicted pot-relative raise size, if bucket is a raise
  value: (B,) optional (not used in pure BC, used when reloading into PPO)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.features import HIST_LEN, MAX_PLAYERS, N_CTX_FLOAT
from ..env.actions import ALL_IN, NUM_ACTIONS, RAISE_START


class BCModel(nn.Module):
    def __init__(
        self,
        d_model: int = 192,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.card_emb = nn.Embedding(53, d_model, padding_idx=52)   # 52 = unknown
        self.street_emb = nn.Embedding(4, d_model)
        self.pos_emb = nn.Embedding(MAX_PLAYERS, d_model)
        self.n_players_emb = nn.Embedding(7, d_model)                # 2..6
        self.action_bucket_emb = nn.Embedding(NUM_ACTIONS + 1, d_model, padding_idx=NUM_ACTIONS)  # extra idx for pad
        self.ctx_proj = nn.Linear(N_CTX_FLOAT, d_model)
        # separate projection for per-history raise fraction (was conflated
        # with ctx_proj via a zero-padding hack in the original model)
        self.hist_rfrac_proj = nn.Linear(1, d_model)
        self.active_proj = nn.Linear(MAX_PLAYERS, d_model)

        # 5 board slots + 2 hole slots + 1 global ctx token + HIST_LEN history tokens
        self.n_board = 5
        self.n_hole = 2
        self.n_ctx_tok = 1
        self.max_len = self.n_board + self.n_hole + self.n_ctx_tok + HIST_LEN

        self.token_type_emb = nn.Embedding(4, d_model)  # 0=board,1=hole,2=ctx,3=hist
        self.seq_pos_emb = nn.Embedding(self.max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)

        # heads
        self.policy_head = nn.Linear(d_model, NUM_ACTIONS)
        self.rfrac_head = nn.Linear(d_model, 1)
        self.value_head = nn.Linear(d_model, 1)  # for RL reuse

    def _encode_history(self, hist_actor, hist_street, hist_bucket, hist_rfrac):
        """Return (B, HIST_LEN, d) tokens + key_padding_mask (B, HIST_LEN)."""
        pad_mask = hist_bucket < 0
        # remap -1 to pad index for embedding lookups
        actor_idx = hist_actor.clamp(min=0)
        street_idx = hist_street.clamp(min=0)
        bucket_idx = torch.where(pad_mask, torch.full_like(hist_bucket, NUM_ACTIONS), hist_bucket)
        rfrac = torch.where(pad_mask, torch.zeros_like(hist_rfrac), hist_rfrac)

        h = (self.pos_emb(actor_idx)
             + self.street_emb(street_idx)
             + self.action_bucket_emb(bucket_idx))
        h = h + self.hist_rfrac_proj(rfrac.unsqueeze(-1).float())
        return h, pad_mask

    def _encode_global(self, ctx_float, ctx_int, active):
        """Return (B, 1, d) global context token."""
        street = self.street_emb(ctx_int[:, 0])
        actor_pos = self.pos_emb(ctx_int[:, 1])
        npl = self.n_players_emb(ctx_int[:, 2].clamp(min=2, max=6))
        act = self.active_proj(active.float())
        ctx = self.ctx_proj(ctx_float)
        glob = (street + actor_pos + npl + act + ctx).unsqueeze(1)  # (B, 1, d)
        return glob

    def forward(self, batch: dict) -> dict:
        ctx_float = batch["ctx_float"]
        ctx_int = batch["ctx_int"]
        board = batch["board"]
        hole = batch["hole"]
        active = batch["active"]
        hist_actor = batch["hist_actor"]
        hist_street = batch["hist_street"]
        hist_bucket = batch["hist_bucket"]
        hist_rfrac = batch["hist_rfrac"]
        B = ctx_float.size(0)
        device = ctx_float.device

        board_tok = self.card_emb(board)                    # (B, 5, d)
        hole_tok = self.card_emb(hole)                      # (B, 2, d)
        glob_tok = self._encode_global(ctx_float, ctx_int, active)  # (B, 1, d)
        hist_tok, hist_pad = self._encode_history(hist_actor, hist_street, hist_bucket, hist_rfrac)

        # Build sequence
        tokens = torch.cat([board_tok, hole_tok, glob_tok, hist_tok], dim=1)  # (B, L, d)
        L = tokens.size(1)
        type_ids = torch.cat([
            torch.zeros(self.n_board, dtype=torch.long, device=device),
            torch.ones(self.n_hole, dtype=torch.long, device=device),
            torch.full((self.n_ctx_tok,), 2, dtype=torch.long, device=device),
            torch.full((HIST_LEN,), 3, dtype=torch.long, device=device),
        ]).unsqueeze(0).expand(B, -1)
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        tokens = tokens + self.token_type_emb(type_ids) + self.seq_pos_emb(pos_ids)

        # pad mask: board unknown(=52) also masked
        board_pad = board == 52
        hole_pad = hole == 52
        glob_pad = torch.zeros(B, self.n_ctx_tok, dtype=torch.bool, device=device)
        key_padding = torch.cat([board_pad, hole_pad, glob_pad, hist_pad], dim=1)

        enc = self.encoder(tokens, src_key_padding_mask=key_padding)
        # Pool: use the global context token (position index n_board + n_hole)
        pool_idx = self.n_board + self.n_hole
        rep = self.ln(enc[:, pool_idx])   # (B, d)

        policy_logits = self.policy_head(rep)
        rfrac_pred = torch.sigmoid(self.rfrac_head(rep).squeeze(-1)) * 3.5  # map (0, 3.5)
        value = self.value_head(rep).squeeze(-1)
        return {"policy_logits": policy_logits, "rfrac_pred": rfrac_pred, "value": value,
                "rep": rep}


def _make_soft_target(
    labels: torch.Tensor,
    num_classes: int = NUM_ACTIONS,
    label_smoothing: float = 0.05,
    raise_neighbor_weight: float = 0.15,
) -> torch.Tensor:
    """Build soft-label target for bucketed action classification.

    Rationale
    ---------
    The 10 action buckets are not categorically independent: buckets 2..8 are
    ordinal raise sizes (0.33x..3x pot). Plain CE treats R0.33 and R3.00 as
    equally far from R0.75, which encourages the model to collapse all raise
    probability onto a single "safe" bucket (we observed this at 0.5x pot in
    self-play). We fix this by assigning part of each raise-label's mass to
    its ±1 neighbor raise buckets.

    For non-raise labels (fold, check/call, all-in) we only apply uniform
    label smoothing.

    Returns
    -------
    (B, num_classes) soft-target distribution (rows sum to 1).
    """
    B = labels.size(0)
    N = num_classes
    device = labels.device
    dtype = torch.float32

    tgt = torch.zeros(B, N, device=device, dtype=dtype)
    tgt.scatter_(1, labels.long().unsqueeze(1), 1.0)

    is_raise = (labels >= RAISE_START) & (labels < ALL_IN)
    if is_raise.any():
        b_idx = torch.nonzero(is_raise, as_tuple=True)[0]
        ctr = labels[is_raise].long()
        tgt[b_idx, ctr] = 1.0 - 2 * raise_neighbor_weight
        for delta in (-1, 1):
            nbr = ctr + delta
            valid = (nbr >= RAISE_START) & (nbr < ALL_IN)
            if valid.any():
                tgt[b_idx[valid], nbr[valid]] = raise_neighbor_weight
        # restore mass for edge buckets (no left/right neighbor)
        row_sums = tgt.sum(dim=-1)
        deficit = 1.0 - row_sums
        all_rows = torch.arange(B, device=device)
        tgt[all_rows, labels.long()] += deficit

    if label_smoothing > 0:
        tgt = tgt * (1 - label_smoothing) + label_smoothing / N
    return tgt


def bc_loss(
    output: dict,
    batch: dict,
    rfrac_weight: float = 0.1,
    label_smoothing: float = 0.05,
    raise_neighbor_weight: float = 0.15,
) -> dict:
    logits = output["policy_logits"]
    labels = batch["label_bucket"]
    labels_long = labels.long()

    with torch.no_grad():
        tgt = _make_soft_target(
            labels_long, num_classes=logits.size(-1),
            label_smoothing=label_smoothing,
            raise_neighbor_weight=raise_neighbor_weight,
        )
    log_probs = F.log_softmax(logits.float(), dim=-1)
    policy_loss = -(tgt * log_probs).sum(dim=-1).mean()

    is_raise = (labels_long >= RAISE_START) & (labels_long <= ALL_IN)
    if is_raise.any():
        rfrac_loss = F.smooth_l1_loss(
            output["rfrac_pred"][is_raise],
            batch["label_rfrac"][is_raise].float().clamp(0, 3.5),
        )
    else:
        rfrac_loss = torch.tensor(0.0, device=logits.device)

    total = policy_loss + rfrac_weight * rfrac_loss
    with torch.no_grad():
        acc = (logits.argmax(dim=-1) == labels_long).float().mean()
    return {
        "loss": total,
        "policy_loss": policy_loss.detach(),
        "rfrac_loss": rfrac_loss.detach(),
        "acc": acc,
    }
