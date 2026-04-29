"""Featurize DecisionSample into tensors suitable for model training.

The encoded sample has fixed shapes so shards can be stored as numpy arrays.
"""
from __future__ import annotations

import numpy as np

from ..env.actions import NUM_ACTIONS
from ..env.cards import UNKNOWN_CARD
from .replay import DecisionSample

HIST_LEN = 48            # max history tokens kept (pre/flop/turn/river all combined)
MAX_PLAYERS = 6
N_CTX_FLOAT = 7          # see _fill_ctx
N_CTX_INT = 3            # street, actor_pos, num_players


def encode_sample(s: DecisionSample) -> dict:
    """Return a dict of fixed-shape numpy arrays for one sample."""
    # context floats (log1p scaled)
    ctx_float = np.zeros(N_CTX_FLOAT, dtype=np.float32)
    ctx_float[0] = np.log1p(s.pot_bb)
    ctx_float[1] = np.log1p(s.to_call_bb)
    ctx_float[2] = np.log1p(s.stack_bb)
    ctx_float[3] = np.log1p(s.effective_stack_bb)
    ctx_float[4] = s.to_call_bb / max(s.pot_bb, 1e-3)  # call price
    ctx_float[5] = s.stack_bb / max(s.pot_bb, 1e-3)    # SPR
    ctx_float[6] = s.num_active / MAX_PLAYERS

    ctx_int = np.zeros(N_CTX_INT, dtype=np.int8)
    ctx_int[0] = s.street
    ctx_int[1] = s.actor_pos
    ctx_int[2] = s.num_players

    board = np.full(5, UNKNOWN_CARD, dtype=np.uint8)
    for i, c in enumerate(s.board[:5]):
        board[i] = c
    hole = np.full(2, UNKNOWN_CARD, dtype=np.uint8)
    for i, c in enumerate(s.hole[:2]):
        hole[i] = c
    active = np.asarray(s.active_mask[:MAX_PLAYERS], dtype=np.bool_)

    # Only keep player-action tokens in history (skip dealer events).
    hist_actor = np.full(HIST_LEN, -1, dtype=np.int8)  # -1 = padding
    hist_street = np.zeros(HIST_LEN, dtype=np.int8)
    hist_bucket = np.full(HIST_LEN, -1, dtype=np.int8)
    hist_rfrac = np.zeros(HIST_LEN, dtype=np.float16)

    tokens = [t for t in s.action_history if t.actor >= 0 and t.bucket >= 0]
    if len(tokens) > HIST_LEN:
        tokens = tokens[-HIST_LEN:]
    for i, t in enumerate(tokens):
        hist_actor[i] = t.actor_pos  # canonical position 0..5
        hist_street[i] = t.street
        hist_bucket[i] = t.bucket
        hist_rfrac[i] = float(t.raise_frac)

    return {
        "ctx_float": ctx_float,
        "ctx_int": ctx_int,
        "board": board,
        "hole": hole,
        "active": active,
        "hist_actor": hist_actor,
        "hist_street": hist_street,
        "hist_bucket": hist_bucket,
        "hist_rfrac": hist_rfrac,
        # labels
        "label_bucket": np.int8(s.bucket),
        "label_rfrac": np.float16(s.raise_frac),
        # meta
        "num_players": np.int8(s.num_players),
        "actor_idx": np.int8(s.actor_idx),
    }


def stack_batch(samples: list[dict]) -> dict:
    """Stack a list of encoded dicts into batched numpy arrays."""
    if not samples:
        return {}
    keys = samples[0].keys()
    out = {}
    for k in keys:
        out[k] = np.stack([s[k] for s in samples])
    return out
