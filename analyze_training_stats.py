"""Compute decision-level action distributions from the BC training shards.

Matches the per-decision statistics produced by ``analyze_bc_selfplay.py``
so the two outputs can be compared apples-to-apples. We can NOT compute
per-hand metrics (VPIP / PFR / 3bet) from the shards alone because samples
don't carry a hand id, so only decision-level conditionals are reported.

Usage:
  python analyze_training_stats.py \
      --shard-glob "processed/shard_*.npz" \
      --max-shards 20 \
      --out stats_training.json
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np

from poker_agent.data.features import HIST_LEN, MAX_PLAYERS
from poker_agent.env.actions import (
    ACTION_NAMES,
    ALL_IN,
    CHECK_CALL,
    FOLD,
    NUM_ACTIONS,
    RAISE_POT_FRACTIONS,
    RAISE_START,
)

POS_NAMES = ["SB", "BB", "UTG", "HJ", "CO", "BTN"]
STREET_NAMES = ["pre", "flop", "turn", "river"]


def _accumulate_shard(
    z: dict,
    action_total: np.ndarray,
    action_per_street: np.ndarray,
    action_per_pos: np.ndarray,
    pf_open: np.ndarray,
    pf_vs_raise: np.ndarray,
    pf_postflop_facing_bet: np.ndarray,
    pf_postflop_no_bet: np.ndarray,
    raise_bucket_hist: np.ndarray,
    raise_sizing_by_street: np.ndarray,
    rfrac_sum_by_bucket: np.ndarray,
    rfrac_count_by_bucket: np.ndarray,
    street_distribution: np.ndarray,
    n_active_distribution: np.ndarray,
):
    label_bucket = z["label_bucket"].astype(np.int64)
    label_rfrac = z["label_rfrac"].astype(np.float32)
    ctx_int = z["ctx_int"].astype(np.int64)    # (N, 3) [street, actor_pos, num_players]
    ctx_float = z["ctx_float"].astype(np.float32)
    hist_actor = z["hist_actor"].astype(np.int64)    # (N, HIST_LEN)
    hist_bucket = z["hist_bucket"].astype(np.int64)
    hist_street = z["hist_street"].astype(np.int64)
    active = z["active"]                             # (N, 6) bool

    street = ctx_int[:, 0]
    pos = ctx_int[:, 1]
    # num_active actually derived from 'active' mask (more reliable than num_players)
    num_active = active.sum(axis=1)
    # to_call = ctx_float[:, 1] is log1p(to_call_bb) -> recover
    to_call_bb = np.expm1(ctx_float[:, 1])

    # street-wise and overall action freq
    for a in range(NUM_ACTIONS):
        action_total[a] += int(((label_bucket == a)).sum())
        for st in range(4):
            action_per_street[st, a] += int(((label_bucket == a) & (street == st)).sum())
        for p in range(MAX_PLAYERS):
            action_per_pos[p, a] += int(((label_bucket == a) & (pos == p)).sum())

    # preflop: count # of raise tokens in history on street 0 before this sample
    # a raise token is bucket in RAISE_START..ALL_IN
    pre_mask = (hist_street == 0) & (hist_bucket >= RAISE_START) & (hist_bucket <= ALL_IN)
    n_preflop_raises = pre_mask.sum(axis=1)  # (N,)

    # "is first voluntary preflop action for this seat" = this seat never acted preflop in history
    # check hist_actor == pos (and hist_street==0 and hist_bucket!=-1)
    seat_acted_pre = ((hist_actor == pos[:, None]) & (hist_street == 0) &
                      (hist_bucket >= 0)).any(axis=1)
    is_first_voluntary_preflop = (~seat_acted_pre) & (street == 0)

    # preflop open (first voluntary + 0 raises so far)
    open_mask = is_first_voluntary_preflop & (n_preflop_raises == 0)
    for p in range(MAX_PLAYERS):
        sel = open_mask & (pos == p)
        if sel.any():
            labs = label_bucket[sel]
            for a in range(NUM_ACTIONS):
                pf_open[p, a] += int((labs == a).sum())

    # preflop vs raise (street=0, n_preflop_raises >= 1)
    vs_raise_mask = (street == 0) & (n_preflop_raises >= 1)
    for p in range(MAX_PLAYERS):
        sel = vs_raise_mask & (pos == p)
        if sel.any():
            labs = label_bucket[sel]
            for a in range(NUM_ACTIONS):
                pf_vs_raise[p, a] += int((labs == a).sum())

    # postflop facing bet
    postflop = street >= 1
    facing = postflop & (to_call_bb > 1e-9)
    no_bet = postflop & (to_call_bb <= 1e-9)
    for p in range(MAX_PLAYERS):
        s1 = facing & (pos == p)
        s2 = no_bet & (pos == p)
        if s1.any():
            labs = label_bucket[s1]
            for a in range(NUM_ACTIONS):
                pf_postflop_facing_bet[p, a] += int((labs == a).sum())
        if s2.any():
            labs = label_bucket[s2]
            for a in range(NUM_ACTIONS):
                pf_postflop_no_bet[p, a] += int((labs == a).sum())

    # raise bucket histogram
    for i, b in enumerate(range(RAISE_START, ALL_IN)):
        raise_bucket_hist[i] += int((label_bucket == b).sum())
    raise_bucket_hist[-1] += int((label_bucket == ALL_IN).sum())

    # per-bucket raise-fraction mean (to compare with rfrac head prediction later)
    for i, b in enumerate(range(RAISE_START, ALL_IN + 1)):  # include all-in at index 7
        sel = label_bucket == b
        if sel.any():
            rfrac_sum_by_bucket[i] += float(label_rfrac[sel].sum())
            rfrac_count_by_bucket[i] += int(sel.sum())

    # raise-sizing x street
    for st in range(4):
        for i, b in enumerate(range(RAISE_START, ALL_IN)):
            raise_sizing_by_street[st, i] += int(((label_bucket == b) & (street == st)).sum())
        raise_sizing_by_street[st, -1] += int(((label_bucket == ALL_IN) & (street == st)).sum())

    # street & num_active distributions (sanity-check the data mix)
    for st in range(4):
        street_distribution[st] += int((street == st).sum())
    for k in range(MAX_PLAYERS + 1):
        n_active_distribution[k] += int((num_active == k).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-glob", default="processed/shard_*.npz")
    ap.add_argument("--max-shards", type=int, default=20,
                    help="limit to this many shards for speed")
    ap.add_argument("--out", default="stats_training.json")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.shard_glob))
    if not paths:
        raise SystemExit(f"No shards match {args.shard_glob}")
    if args.max_shards > 0 and len(paths) > args.max_shards:
        # evenly sample
        idxs = np.linspace(0, len(paths) - 1, num=args.max_shards).astype(int)
        paths = [paths[i] for i in idxs]
    print(f"Reading {len(paths)} shards", flush=True)

    action_total = np.zeros(NUM_ACTIONS, dtype=np.int64)
    action_per_street = np.zeros((4, NUM_ACTIONS), dtype=np.int64)
    action_per_pos = np.zeros((MAX_PLAYERS, NUM_ACTIONS), dtype=np.int64)
    pf_open = np.zeros((MAX_PLAYERS, NUM_ACTIONS), dtype=np.int64)
    pf_vs_raise = np.zeros((MAX_PLAYERS, NUM_ACTIONS), dtype=np.int64)
    pf_postflop_facing_bet = np.zeros((MAX_PLAYERS, NUM_ACTIONS), dtype=np.int64)
    pf_postflop_no_bet = np.zeros((MAX_PLAYERS, NUM_ACTIONS), dtype=np.int64)
    raise_bucket_hist = np.zeros(len(RAISE_POT_FRACTIONS) + 1, dtype=np.int64)
    raise_sizing_by_street = np.zeros((4, len(RAISE_POT_FRACTIONS) + 1), dtype=np.int64)
    rfrac_sum_by_bucket = np.zeros(len(RAISE_POT_FRACTIONS) + 1, dtype=np.float64)
    rfrac_count_by_bucket = np.zeros(len(RAISE_POT_FRACTIONS) + 1, dtype=np.int64)
    street_distribution = np.zeros(4, dtype=np.int64)
    n_active_distribution = np.zeros(MAX_PLAYERS + 1, dtype=np.int64)

    t0 = time.time()
    total_decisions = 0
    for i, p in enumerate(paths):
        with np.load(p) as z:
            data = {k: z[k] for k in z.files}
        _accumulate_shard(
            data,
            action_total, action_per_street, action_per_pos,
            pf_open, pf_vs_raise,
            pf_postflop_facing_bet, pf_postflop_no_bet,
            raise_bucket_hist, raise_sizing_by_street,
            rfrac_sum_by_bucket, rfrac_count_by_bucket,
            street_distribution, n_active_distribution,
        )
        total_decisions += len(data["label_bucket"])
        if (i + 1) % 5 == 0 or i == len(paths) - 1:
            print(f"  shard {i+1}/{len(paths)}  total_samples={total_decisions:,}  "
                  f"elapsed={time.time()-t0:.1f}s",
                  flush=True)

    def _norm(row):
        s = row.sum()
        return (row / s).tolist() if s > 0 else [0.0] * len(row)

    def _mat_norm(mat):
        return [_norm(mat[i]) for i in range(mat.shape[0])]

    rfrac_mean = [
        float(rfrac_sum_by_bucket[i] / rfrac_count_by_bucket[i]) if rfrac_count_by_bucket[i] > 0 else 0.0
        for i in range(len(rfrac_sum_by_bucket))
    ]

    out = {
        "n_decisions": total_decisions,
        "n_shards": len(paths),
        "action_names": list(ACTION_NAMES),
        "street_names": STREET_NAMES,
        "pos_names": POS_NAMES,
        "action_freq_overall": _norm(action_total),
        "action_count_overall": action_total.tolist(),
        "action_freq_per_street": _mat_norm(action_per_street),
        "action_count_per_street": action_per_street.tolist(),
        "action_freq_per_pos": _mat_norm(action_per_pos),
        "action_count_per_pos": action_per_pos.tolist(),
        "pf_open_freq_per_pos": _mat_norm(pf_open),
        "pf_open_count_per_pos": pf_open.tolist(),
        "pf_vs_raise_freq_per_pos": _mat_norm(pf_vs_raise),
        "pf_vs_raise_count_per_pos": pf_vs_raise.tolist(),
        "postflop_facing_bet_freq": _mat_norm(pf_postflop_facing_bet),
        "postflop_facing_bet_count": pf_postflop_facing_bet.tolist(),
        "postflop_no_bet_freq": _mat_norm(pf_postflop_no_bet),
        "postflop_no_bet_count": pf_postflop_no_bet.tolist(),
        "raise_bucket_count": raise_bucket_hist.tolist(),
        "raise_bucket_labels": [f"{f:g}x-pot" for f in RAISE_POT_FRACTIONS] + ["all-in"],
        "raise_bucket_freq": _norm(raise_bucket_hist),
        "raise_sizing_by_street_count": raise_sizing_by_street.tolist(),
        "raise_sizing_by_street_freq": _mat_norm(raise_sizing_by_street),
        "raise_rfrac_mean_by_bucket": rfrac_mean,
        "street_distribution_freq": _norm(street_distribution),
        "street_distribution_count": street_distribution.tolist(),
        "num_active_distribution_freq": _norm(n_active_distribution),
        "num_active_distribution_count": n_active_distribution.tolist(),
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}  (total decisions {total_decisions:,})")


if __name__ == "__main__":
    main()
