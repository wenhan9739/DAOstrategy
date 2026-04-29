"""Dump human-readable 6-max NLHE hands between a HERO ckpt and the frozen BC pool.

Usage (defaults shown):
    python dump_hands.py
        --hero-ckpt checkpoints/ppo_full/iter_00500.pt
        --opp-ckpt  checkpoints/bc_full/best.pt
        --n-hands   10000
        --out       hands_iter500.txt
        --device    cpu

Runs on CPU by default so it does not compete with an ongoing GPU training run.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from poker_agent.data.features import MAX_PLAYERS
from poker_agent.env.actions import (
    ACTION_NAMES,
    ALL_IN,
    CHECK_CALL,
    FOLD,
    RAISE_POT_FRACTIONS,
    RAISE_START,
)
from poker_agent.env.cards import int_to_card
from poker_agent.env.holdem import (
    _start_hand,
    apply_action,
    get_observation,
    legal_action_mask,
)
from poker_agent.models.ppo import load_bc_checkpoint, masked_logits
from poker_agent.training.train_ppo import obs_to_tensor

POS_NAMES = ["SB", "BB", "UTG", "HJ", "CO", "BTN"]
START_STACK_BB = 100.0
SB = 0.5
BB = 1.0


def fmt_cards(cs) -> str:
    return " ".join(int_to_card(int(c)) for c in cs)


def bucket_label(bucket: int) -> str:
    if bucket == FOLD:
        return "fold"
    if bucket == CHECK_CALL:
        return "check/call"
    if bucket == ALL_IN:
        return "ALL-IN"
    if RAISE_START <= bucket < ALL_IN:
        frac = RAISE_POT_FRACTIONS[bucket - RAISE_START]
        return f"raise {frac:g}x-pot"
    return ACTION_NAMES[bucket] if 0 <= bucket < len(ACTION_NAMES) else f"bucket{bucket}"


@torch.no_grad()
def batched_act(model, obs_list, mask_list, device):
    """Single batched forward for a list of obs/masks. Returns list[int] of actions."""
    if not obs_list:
        return []
    batch = {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}
    tb = obs_to_tensor(batch, device)
    out = model(tb)
    logits = out["policy_logits"].float()
    masks = torch.from_numpy(np.stack(mask_list)).to(device)
    ml = masked_logits(logits, masks)
    probs = F.softmax(ml, dim=-1)
    # sample (matches training-time stochastic play)
    actions = torch.multinomial(probs, num_samples=1).squeeze(-1).cpu().numpy()
    return [int(a) for a in actions]


def play_one_hand(s, hero_model, opp_model, device) -> list[dict]:
    """Play one HandState to completion, return list of events for pretty-printing."""
    events = []
    prev_board_len = 0
    guard = 0
    while not s.done and guard < 2000:
        guard += 1
        actor = s.to_act
        if actor < 0:
            break
        p = s.players[actor]

        to_call_before = max(0.0, s.current_bet - p.committed_street)
        committed_total_before = p.committed_total
        stack_before = p.stack
        pot_before = s.pot
        street_before = s.street

        obs = get_observation(s, actor)
        mask = legal_action_mask(s)

        model = hero_model if actor == s.hero_seat else opp_model
        action = batched_act(model, [obs], [mask], device)[0]
        if not mask[action]:  # safety: env would also sanitize
            action = CHECK_CALL if mask[CHECK_CALL] else int(np.flatnonzero(mask)[0])

        apply_action(s, action)

        # Use committed_total / stack diff for the amount put in this action:
        # committed_street may have been reset to 0 by _end_street if the
        # action ended the round.
        amt_put_in = stack_before - p.stack  # chips added to the pot this action
        committed_after = p.committed_total - committed_total_before  # same value, kept for display

        events.append({
            "street": street_before,
            "actor": actor,
            "is_hero": actor == s.hero_seat,
            "bucket": action,
            "to_call_before": to_call_before,
            "pot_before": pot_before,
            "amt_put_in": amt_put_in,
            "committed_after": committed_after,
            "stack_after": p.stack,
            "is_all_in_after": p.all_in,
        })

        # detect newly dealt board cards from _advance_street
        if len(s.board) > prev_board_len:
            new_cards = s.board[prev_board_len:]
            labels = {3: "Flop", 4: "Turn", 5: "River"}
            label = labels.get(len(s.board), f"Board({len(s.board)})")
            events.append({
                "deal": label,
                "cards": list(new_cards),
                "pot_before": s.pot,
            })
            prev_board_len = len(s.board)

    return events


def render_hand(hand_idx: int, seed: int, s_final, events, hero_hole, all_holes) -> str:
    """Render one hand as a multiline text block."""
    out = []
    out.append(f"========== Hand #{hand_idx}  (seed={seed}) ==========")
    hs = s_final.hero_seat
    out.append(f"Hero: seat {hs} ({POS_NAMES[hs]})  hole: {fmt_cards(hero_hole)}")
    out.append("Seats: " + " | ".join(
        f"{POS_NAMES[i]}={START_STACK_BB:g}bb" for i in range(MAX_PLAYERS)
    ))
    out.append(f"Blinds: SB {SB} / BB {BB}  (initial pot {SB + BB})")

    # Group events by street
    street_names = {0: "Preflop", 1: "Flop", 2: "Turn", 3: "River"}
    deal_street_idx = {"Flop": 1, "Turn": 2, "River": 3}
    current_street = -1
    for ev in events:
        if "deal" in ev:
            out.append(f"-- {ev['deal']}: [{fmt_cards(ev['cards'])}]  (pot now {ev['pot_before']:.1f}) --")
            # mark this street as already-headered to avoid a redundant "-- Street --" line
            current_street = deal_street_idx.get(ev["deal"], current_street)
            continue
        if ev["street"] != current_street:
            current_street = ev["street"]
            out.append(f"-- {street_names.get(current_street, f'Street{current_street}')} --")
        who = POS_NAMES[ev["actor"]] + (" [HERO]" if ev["is_hero"] else "")
        b = ev["bucket"]
        if b == FOLD:
            line = f"  {who}: folds"
        elif b == CHECK_CALL:
            if ev["to_call_before"] <= 1e-9:
                line = f"  {who}: checks"
            else:
                line = f"  {who}: calls {ev['amt_put_in']:.2f}  (to_call was {ev['to_call_before']:.2f})"
        else:
            # raise/all-in
            tag = bucket_label(b)
            line = (f"  {who}: {tag}  to {ev['committed_after']:.2f}  "
                    f"(+{ev['amt_put_in']:.2f}, pot_before {ev['pot_before']:.1f})")
            if ev["is_all_in_after"]:
                line += "  [ALL-IN]"
        out.append(line)

    # board & showdown
    board_line = fmt_cards(s_final.board) if s_final.board else "(no board)"
    out.append(f"Final board: [{board_line}]")
    out.append(f"Final pot (before settle): ... winners took from pot={s_final.pot:.2f}")
    # winners
    winner_labels = [POS_NAMES[w] for w in s_final.winner_seats] or ["(none)"]
    out.append(f"Winners: {', '.join(winner_labels)}")
    # hero PnL in bb
    hero_final_stack = s_final.players[hs].stack
    hero_pnl = hero_final_stack - s_final.initial_stacks[hs]
    out.append(f"HERO PnL: {hero_pnl:+.2f} bb  (final stack {hero_final_stack:.2f})")
    # Showdown disclosure: reveal non-folded hands at showdown (multiple left)
    non_folded = [pl for pl in s_final.players if not pl.folded]
    if len(non_folded) >= 2:
        rows = []
        for pl in non_folded:
            rows.append(f"{POS_NAMES[pl.pos]}={fmt_cards(pl.hole)}")
        out.append("Showdown: " + "  ".join(rows))
    out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hero-ckpt", default="checkpoints/ppo_full/iter_00500.pt")
    ap.add_argument("--opp-ckpt", default="checkpoints/bc_full/best.pt")
    ap.add_argument("--n-hands", type=int, default=10000)
    ap.add_argument("--out", default="hands_iter500.txt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=123456)
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading HERO from {args.hero_ckpt}", flush=True)
    hero = load_bc_checkpoint(args.hero_ckpt, map_location=device).to(device).eval()
    print(f"Loading OPP  from {args.opp_ckpt}", flush=True)
    opp = load_bc_checkpoint(args.opp_ckpt, map_location=device).to(device).eval()

    rng = random.Random(args.seed)

    out_path = Path(args.out)
    fout = out_path.open("w", encoding="utf-8")
    header = (
        f"# Dump of {args.n_hands} hands\n"
        f"# HERO  : {args.hero_ckpt}\n"
        f"# OPP   : {args.opp_ckpt}\n"
        f"# seed  : {args.seed}\n"
        f"# stack : {START_STACK_BB} bb | blinds {SB}/{BB}\n"
        f"# positions: 0=SB 1=BB 2=UTG 3=HJ 4=CO 5=BTN (hero rotates)\n"
        f"\n"
    )
    fout.write(header)

    t0 = time.time()
    total_pnl_bb = 0.0
    for hand_idx in range(1, args.n_hands + 1):
        hero_seat = rng.randrange(MAX_PLAYERS)
        stacks = [START_STACK_BB] * MAX_PLAYERS
        s = _start_hand(MAX_PLAYERS, stacks, SB, BB, hero_seat, rng)
        hero_hole = list(s.players[hero_seat].hole)  # capture before mutation
        all_holes = [list(p.hole) for p in s.players]
        events = play_one_hand(s, hero, opp, device)
        text = render_hand(hand_idx, args.seed, s, events, hero_hole, all_holes)
        fout.write(text)
        fout.write("\n")
        total_pnl_bb += s.players[hero_seat].stack - s.initial_stacks[hero_seat]
        if hand_idx % 500 == 0:
            fout.flush()
            elapsed = time.time() - t0
            rate = hand_idx / max(1e-6, elapsed)
            mbb = (total_pnl_bb / hand_idx) * 1000.0
            eta = (args.n_hands - hand_idx) / max(1e-6, rate)
            print(f"  {hand_idx:6d}/{args.n_hands}  rate={rate:.1f} hand/s  "
                  f"mbb/hand(running)={mbb:+.1f}  elapsed={elapsed:.0f}s  eta={eta:.0f}s",
                  flush=True)

    footer = (
        f"\n# === Summary ===\n"
        f"# Hands     : {args.n_hands}\n"
        f"# Mean PnL  : {(total_pnl_bb/args.n_hands):+.4f} bb/hand  "
        f"({(total_pnl_bb/args.n_hands)*1000.0:+.1f} mbb/hand)\n"
        f"# Wall time : {time.time()-t0:.0f} s\n"
    )
    fout.write(footer)
    fout.close()
    print(footer, flush=True)
    print(f"Wrote {out_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
