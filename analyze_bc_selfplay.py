"""Run 6-BC self-play and collect aggregate action-distribution statistics.

All 6 seats use the same BC model. Hands are run in parallel (batched across
parallel games) for speed: each tick we collect all to-act observations,
batch them through the model, then apply actions.

Outputs:
  - stats_selfplay.json: aggregate counters (dict of dicts)
  - optional sample text of first --dump-first N hands for manual spot-check

Usage:
  python analyze_bc_selfplay.py \
      --ckpt checkpoints/bc_full/best.pt \
      --n-hands 2000 \
      --parallel 128 \
      --out stats_selfplay.json \
      --dump-first 20 --dump-out sample_bc_selfplay.txt
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
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
    NUM_ACTIONS,
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
STREET_NAMES = ["pre", "flop", "turn", "river"]
SB, BB = 0.5, 1.0
START_STACK = 100.0


# --------------- pretty printing for spot-check samples ----------------- #

def _fmt_cards(cs):
    return " ".join(int_to_card(int(c)) for c in cs)


def _bucket_label(b):
    if b == FOLD:
        return "fold"
    if b == CHECK_CALL:
        return "check/call"
    if b == ALL_IN:
        return "ALL-IN"
    if RAISE_START <= b < ALL_IN:
        return f"raise {RAISE_POT_FRACTIONS[b - RAISE_START]:g}x-pot"
    return ACTION_NAMES[b]


# --------------- stat collection ----------------------------------------- #

class StatCollector:
    """Aggregate counters for BC action distribution."""

    def __init__(self) -> None:
        # global action freq
        self.action_total = np.zeros(NUM_ACTIONS, dtype=np.int64)
        # per-street action freq
        self.action_per_street = np.zeros((4, NUM_ACTIONS), dtype=np.int64)
        # per-position action freq
        self.action_per_pos = np.zeros((MAX_PLAYERS, NUM_ACTIONS), dtype=np.int64)
        # preflop open (first voluntary action, num_active=6, to_call==bb) action per pos
        self.pfr_open = np.zeros((MAX_PLAYERS, NUM_ACTIONS), dtype=np.int64)
        # vs-3bet: preflop when someone already raised beyond BB and we haven't acted
        self.pf_vs_raise = np.zeros((MAX_PLAYERS, NUM_ACTIONS), dtype=np.int64)
        # facing-bet (to_call > 0) postflop action
        self.pf_postflop_facing_bet = np.zeros((MAX_PLAYERS, NUM_ACTIONS), dtype=np.int64)
        # no-bet (to_call==0) postflop action (checks vs bets)
        self.pf_postflop_no_bet = np.zeros((MAX_PLAYERS, NUM_ACTIONS), dtype=np.int64)
        # raise-bucket histogram (conditioning on raise chosen)
        self.raise_bucket_hist = np.zeros(len(RAISE_POT_FRACTIONS) + 1, dtype=np.int64)  # +1 for ALL_IN
        # poker indices (per hand)
        self.vpip_hands = 0   # hands where at least one non-blind voluntary money in by any seat
        self.hand_count = 0
        # per-seat per-hand indicators
        self.vpip_by_seat = np.zeros(MAX_PLAYERS, dtype=np.int64)
        self.pfr_by_seat = np.zeros(MAX_PLAYERS, dtype=np.int64)
        self.threebet_opp_by_seat = np.zeros(MAX_PLAYERS, dtype=np.int64)  # denominator: got to face a raise preflop
        self.threebet_by_seat = np.zeros(MAX_PLAYERS, dtype=np.int64)     # reraised preflop
        self.fold_to_threebet_by_seat = np.zeros(MAX_PLAYERS, dtype=np.int64)
        self.faced_threebet_by_seat = np.zeros(MAX_PLAYERS, dtype=np.int64)
        # showdown & length
        self.showdown_hands = 0
        self.all_in_hands = 0
        self.hand_lengths: list[int] = []
        self.street_reached = np.zeros(4, dtype=np.int64)  # pre,flop,turn,river
        # mask vs predicted legality
        self.illegal_sampled = 0
        self.total_decisions = 0
        # policy entropy (approx)
        self.entropy_sum = 0.0
        self.entropy_n = 0

    def record_decision(self, *, street, pos, to_call, current_bet, num_preflop_raises,
                        action, raise_inc_bb_so_far, is_first_voluntary_preflop,
                        entropy):
        self.action_total[action] += 1
        self.action_per_street[street, action] += 1
        self.action_per_pos[pos, action] += 1
        self.total_decisions += 1
        self.entropy_sum += entropy
        self.entropy_n += 1
        # raise sub-hist
        if RAISE_START <= action < ALL_IN:
            self.raise_bucket_hist[action - RAISE_START] += 1
        elif action == ALL_IN:
            self.raise_bucket_hist[-1] += 1

        if street == 0:  # preflop
            if is_first_voluntary_preflop and num_preflop_raises == 0:
                self.pfr_open[pos, action] += 1
            elif num_preflop_raises >= 1:
                self.pf_vs_raise[pos, action] += 1
        else:
            if to_call > 1e-9:
                self.pf_postflop_facing_bet[pos, action] += 1
            else:
                self.pf_postflop_no_bet[pos, action] += 1

    def finalize_hand(self, *, per_seat_vpip, per_seat_pfr,
                      per_seat_faced_raise_pf, per_seat_reraised_pf,
                      per_seat_faced_3bet, per_seat_folded_to_3bet,
                      final_street, was_showdown, any_all_in, hand_action_count):
        self.hand_count += 1
        self.vpip_hands += int(any(per_seat_vpip))
        self.vpip_by_seat += per_seat_vpip.astype(np.int64)
        self.pfr_by_seat += per_seat_pfr.astype(np.int64)
        self.threebet_opp_by_seat += per_seat_faced_raise_pf.astype(np.int64)
        self.threebet_by_seat += per_seat_reraised_pf.astype(np.int64)
        self.faced_threebet_by_seat += per_seat_faced_3bet.astype(np.int64)
        self.fold_to_threebet_by_seat += per_seat_folded_to_3bet.astype(np.int64)
        self.showdown_hands += int(was_showdown)
        self.all_in_hands += int(any_all_in)
        self.hand_lengths.append(hand_action_count)
        self.street_reached[: final_street + 1] += 1

    def to_dict(self) -> dict:
        def _norm_row(row):
            s = row.sum()
            return (row / s).tolist() if s > 0 else [0.0] * len(row)

        def _matrix_freq(mat):
            return [_norm_row(mat[i]) for i in range(mat.shape[0])]

        return {
            "n_hands": int(self.hand_count),
            "n_decisions": int(self.total_decisions),
            "action_names": list(ACTION_NAMES),
            "street_names": STREET_NAMES,
            "pos_names": POS_NAMES,
            # overall action dist
            "action_freq_overall": _norm_row(self.action_total),
            "action_count_overall": self.action_total.tolist(),
            # per-street action dist
            "action_freq_per_street": _matrix_freq(self.action_per_street),
            "action_count_per_street": self.action_per_street.tolist(),
            # per-position action dist
            "action_freq_per_pos": _matrix_freq(self.action_per_pos),
            "action_count_per_pos": self.action_per_pos.tolist(),
            # preflop open-raise / cold-call / fold
            "pf_open_freq_per_pos": _matrix_freq(self.pfr_open),
            "pf_open_count_per_pos": self.pfr_open.tolist(),
            # preflop vs raise
            "pf_vs_raise_freq_per_pos": _matrix_freq(self.pf_vs_raise),
            "pf_vs_raise_count_per_pos": self.pf_vs_raise.tolist(),
            # postflop facing bet / checked-to
            "postflop_facing_bet_freq": _matrix_freq(self.pf_postflop_facing_bet),
            "postflop_facing_bet_count": self.pf_postflop_facing_bet.tolist(),
            "postflop_no_bet_freq": _matrix_freq(self.pf_postflop_no_bet),
            "postflop_no_bet_count": self.pf_postflop_no_bet.tolist(),
            # raise bucket histogram (7 pot-frac + all-in)
            "raise_bucket_count": self.raise_bucket_hist.tolist(),
            "raise_bucket_labels": [f"{f:g}x-pot" for f in RAISE_POT_FRACTIONS] + ["all-in"],
            "raise_bucket_freq": _norm_row(self.raise_bucket_hist),
            # per-seat metrics per-hand
            "vpip_per_seat": (self.vpip_by_seat / max(1, self.hand_count)).tolist(),
            "pfr_per_seat":  (self.pfr_by_seat / max(1, self.hand_count)).tolist(),
            "threebet_per_seat_when_opportunity": (
                self.threebet_by_seat / np.maximum(1, self.threebet_opp_by_seat)
            ).tolist(),
            "fold_to_threebet_per_seat": (
                self.fold_to_threebet_by_seat / np.maximum(1, self.faced_threebet_by_seat)
            ).tolist(),
            # hand-level
            "showdown_rate": self.showdown_hands / max(1, self.hand_count),
            "all_in_rate": self.all_in_hands / max(1, self.hand_count),
            "avg_actions_per_hand": float(np.mean(self.hand_lengths) if self.hand_lengths else 0.0),
            "street_reach_rate": (self.street_reached / max(1, self.hand_count)).tolist(),
            "policy_entropy_nats_avg": self.entropy_sum / max(1, self.entropy_n),
        }


# --------------- parallel play driver ----------------------------------- #

@torch.no_grad()
def _batched_action(model, obs_list, mask_list, device):
    batch = {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}
    tb = obs_to_tensor(batch, device)
    out = model(tb)
    logits = out["policy_logits"].float()
    masks = torch.from_numpy(np.stack(mask_list)).to(device)
    ml = masked_logits(logits, masks)
    probs = F.softmax(ml, dim=-1)
    # entropy per sample
    ent = -(probs.clamp_min(1e-12).log() * probs).sum(-1).cpu().numpy()
    actions = torch.multinomial(probs, num_samples=1).squeeze(-1).cpu().numpy()
    return actions.astype(np.int64), ent


def run_selfplay(ckpt_path: str, n_hands: int, parallel: int, device: str,
                 seed: int, stats: StatCollector,
                 dump_first: int = 0, dump_out: str | None = None) -> None:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"Loading BC from {ckpt_path} on {device}", flush=True)
    model = load_bc_checkpoint(ckpt_path, map_location=device).to(device).eval()

    dump_fh = open(dump_out, "w", encoding="utf-8") if (dump_first > 0 and dump_out) else None

    hands_done = 0
    # per-game bookkeeping state for the parallel slots
    slots = []

    def _new_slot(hand_id: int):
        hero_seat = rng.randrange(MAX_PLAYERS)  # not used for 6-BC but keeps env happy
        stacks = [START_STACK] * MAX_PLAYERS
        s = _start_hand(MAX_PLAYERS, stacks, SB, BB, hero_seat, rng)
        return {
            "s": s,
            "hand_id": hand_id,
            # preflop bookkeeping
            "pf_raises": 0,                                    # # of raises so far preflop
            "pf_open_raiser": -1,
            "pf_three_bettor": -1,
            "pf_first_voluntary_seat": -1,                     # first seat beyond BB to act (not blinded-in)
            "seat_first_preflop_action": np.zeros(MAX_PLAYERS, dtype=bool),
            "vpip": np.zeros(MAX_PLAYERS, dtype=bool),
            "pfr": np.zeros(MAX_PLAYERS, dtype=bool),
            "faced_raise_pf": np.zeros(MAX_PLAYERS, dtype=bool),
            "reraised_pf": np.zeros(MAX_PLAYERS, dtype=bool),
            "faced_3bet": np.zeros(MAX_PLAYERS, dtype=bool),
            "folded_to_3bet": np.zeros(MAX_PLAYERS, dtype=bool),
            "action_count": 0,
            "events": [] if (dump_fh and hand_id <= dump_first) else None,
            "hero_hole": None,
        }

    for i in range(min(parallel, n_hands)):
        slots.append(_new_slot(hands_done + 1))
        hands_done += 1
    t0 = time.time()
    ticks = 0
    while True:
        # Collect actors that need a decision
        obs_list = []
        mask_list = []
        idx_list = []
        for idx, slot in enumerate(slots):
            if slot is None:
                continue
            s = slot["s"]
            if s.done or s.to_act < 0:
                continue
            obs = get_observation(s, s.to_act)
            mask = legal_action_mask(s)
            if not mask.any():
                # no legal action — force terminate
                s.done = True
                continue
            obs_list.append(obs)
            mask_list.append(mask)
            idx_list.append(idx)

        if obs_list:
            actions, entropies = _batched_action(model, obs_list, mask_list, device)
        else:
            actions, entropies = [], []

        # apply
        for k, slot_idx in enumerate(idx_list):
            slot = slots[slot_idx]
            s = slot["s"]
            pos = s.to_act
            p = s.players[pos]
            street = s.street
            to_call = max(0.0, s.current_bet - p.committed_street)
            cur_bet = s.current_bet
            action = int(actions[k])
            # safety: if sampled illegal (shouldn't happen thanks to mask), clamp
            if not mask_list[k][action]:
                stats.illegal_sampled += 1
                action = CHECK_CALL if mask_list[k][CHECK_CALL] else int(np.flatnonzero(mask_list[k])[0])

            # for PFR open detection we need to know: is this seat's FIRST voluntary preflop
            # action and are there zero raises so far? "first voluntary" excludes blinds.
            is_first_voluntary_preflop = False
            if street == 0 and not slot["seat_first_preflop_action"][pos]:
                slot["seat_first_preflop_action"][pos] = True
                is_first_voluntary_preflop = True

            # record stat BEFORE state mutates
            stats.record_decision(
                street=street, pos=pos,
                to_call=to_call, current_bet=cur_bet,
                num_preflop_raises=slot["pf_raises"],
                action=action,
                raise_inc_bb_so_far=cur_bet,
                is_first_voluntary_preflop=is_first_voluntary_preflop,
                entropy=float(entropies[k]),
            )

            # preflop per-seat per-hand stats
            if street == 0:
                # VPIP = voluntarily put chips in (SB completing, BB calling a raise,
                # non-blind call or raise). BB checking a limp is NOT VPIP.
                if action == CHECK_CALL and to_call > 1e-9:
                    slot["vpip"][pos] = True
                if RAISE_START <= action <= ALL_IN:
                    slot["vpip"][pos] = True
                    slot["pfr"][pos] = True
                    if slot["pf_raises"] >= 1:
                        # this is a 3bet+ (over a raise)
                        slot["reraised_pf"][pos] = True
                    if slot["pf_raises"] == 0:
                        slot["pf_open_raiser"] = pos
                    elif slot["pf_raises"] == 1:
                        slot["pf_three_bettor"] = pos
                # being forced to face a raise (for 3bet denominator)
                if slot["pf_raises"] >= 1 and pos != slot["pf_open_raiser"]:
                    slot["faced_raise_pf"][pos] = True
                # facing a 3bet (open raiser after a reraise)
                if slot["pf_raises"] >= 2 and pos == slot["pf_open_raiser"]:
                    slot["faced_3bet"][pos] = True
                    if action == FOLD:
                        slot["folded_to_3bet"][pos] = True

            # store event for dump
            if slot["events"] is not None:
                stack_before = p.stack
                pot_before = s.pot
                if slot["hero_hole"] is None:
                    slot["hero_hole"] = list(s.players[s.hero_seat].hole)
                apply_action(s, action)
                slot["events"].append({
                    "street": street,
                    "pos": pos,
                    "action": action,
                    "to_call": to_call,
                    "pot_before": pot_before,
                    "amt_in": stack_before - p.stack,
                    "all_in": p.all_in,
                })
            else:
                apply_action(s, action)

            slot["action_count"] += 1
            # update raise count after apply: increase when bucket was a raise or all-in
            if street == 0 and RAISE_START <= action <= ALL_IN:
                slot["pf_raises"] += 1

        # finalize any finished slots + start new hands
        for idx, slot in enumerate(slots):
            if slot is None:
                continue
            s = slot["s"]
            if not s.done:
                continue
            # finalize
            non_folded = [pl for pl in s.players if not pl.folded]
            was_showdown = len(non_folded) >= 2
            any_all_in = any(pl.all_in for pl in s.players)
            nb = len(s.board)
            final_street = 3 if nb >= 5 else (2 if nb == 4 else (1 if nb >= 3 else 0))
            stats.finalize_hand(
                per_seat_vpip=slot["vpip"],
                per_seat_pfr=slot["pfr"],
                per_seat_faced_raise_pf=slot["faced_raise_pf"],
                per_seat_reraised_pf=slot["reraised_pf"],
                per_seat_faced_3bet=slot["faced_3bet"],
                per_seat_folded_to_3bet=slot["folded_to_3bet"],
                final_street=final_street,
                was_showdown=was_showdown,
                any_all_in=any_all_in,
                hand_action_count=slot["action_count"],
            )
            # dump text for early hands
            if slot["events"] is not None and dump_fh is not None:
                _write_dump_hand(dump_fh, slot, s)

            # next hand
            if hands_done < n_hands:
                slots[idx] = _new_slot(hands_done + 1)
                hands_done += 1
            else:
                slots[idx] = None

        ticks += 1
        if ticks % 50 == 0:
            done = stats.hand_count
            alive = sum(1 for x in slots if x is not None)
            print(f"  tick {ticks:5d}  hands_done={done:5d}/{n_hands}  "
                  f"decisions={stats.total_decisions:7d}  alive={alive}  "
                  f"elapsed={time.time()-t0:.1f}s",
                  flush=True)

        if all(x is None for x in slots):
            break

    if dump_fh is not None:
        dump_fh.close()

    print(f"Self-play done: hands={stats.hand_count}, decisions={stats.total_decisions}, "
          f"elapsed={time.time()-t0:.1f}s")


def _write_dump_hand(fh, slot, s_final):
    hid = slot["hand_id"]
    hh = slot["hero_hole"] or []
    fh.write(f"========== Hand #{hid} ==========\n")
    fh.write(f"Hero seat {s_final.hero_seat} ({POS_NAMES[s_final.hero_seat]}) "
             f"hole {_fmt_cards(hh)}\n")
    current_street = -1
    for ev in slot["events"]:
        if ev["street"] != current_street:
            current_street = ev["street"]
            fh.write(f"-- {STREET_NAMES[current_street]} --\n")
        who = POS_NAMES[ev["pos"]]
        b = ev["action"]
        if b == FOLD:
            fh.write(f"  {who}: folds\n")
        elif b == CHECK_CALL:
            if ev["to_call"] <= 1e-9:
                fh.write(f"  {who}: checks\n")
            else:
                fh.write(f"  {who}: calls {ev['amt_in']:.2f} (to_call {ev['to_call']:.2f})\n")
        else:
            tag = _bucket_label(b)
            flag = " [ALL-IN]" if ev["all_in"] else ""
            fh.write(f"  {who}: {tag}  (+{ev['amt_in']:.2f}, pot_before {ev['pot_before']:.1f}){flag}\n")
    fh.write(f"Final board: [{_fmt_cards(s_final.board)}]\n")
    winners = [POS_NAMES[w] for w in s_final.winner_seats] or ["(none)"]
    fh.write(f"Winners: {', '.join(winners)}\n\n")


# --------------- main --------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/bc_full/best.pt")
    ap.add_argument("--n-hands", type=int, default=2000)
    ap.add_argument("--parallel", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=20260422)
    ap.add_argument("--out", default="stats_selfplay.json")
    ap.add_argument("--dump-first", type=int, default=20,
                    help="write this many hands in human-readable text for spot-check")
    ap.add_argument("--dump-out", default="sample_bc_selfplay.txt")
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    stats = StatCollector()
    run_selfplay(args.ckpt, args.n_hands, args.parallel, device, args.seed,
                 stats, dump_first=args.dump_first, dump_out=args.dump_out)

    d = stats.to_dict()
    Path(args.out).write_text(json.dumps(d, indent=2), encoding="utf-8")
    print(f"Wrote stats to {args.out}")


if __name__ == "__main__":
    main()
