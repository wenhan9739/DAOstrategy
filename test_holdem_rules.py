"""Unit-level smoke tests for the fixed 6-max NLHE env.

Run: python test_holdem_rules.py
"""
from __future__ import annotations

import random
import sys

from poker_agent.data.features import MAX_PLAYERS
from poker_agent.env.actions import ALL_IN, CHECK_CALL, FOLD, RAISE_START
from poker_agent.env.holdem import _start_hand, apply_action, legal_action_mask

SB, BB = 0.5, 1.0
POS = ["SB", "BB", "UTG", "HJ", "CO", "BTN"]


def new_hand(seed: int = 0):
    rng = random.Random(seed)
    stacks = [100.0] * MAX_PLAYERS
    return _start_hand(MAX_PLAYERS, stacks, SB, BB, hero_seat=0, rng=rng)


def assert_eq(actual, expected, msg: str):
    if actual != expected:
        raise AssertionError(f"{msg}: expected {expected}, got {actual}")


def check(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def act_sequence(s, seq):
    """Apply (expected_seat, bucket) pairs; verify to_act at each step."""
    for i, (seat, b) in enumerate(seq):
        assert_eq(s.to_act, seat, f"step {i}: to_act mismatch")
        mask = legal_action_mask(s)
        # if action illegal, substitute for inspection (but warn)
        if not mask[b]:
            raise AssertionError(f"step {i}: action {b} illegal for seat {seat} mask={list(mask)}")
        apply_action(s, b)


# -----------------------------------------------------------------------
# 1) Preflop: UTG raise, folds, SB 3-bet, BB fold, UTG call, flop reached.
# -----------------------------------------------------------------------
def test_preflop_3bet_then_call_advances_to_flop():
    s = new_hand()
    r_half = RAISE_START + 1  # "0.5x pot" bucket
    r_1x   = RAISE_START + 3  # "1x pot"
    seq = [
        (2, r_half),   # UTG opens 0.5x pot
        (3, FOLD),
        (4, FOLD),
        (5, FOLD),
        (0, r_1x),     # SB 3-bets
        (1, FOLD),
        (2, CHECK_CALL),  # UTG calls -> round should END, go to flop
    ]
    act_sequence(s, seq)
    # After UTG calls, we expect street to have advanced to the flop.
    check(s.street == 1, f"expected street=1 (flop) after UTG call, got {s.street}")
    check(len(s.board) == 3, f"expected flop to be 3 cards, got {len(s.board)}")
    # Post-flop, first-to-act among non-folded non-all-in is SB (seat 0).
    check(s.to_act == 0, f"expected SB to act first post-flop, got {s.to_act}")


# -----------------------------------------------------------------------
# 2) Post-flop: SB checks, BB must still get option (NOT street-end).
# -----------------------------------------------------------------------
def test_postflop_first_check_does_not_end_street():
    s = new_hand(seed=1)
    # Drive to flop: everyone folds to SB, SB limps, BB checks option.
    seq_pre = [
        (2, FOLD), (3, FOLD), (4, FOLD), (5, FOLD),
        (0, CHECK_CALL),   # SB completes to 1.0
        (1, CHECK_CALL),   # BB checks option -> flop
    ]
    act_sequence(s, seq_pre)
    check(s.street == 1, f"expected flop after BB checks option, got street={s.street}")
    check(s.to_act == 0, f"expected SB to act first on flop, got {s.to_act}")

    # SB checks — this used to buggy-end the street. Now: action must go to BB.
    apply_action(s, CHECK_CALL)
    check(s.street == 1, "SB's check MUST NOT end the flop; BB has not acted yet")
    check(s.to_act == 1, f"after SB check, BB should be to act, got {s.to_act}")

    # BB checks — NOW street ends, turn dealt.
    apply_action(s, CHECK_CALL)
    check(s.street == 2, f"after BB check, turn should come; got street={s.street}")
    check(len(s.board) == 4, f"expected 4 board cards after turn; got {len(s.board)}")
    check(s.to_act == 0, "turn first-to-act should be SB")


# -----------------------------------------------------------------------
# 3) BB gets option preflop when everyone limps.
# -----------------------------------------------------------------------
def test_bb_option_preflop_when_all_limp():
    s = new_hand(seed=2)
    seq = [
        (2, CHECK_CALL),  # UTG limp
        (3, CHECK_CALL),  # HJ limp
        (4, CHECK_CALL),  # CO limp
        (5, CHECK_CALL),  # BTN limp
        (0, CHECK_CALL),  # SB completes
    ]
    act_sequence(s, seq)
    # BB should now have the option (not auto-check-end).
    check(s.street == 0, f"BB option preflop: street must still be 0, got {s.street}")
    check(s.to_act == 1, f"BB must be to_act, got {s.to_act}")

    # BB raises (should reopen action to the other 5 players).
    r_1x = RAISE_START + 3
    apply_action(s, r_1x)
    check(s.street == 0, "BB's raise must not advance street")
    check(s.to_act == 2, f"after BB's option-raise action goes to UTG (2); got {s.to_act}")


# -----------------------------------------------------------------------
# 4) 2-bet call preflop: the last aggressor must NOT get extra action.
# -----------------------------------------------------------------------
def test_preflop_open_call_goes_to_flop_not_back_to_opener():
    s = new_hand(seed=3)
    r_half = RAISE_START + 1
    seq = [
        (2, r_half),       # UTG opens
        (3, FOLD), (4, FOLD), (5, FOLD), (0, FOLD),
        (1, CHECK_CALL),   # BB calls the open — round ends, flop
    ]
    act_sequence(s, seq)
    check(s.street == 1, "after BB calls UTG's open, flop must come")
    check(s.to_act == 1, "first to act post-flop should be first non-folded from SB = BB (seat 1)")


# -----------------------------------------------------------------------
# 5) All-in showdown fast-forward works and hand ends.
# -----------------------------------------------------------------------
def test_all_in_preflop_terminates_hand():
    s = new_hand(seed=4)
    # UTG shoves; others fold to BB; BB calls all-in.
    seq = [
        (2, ALL_IN),
        (3, FOLD), (4, FOLD), (5, FOLD), (0, FOLD),
        (1, CHECK_CALL),   # BB calls all-in (may go all-in too)
    ]
    act_sequence(s, seq)
    check(s.done, f"hand must end after BB calls all-in preflop; done={s.done}")
    check(s.street == 4, f"street should equal 4 at settle; got {s.street}")
    check(len(s.winner_seats) >= 1, "at least one winner must be set")


# -----------------------------------------------------------------------
# 6) 4-bet/5-bet with re-open action.
# -----------------------------------------------------------------------
def test_preflop_four_bet_chain_reopens_action():
    s = new_hand(seed=5)
    r_half = RAISE_START + 1
    r_1x   = RAISE_START + 3
    r_2x   = RAISE_START + 5
    seq = [
        (2, r_half),       # UTG open
        (3, FOLD), (4, FOLD),
        (5, r_1x),         # BTN 3-bets
        (0, FOLD), (1, FOLD),
        (2, r_2x),         # UTG 4-bets -> action must go back to BTN
    ]
    act_sequence(s, seq)
    check(s.street == 0, "still preflop after 4-bet")
    check(s.to_act == 5, f"BTN must be to act after UTG 4-bets; got {s.to_act}")

    # BTN calls the 4-bet -> round ends, flop.
    apply_action(s, CHECK_CALL)
    check(s.street == 1, "after BTN calls 4-bet, flop must come")


ALL_TESTS = [
    test_preflop_3bet_then_call_advances_to_flop,
    test_postflop_first_check_does_not_end_street,
    test_bb_option_preflop_when_all_limp,
    test_preflop_open_call_goes_to_flop_not_back_to_opener,
    test_all_in_preflop_terminates_hand,
    test_preflop_four_bet_chain_reopens_action,
]


def main():
    fails = 0
    for t in ALL_TESTS:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            fails += 1
            print(f"FAIL  {t.__name__}: {e}", file=sys.stderr)
    if fails:
        print(f"\n{fails}/{len(ALL_TESTS)} tests FAILED", file=sys.stderr)
        sys.exit(1)
    print(f"\nAll {len(ALL_TESTS)} tests PASSED.")


if __name__ == "__main__":
    main()
