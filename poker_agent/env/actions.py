"""Action space: 10 discrete buckets.

  0  FOLD
  1  CHECK_CALL
  2  RAISE_0.33x pot
  3  RAISE_0.50x pot
  4  RAISE_0.75x pot
  5  RAISE_1.00x pot
  6  RAISE_1.50x pot
  7  RAISE_2.00x pot
  8  RAISE_3.00x pot
  9  ALL_IN

Raise sizes are expressed as a multiplier of the *current pot after the
caller has matched the previous bet* (the standard "pot-relative" raise
convention used by most online clients).
"""
from __future__ import annotations

from dataclasses import dataclass

FOLD = 0
CHECK_CALL = 1
RAISE_START = 2
ALL_IN = 9
NUM_ACTIONS = 10

RAISE_POT_FRACTIONS: tuple[float, ...] = (0.33, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
assert len(RAISE_POT_FRACTIONS) == ALL_IN - RAISE_START
RAISE_BUCKETS = tuple(range(RAISE_START, ALL_IN))  # (2..8)

ACTION_NAMES = (
    "F", "X/C",
    "R0.33", "R0.50", "R0.75", "R1.00", "R1.50", "R2.00", "R3.00",
    "ALL_IN",
)


@dataclass(frozen=True)
class LegalActionMask:
    """Which of the 9 bucketed actions are legal in the current state."""

    mask: tuple[bool, ...]  # length NUM_ACTIONS

    def as_list(self) -> list[int]:
        return [i for i, m in enumerate(self.mask) if m]


def bucketize_raise(raise_to_chips: float, pot_before_call: float, to_call: float,
                    stack_remaining: float) -> int:
    """Given an actual raise amount (total bet in chips), snap to nearest bucket.

    Args:
        raise_to_chips:  target TOTAL bet the player is raising to (chips).
        pot_before_call: pot size BEFORE the acting player matches to_call.
        to_call:         chips the acting player still owes to match.
        stack_remaining: acting player's remaining stack (before this action).

    Returns one of RAISE_START..ALL_IN.
    """
    raise_over_call = max(0.0, raise_to_chips - to_call)  # the "increment" above call
    if raise_over_call >= stack_remaining - 1e-6:
        return ALL_IN
    # Effective post-call pot that the raise is relative to.
    post_call_pot = pot_before_call + to_call
    if post_call_pot <= 1e-9:
        return RAISE_START  # degenerate, shouldn't happen
    ratio = raise_over_call / post_call_pot
    # Find closest fraction.
    best = RAISE_START
    best_d = abs(ratio - RAISE_POT_FRACTIONS[0])
    for i, f in enumerate(RAISE_POT_FRACTIONS):
        d = abs(ratio - f)
        if d < best_d:
            best_d = d
            best = RAISE_START + i
    return best


def desnap_raise_to(bucket: int, pot_before_call: float, to_call: float,
                     stack_remaining: float, min_raise_to: float) -> float:
    """Convert a bucket back to an actual 'raise to' chip amount.

    Used by the RL env when the hero picks a bucket.
    """
    if bucket == ALL_IN:
        return to_call + stack_remaining  # actually all-in
    if bucket < RAISE_START or bucket >= ALL_IN:
        raise ValueError(f"not a raise bucket: {bucket}")
    frac = RAISE_POT_FRACTIONS[bucket - RAISE_START]
    post_call_pot = pot_before_call + to_call
    raise_over_call = frac * post_call_pot
    raise_to = to_call + raise_over_call
    # enforce min raise and stack cap
    raise_to = max(raise_to, min_raise_to)
    raise_to = min(raise_to, to_call + stack_remaining)
    return raise_to
