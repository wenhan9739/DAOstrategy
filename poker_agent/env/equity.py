"""Hand strength evaluation for 7-card Texas Hold'em.

Fast drop-in backed by ``phevaluator`` (C++ lookup-table evaluator).

``phevaluator.evaluate_cards`` returns a rank where **lower = better**
(1 = royal flush, 7462 = worst high card). We expose a ``pick_winners``
compatible with the previous pokerkit-based API.

Measured ~40x faster than the pokerkit-based version, which removes the
showdown bottleneck in PPO rollouts.
"""
from __future__ import annotations

from typing import Sequence

from phevaluator import evaluate_cards

from .cards import UNKNOWN_CARD, int_to_card


# Precompute int -> 2-char card string once. Index 52 unused (we filter it out).
_CARD_STR: tuple[str, ...] = tuple(int_to_card(i) for i in range(52))

_WORST_RANK = 99999  # higher than any real phevaluator rank


def _rank7(hole: Sequence[int], board: Sequence[int]) -> int:
    """Return phevaluator rank (lower = better). Unknown cards are skipped.

    Requires at least 5 known cards to produce a meaningful rank.
    """
    strs: list[str] = []
    for c in hole:
        if c != UNKNOWN_CARD:
            strs.append(_CARD_STR[c])
    for c in board:
        if c != UNKNOWN_CARD:
            strs.append(_CARD_STR[c])
    if len(strs) < 5:
        return _WORST_RANK
    return evaluate_cards(*strs)


def best_hand_rank(hole: Sequence[int], board: Sequence[int]) -> int:
    """Return rank integer; lower is better (matches phevaluator convention)."""
    return _rank7(hole, board)


def compare(hole_a: Sequence[int], hole_b: Sequence[int], board: Sequence[int]) -> int:
    """-1 / 0 / +1 like before. a wins -> 1, b wins -> -1."""
    ra = _rank7(hole_a, board)
    rb = _rank7(hole_b, board)
    if ra < rb:
        return 1
    if ra > rb:
        return -1
    return 0


def pick_winners(holes: list[list[int]], board: list[int]) -> list[int]:
    """Return indices of winning player(s) among ``holes`` on ``board``.

    ``holes``: list of 2-card hands for players still in showdown.
    ``board``: up to 5 community cards (unknowns filtered out).
    """
    if not holes:
        return []
    ranks = [_rank7(h, board) for h in holes]
    best = min(ranks)
    return [i for i, r in enumerate(ranks) if r == best]
