"""Card utilities.

Cards are encoded as integers 0..51:
  rank  = card // 4   in 0..12  (2,3,4,5,6,7,8,9,T,J,Q,K,A)
  suit  = card  % 4   in 0..3   (c,d,h,s)

String form uses the standard PHH format, e.g. ``'Ts'``, ``'Ac'``.
"""
from __future__ import annotations

from typing import Iterable

RANKS = "23456789TJQKA"
SUITS = "cdhs"
RANK_TO_IDX = {r: i for i, r in enumerate(RANKS)}
SUIT_TO_IDX = {s: i for i, s in enumerate(SUITS)}

UNKNOWN_CARD = 52  # padding / "????" sentinel for tensor embedding


def card_to_int(s: str) -> int:
    """Convert two-char card string (e.g. 'Ts') to int in [0, 51]."""
    if len(s) != 2:
        raise ValueError(f"bad card string: {s!r}")
    r, u = s[0], s[1].lower()
    if r not in RANK_TO_IDX or u not in SUIT_TO_IDX:
        raise ValueError(f"bad card string: {s!r}")
    return RANK_TO_IDX[r] * 4 + SUIT_TO_IDX[u]


def int_to_card(c: int) -> str:
    if not (0 <= c < 52):
        return "??"
    return RANKS[c // 4] + SUITS[c % 4]


def parse_cards(token: str) -> list[int]:
    """Parse contiguous card string like '7d5h9d' or 'As' or '????'.

    Returns list of ints; unknown cards become ``UNKNOWN_CARD``.
    """
    token = token.strip()
    if not token:
        return []
    out = []
    i = 0
    while i < len(token):
        pair = token[i : i + 2]
        if pair == "??":
            out.append(UNKNOWN_CARD)
        else:
            out.append(card_to_int(pair))
        i += 2
    return out


def cards_to_str(cards: Iterable[int]) -> str:
    return "".join(int_to_card(c) for c in cards)
