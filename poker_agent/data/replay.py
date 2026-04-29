"""Replay a HandRecord into a sequence of (state, action) training samples.

For BC we want, at every acting decision point of a human/player, a snapshot of:
  * public info visible to all players
  * the acting player's own hole cards (if known, else UNKNOWN)
  * the discrete action bucket they chose
  * the normalized raise size (if raise), else 0

We walk the action list with a minimal NLHE state machine. Only NT variant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional

from ..env.actions import (
    ALL_IN,
    CHECK_CALL,
    FOLD,
    NUM_ACTIONS,
    RAISE_START,
    bucketize_raise,
)
from ..env.cards import UNKNOWN_CARD, parse_cards
from .parser import HandRecord

MAX_PLAYERS = 6

STREET_PRE = 0
STREET_FLOP = 1
STREET_TURN = 2
STREET_RIVER = 3

# Position names assuming first two entries are SB, BB, then UTG,...,BTN.
# For HU: p1=BTN/SB, p2=BB.
def _assign_positions(n: int) -> list[int]:
    """Return position id (0..5) for p1..pn.

    We always map to a 6-max frame of reference where:
      0 = SB, 1 = BB, 2 = UTG, 3 = HJ, 4 = CO, 5 = BTN.
    For <6 players, missing positions are collapsed from the UTG side.
    """
    if n == 2:
        return [5, 1]  # BTN (=SB in HU), BB
    if n == 3:
        return [0, 1, 5]  # SB, BB, BTN
    if n == 4:
        return [0, 1, 4, 5]
    if n == 5:
        return [0, 1, 3, 4, 5]
    if n == 6:
        return [0, 1, 2, 3, 4, 5]
    # fallback: truncate
    return list(range(n))


@dataclass
class ActionToken:
    """One action event in history, including dealer events."""

    actor: int        # 0..n-1 player idx; -1 for dealer
    actor_pos: int    # canonical position 0..5; -1 for dealer
    street: int       # 0..3
    bucket: int       # action bucket (for player actions); -1 for dealer events
    raise_frac: float # normalized bet size (fraction of pot); 0 otherwise
    to_chips: float   # absolute chips for player bets (for debug)


@dataclass
class DecisionSample:
    """A single (state, action) supervised example for BC."""

    hand_id: str
    street: int
    actor_idx: int         # 0..n-1 player index
    actor_pos: int         # 0..5 canonical position
    num_players: int
    board: list[int]        # up to 5 cards; pad with UNKNOWN_CARD
    hole: list[int]         # 2 cards (may be UNKNOWN)
    pot_bb: float
    to_call_bb: float
    stack_bb: float         # acting player's remaining stack
    effective_stack_bb: float
    num_active: int
    action_history: list[ActionToken]  # all tokens up to (not including) this action
    active_mask: list[bool] # size 6 (pad with False): whether each position is still alive
    bucket: int             # chosen action bucket
    raise_frac: float       # 0 if fold/check/call; else pot-fraction


@dataclass
class _PlayerState:
    idx: int
    pos: int                  # canonical position 0..5
    stack: float              # remaining chips
    committed_street: float   # chips put in this street
    committed_total: float    # chips put in total
    folded: bool = False
    all_in: bool = False
    hole: list[int] = field(default_factory=list)


def replay_hand(hand: HandRecord, bb: float | None = None) -> Iterator[DecisionSample]:
    """Yield DecisionSample per player decision. Units are BB."""
    n = hand.num_players
    if bb is None:
        bb = max(hand.blinds) if hand.blinds and max(hand.blinds) > 0 else 1.0
    positions = _assign_positions(n)
    players = [
        _PlayerState(idx=i, pos=positions[i], stack=hand.starting_stacks[i] / bb,
                     committed_street=0.0, committed_total=0.0)
        for i in range(n)
    ]
    # post antes
    for i, ante in enumerate(hand.antes):
        if i < n and ante > 0:
            a = ante / bb
            a = min(a, players[i].stack)
            players[i].stack -= a
            players[i].committed_total += a  # antes do NOT count towards current street bet
            if players[i].stack <= 1e-9:
                players[i].all_in = True
    # post blinds
    for i, b in enumerate(hand.blinds):
        if i < n and b > 0:
            amt = b / bb
            amt = min(amt, players[i].stack)
            players[i].stack -= amt
            players[i].committed_street += amt
            players[i].committed_total += amt
            if players[i].stack <= 1e-9:
                players[i].all_in = True
    current_bet = max((p.committed_street for p in players), default=0.0)
    min_raise = max(hand.min_bet / bb, 1.0)  # chip size of smallest raise increment

    pot = sum(p.committed_total for p in players)
    board: list[int] = []
    street = STREET_PRE
    history: list[ActionToken] = []
    hand_id = f"{hand.src_file}#{hand.src_idx}"

    def num_active() -> int:
        return sum(1 for p in players if not p.folded)

    def next_street():
        nonlocal current_bet, min_raise
        for p in players:
            p.committed_street = 0.0
        current_bet = 0.0
        min_raise = 1.0

    for raw in hand.actions:
        parts = raw.split()
        if not parts:
            continue
        actor_str = parts[0]

        if actor_str == "d":
            op = parts[1]
            if op == "dh":
                # deal hole cards: 'd dh p3 Ts7s' or 'd dh p3 ????'
                pid = int(parts[2][1:]) - 1
                cards = parse_cards(parts[3]) if len(parts) > 3 else [UNKNOWN_CARD, UNKNOWN_CARD]
                # pad
                while len(cards) < 2:
                    cards.append(UNKNOWN_CARD)
                players[pid].hole = cards[:2]
                history.append(ActionToken(actor=-1, actor_pos=-1, street=street, bucket=-1, raise_frac=0.0, to_chips=0.0))
            elif op == "db":
                # deal board
                new_cards = parse_cards(parts[2])
                board.extend(new_cards)
                if len(board) == 3:
                    next_street()
                    street = STREET_FLOP
                elif len(board) == 4:
                    next_street()
                    street = STREET_TURN
                elif len(board) == 5:
                    next_street()
                    street = STREET_RIVER
                history.append(ActionToken(actor=-1, actor_pos=-1, street=street, bucket=-1, raise_frac=0.0, to_chips=0.0))
            else:
                continue  # ignore unknown dealer ops (e.g. 'd sd')
            continue

        # player action
        if not actor_str.startswith("p"):
            continue
        pid = int(actor_str[1:]) - 1
        if pid < 0 or pid >= n:
            continue
        p = players[pid]
        op = parts[1] if len(parts) > 1 else ""

        if op == "sm":  # showdown; reveal cards but no more betting action
            cards = parse_cards(parts[2]) if len(parts) > 2 else []
            while len(cards) < 2:
                cards.append(UNKNOWN_CARD)
            # Overwrite hole with revealed cards if known
            if cards[0] != UNKNOWN_CARD:
                p.hole = cards[:2]
            continue

        to_call = max(0.0, current_bet - p.committed_street)
        pot_before_call = pot  # current total pot
        stack_remaining = p.stack

        # Determine bucket & raise fraction + apply state transition.
        bucket = -1
        raise_frac = 0.0
        target_to = 0.0
        if op == "f":
            bucket = FOLD
        elif op == "cc":
            if to_call > stack_remaining + 1e-9:
                # all-in call
                bucket = CHECK_CALL  # still model as check/call; env caps at stack
            else:
                bucket = CHECK_CALL
        elif op == "cbr":
            try:
                target_to = float(parts[2]) / bb
            except Exception:
                continue
            bucket = bucketize_raise(
                raise_to_chips=target_to,
                pot_before_call=pot_before_call,
                to_call=to_call,
                stack_remaining=stack_remaining,
            )
            post_call_pot = pot_before_call + to_call
            raise_over_call = max(0.0, target_to - to_call)
            raise_frac = raise_over_call / post_call_pot if post_call_pot > 1e-9 else 0.0
        else:
            continue  # unknown op

        # ---- emit sample (before applying) ----
        board_padded = (board + [UNKNOWN_CARD] * 5)[:5]
        hole_padded = (p.hole + [UNKNOWN_CARD] * 2)[:2]
        effective_stack = min(p.stack, max((q.stack for q in players if not q.folded and q.idx != p.idx), default=p.stack))
        active_mask = [False] * MAX_PLAYERS
        for q in players:
            if not q.folded:
                active_mask[q.pos] = True
        yield DecisionSample(
            hand_id=hand_id,
            street=street,
            actor_idx=pid,
            actor_pos=p.pos,
            num_players=n,
            board=board_padded,
            hole=hole_padded,
            pot_bb=pot,
            to_call_bb=to_call,
            stack_bb=stack_remaining,
            effective_stack_bb=effective_stack,
            num_active=num_active(),
            action_history=list(history),
            active_mask=active_mask,
            bucket=bucket,
            raise_frac=raise_frac,
        )

        # ---- apply action ----
        if bucket == FOLD:
            p.folded = True
            history.append(ActionToken(actor=pid, actor_pos=p.pos, street=street,
                                       bucket=FOLD, raise_frac=0.0, to_chips=0.0))
        elif bucket == CHECK_CALL:
            pay = min(to_call, stack_remaining)
            p.stack -= pay
            p.committed_street += pay
            p.committed_total += pay
            pot += pay
            if p.stack <= 1e-9:
                p.all_in = True
            history.append(ActionToken(actor=pid, actor_pos=p.pos, street=street,
                                       bucket=CHECK_CALL, raise_frac=0.0,
                                       to_chips=p.committed_street))
        else:
            # raise: pay up to target_to total for this street
            pay = min(target_to - p.committed_street, stack_remaining)
            if pay < 0:
                pay = 0
            p.stack -= pay
            p.committed_street += pay
            p.committed_total += pay
            pot += pay
            if p.stack <= 1e-9:
                p.all_in = True
            new_bet = p.committed_street
            raise_inc = new_bet - current_bet
            if raise_inc > min_raise - 1e-9:
                min_raise = raise_inc
            current_bet = max(current_bet, new_bet)
            history.append(ActionToken(actor=pid, actor_pos=p.pos, street=street,
                                       bucket=bucket if not p.all_in else ALL_IN,
                                       raise_frac=raise_frac, to_chips=new_bet))
