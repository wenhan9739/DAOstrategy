"""Lightweight single-hand 6-max NLHE environment for RL.

We support a *vectorized* wrapper that drives N independent hands at once.
Between hero decisions, opponents are advanced automatically (via an
injected ``opponent_fn`` that takes a batch of observations and returns
action buckets).

Conventions:
  - Stacks in BB units (float), same as BC training.
  - Seats are 0..5 in canonical position order (0=SB, 1=BB, ..., 5=BTN).
  - Hero seat is configurable per hand (rotates).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..data.features import HIST_LEN, MAX_PLAYERS
from ..data.replay import ActionToken
from ..env.actions import (
    ALL_IN,
    CHECK_CALL,
    FOLD,
    NUM_ACTIONS,
    RAISE_POT_FRACTIONS,
    RAISE_START,
    bucketize_raise,
    desnap_raise_to,
)
from ..env.cards import UNKNOWN_CARD
from ..env.equity import pick_winners


@dataclass
class _Player:
    pos: int
    stack: float
    committed_street: float = 0.0
    committed_total: float = 0.0
    folded: bool = False
    all_in: bool = False
    hole: list[int] = field(default_factory=list)
    # NLHE rule: a betting round ends only when every still-able-to-act player
    # has acted at least once since the last raise AND matched the bet.
    # Posting SB/BB counts as "putting money in", NOT as "acting" — BB still
    # gets the option preflop, and the first post-flop check does not kill
    # the street.
    has_acted_this_street: bool = False


@dataclass
class HandState:
    players: list[_Player]
    board: list[int]
    pot: float
    street: int                # 0..3 (pre/flop/turn/river) or 4 = done
    current_bet: float
    min_raise: float
    to_act: int                # seat index of next player to act, or -1
    last_raiser: int
    history: list[ActionToken]
    hero_seat: int
    deck: list[int]
    initial_stacks: list[float]
    done: bool = False
    winner_seats: list[int] = field(default_factory=list)


def _build_deck(rng: random.Random) -> list[int]:
    d = list(range(52))
    rng.shuffle(d)
    return d


def _active_seats(s: HandState) -> list[int]:
    return [p.pos for p in s.players if not p.folded]


def _alive_for_action(s: HandState) -> list[int]:
    return [p.pos for p in s.players if not p.folded and not p.all_in]


def _start_hand(
    n_players: int,
    starting_stacks: list[float],
    sb: float,
    bb: float,
    hero_seat: int,
    rng: random.Random,
) -> HandState:
    assert n_players == MAX_PLAYERS, "fixed 6-max for now"
    players = [_Player(pos=i, stack=starting_stacks[i]) for i in range(n_players)]
    deck = _build_deck(rng)
    # deal hole cards
    for p in players:
        p.hole = [deck.pop(), deck.pop()]
    # post SB at seat 0, BB at seat 1. Posting blinds is NOT "acting" — SB still
    # needs to decide, BB gets option to check/raise when action returns.
    players[0].stack -= sb
    players[0].committed_street += sb
    players[0].committed_total += sb
    players[1].stack -= bb
    players[1].committed_street += bb
    players[1].committed_total += bb
    pot = sb + bb
    # first to act preflop in 6-max is UTG (seat 2 in our canonical ordering).
    to_act = 2 % n_players
    s = HandState(
        players=players,
        board=[],
        pot=pot,
        street=0,
        current_bet=bb,
        min_raise=bb,
        to_act=to_act,
        # No raise yet; BB's big-blind post is tracked by current_bet, not by
        # last_raiser. Round-done logic uses has_acted_this_street, so BB
        # automatically gets the option.
        last_raiser=-1,
        history=[],
        hero_seat=hero_seat,
        deck=deck,
        initial_stacks=list(starting_stacks),
    )
    return s


def _end_street(s: HandState) -> None:
    for p in s.players:
        p.committed_street = 0.0
        p.has_acted_this_street = False  # everyone owes one action on the new street
    s.current_bet = 0.0
    s.min_raise = 1.0
    s.last_raiser = -1
    # first to act postflop: first non-folded, non-all-in seat starting from 0 (SB)
    active = [p.pos for p in s.players if not p.folded and not p.all_in]
    s.to_act = active[0] if active else -1


def _advance_street(s: HandState) -> None:
    if s.street == 0:
        s.board.extend([s.deck.pop(), s.deck.pop(), s.deck.pop()])
        s.street = 1
    elif s.street == 1:
        s.board.append(s.deck.pop())
        s.street = 2
    elif s.street == 2:
        s.board.append(s.deck.pop())
        s.street = 3
    else:
        s.street = 4  # showdown
    _end_street(s)


def _all_remaining_all_in(s: HandState) -> bool:
    non_folded = [p for p in s.players if not p.folded]
    if len(non_folded) <= 1:
        return True
    can_act = [p for p in non_folded if not p.all_in]
    return len(can_act) <= 1


def _check_round_done(s: HandState) -> bool:
    """Return True if the current betting round is complete.

    Correct NLHE rule: a round ends when every player who can still act has
    (1) matched the current bet AND (2) acted at least once since the most
    recent raise. Posting SB/BB preflop does NOT count as acting.
    """
    can_act = [p for p in s.players if not p.folded and not p.all_in]
    if not can_act:
        return True
    for p in can_act:
        if abs(p.committed_street - s.current_bet) > 1e-9:
            return False
        if not p.has_acted_this_street:
            return False
    return True


def _settle(s: HandState) -> None:
    """Award the pot. Handles side pots."""
    non_folded = [p for p in s.players if not p.folded]
    n = len(s.players)

    if len(non_folded) == 1:
        w = non_folded[0]
        w.stack += s.pot
        s.winner_seats = [w.pos]
        s.done = True
        s.street = 4
        return

    # Need showdown with side pots based on committed_total.
    # Fast-forward any remaining streets if all-in.
    while s.street < 3:
        if s.street == 0:
            s.board.extend([s.deck.pop(), s.deck.pop(), s.deck.pop()])
            s.street = 1
        elif s.street == 1:
            s.board.append(s.deck.pop())
            s.street = 2
        elif s.street == 2:
            s.board.append(s.deck.pop())
            s.street = 3
    # now build pots by contribution level
    contribs = sorted({p.committed_total for p in s.players if p.committed_total > 0})
    prev = 0.0
    winners_flag = [False] * n
    for lvl in contribs:
        layer_contributors = [p for p in s.players if p.committed_total >= lvl]
        if not layer_contributors:
            prev = lvl
            continue
        layer_amt = (lvl - prev) * len(layer_contributors)
        layer_elig = [p for p in layer_contributors if not p.folded]
        prev = lvl
        if not layer_elig:
            continue
        holes = [p.hole for p in layer_elig]
        winners_idx = pick_winners(holes, s.board[:5])
        share = layer_amt / len(winners_idx)
        for wi in winners_idx:
            layer_elig[wi].stack += share
            winners_flag[layer_elig[wi].pos] = True
    s.winner_seats = [i for i, f in enumerate(winners_flag) if f]
    s.done = True
    s.street = 4


def _next_to_act(s: HandState) -> None:
    n = len(s.players)
    cur = s.to_act
    for _ in range(n):
        cur = (cur + 1) % n
        p = s.players[cur]
        if not p.folded and not p.all_in:
            s.to_act = cur
            return
    s.to_act = -1


def apply_action(s: HandState, bucket: int) -> None:
    """Mutate state by applying bucketed action from the current to-act player."""
    if s.done or s.to_act < 0:
        return
    p = s.players[s.to_act]
    to_call = max(0.0, s.current_bet - p.committed_street)
    pot_before_call = s.pot
    stack_rem = p.stack
    pos = p.pos
    street = s.street

    # clamp illegal actions
    if bucket == FOLD and to_call <= 0:
        bucket = CHECK_CALL   # can't fold if free

    if bucket == FOLD:
        p.folded = True
        p.has_acted_this_street = True
        s.history.append(ActionToken(actor=pos, actor_pos=pos, street=street,
                                     bucket=FOLD, raise_frac=0.0, to_chips=0.0))
    elif bucket == CHECK_CALL:
        pay = min(to_call, stack_rem)
        p.stack -= pay
        p.committed_street += pay
        p.committed_total += pay
        s.pot += pay
        if p.stack <= 1e-9:
            p.all_in = True
        p.has_acted_this_street = True
        s.history.append(ActionToken(actor=pos, actor_pos=pos, street=street,
                                     bucket=CHECK_CALL, raise_frac=0.0,
                                     to_chips=p.committed_street))
    else:
        # raise bucket (2..9)
        min_raise_to = s.current_bet + s.min_raise
        target_to = desnap_raise_to(
            bucket=bucket,
            pot_before_call=pot_before_call,
            to_call=to_call,
            stack_remaining=stack_rem,
            min_raise_to=min_raise_to,
        )
        pay = min(target_to - p.committed_street, stack_rem)
        if pay <= to_call:
            # below min raise becomes a call (protect env)
            pay = min(to_call, stack_rem)
            p.stack -= pay
            p.committed_street += pay
            p.committed_total += pay
            s.pot += pay
            if p.stack <= 1e-9:
                p.all_in = True
            p.has_acted_this_street = True
            s.history.append(ActionToken(actor=pos, actor_pos=pos, street=street,
                                         bucket=CHECK_CALL, raise_frac=0.0,
                                         to_chips=p.committed_street))
        else:
            p.stack -= pay
            p.committed_street += pay
            p.committed_total += pay
            s.pot += pay
            raise_inc = p.committed_street - s.current_bet
            if raise_inc > s.min_raise - 1e-9:
                s.min_raise = raise_inc
            s.current_bet = max(s.current_bet, p.committed_street)
            s.last_raiser = pos
            if p.stack <= 1e-9:
                p.all_in = True
            # A raise re-opens the action for everyone else who can still act.
            # (Note: a short all-in below min-raise technically should not re-open
            # for players who already acted fully; we keep the simplification of
            # re-opening to keep logic uniform.)
            p.has_acted_this_street = True
            for q in s.players:
                if q.pos != pos and not q.folded and not q.all_in:
                    q.has_acted_this_street = False
            rfrac = RAISE_POT_FRACTIONS[min(bucket - RAISE_START, len(RAISE_POT_FRACTIONS) - 1)] \
                if bucket < ALL_IN else 0.0
            s.history.append(ActionToken(actor=pos, actor_pos=pos, street=street,
                                         bucket=bucket if not p.all_in else ALL_IN,
                                         raise_frac=rfrac, to_chips=p.committed_street))

    # end-of-hand check: only one non-folded left
    non_folded = [q for q in s.players if not q.folded]
    if len(non_folded) == 1:
        _settle(s)
        return

    # check betting round end
    if _check_round_done(s):
        if s.street >= 3 or _all_remaining_all_in(s):
            _settle(s)
            return
        _advance_street(s)
    else:
        _next_to_act(s)


def legal_action_mask(s: HandState) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    if s.done or s.to_act < 0:
        return mask
    p = s.players[s.to_act]
    to_call = max(0.0, s.current_bet - p.committed_street)
    # fold legal only if there's something to call
    if to_call > 0:
        mask[FOLD] = True
    mask[CHECK_CALL] = True
    min_raise_to = s.current_bet + s.min_raise
    can_raise_over_call = (p.stack + p.committed_street) > s.current_bet + 1e-9
    # bucket raises require enough stack to at least min-raise
    if can_raise_over_call and p.stack > to_call + 1e-9:
        for b in range(RAISE_START, ALL_IN):
            # only include bucket if its de-snapped raise_to >= min_raise_to AND <= all-in
            target = desnap_raise_to(b, s.pot, to_call, p.stack, min_raise_to)
            over_call = target - to_call
            if over_call > 1e-6 and target < to_call + p.stack - 1e-9:
                mask[b] = True
        mask[ALL_IN] = True
    return mask


def get_observation(s: HandState, viewer_seat: int) -> dict:
    """Return observation from viewer's perspective (numpy arrays).

    If viewer_seat equals s.to_act this is a standard decision observation.
    """
    p = s.players[viewer_seat]
    opp_stacks = [q.stack for q in s.players if q.pos != viewer_seat and not q.folded]
    eff_stack = min(p.stack, min(opp_stacks) if opp_stacks else p.stack)
    to_call = max(0.0, s.current_bet - p.committed_street)
    active_mask = np.zeros(MAX_PLAYERS, dtype=np.bool_)
    for q in s.players:
        if not q.folded:
            active_mask[q.pos] = True
    board_arr = np.full(5, UNKNOWN_CARD, dtype=np.uint8)
    for i, c in enumerate(s.board[:5]):
        board_arr[i] = c
    hole_arr = np.full(2, UNKNOWN_CARD, dtype=np.uint8)
    if viewer_seat == s.to_act or not p.folded:
        for i, c in enumerate(p.hole[:2]):
            hole_arr[i] = c

    # history (player actions only, last HIST_LEN)
    hist_actor = np.full(HIST_LEN, -1, dtype=np.int8)
    hist_street = np.zeros(HIST_LEN, dtype=np.int8)
    hist_bucket = np.full(HIST_LEN, -1, dtype=np.int8)
    hist_rfrac = np.zeros(HIST_LEN, dtype=np.float16)
    toks = [t for t in s.history if t.actor >= 0 and t.bucket >= 0]
    if len(toks) > HIST_LEN:
        toks = toks[-HIST_LEN:]
    for i, t in enumerate(toks):
        hist_actor[i] = t.actor_pos
        hist_street[i] = t.street
        hist_bucket[i] = t.bucket
        hist_rfrac[i] = float(t.raise_frac)

    num_players = len(s.players)
    num_active = sum(1 for q in s.players if not q.folded)
    ctx_float = np.array([
        np.log1p(s.pot),
        np.log1p(to_call),
        np.log1p(p.stack),
        np.log1p(eff_stack),
        to_call / max(s.pot, 1e-3),
        p.stack / max(s.pot, 1e-3),
        num_active / MAX_PLAYERS,
    ], dtype=np.float32)
    street_clamped = min(3, max(0, s.street))  # terminal state may set 4; clamp for embedding
    ctx_int = np.array([street_clamped, viewer_seat, num_players], dtype=np.int8)
    return {
        "ctx_float": ctx_float,
        "ctx_int": ctx_int,
        "board": board_arr,
        "hole": hole_arr,
        "active": active_mask,
        "hist_actor": hist_actor,
        "hist_street": hist_street,
        "hist_bucket": hist_bucket,
        "hist_rfrac": hist_rfrac,
    }


class PokerVecEnv:
    """Vectorized 6-max NLHE with a single hero seat and opponents driven by ``opponent_fn``.

    Usage:
        env = PokerVecEnv(n_envs=256, opponent_fn=bc_callable)
        obs, mask = env.reset()          # obs is batch dict, mask is (B, NUM_ACTIONS)
        for step in range(T):
            hero_actions = hero_policy(obs, mask)
            obs, reward, done, mask, info = env.step(hero_actions)
    """

    def __init__(
        self,
        n_envs: int,
        opponent_fn: Callable[[list[dict]], list[int]],
        start_stack_bb: float = 100.0,
        sb_bb: float = 0.5,
        bb_bb: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.n_envs = n_envs
        self.opponent_fn = opponent_fn
        self.start_stack = start_stack_bb
        self.sb = sb_bb
        self.bb = bb_bb
        self.rng = random.Random(seed)
        self.states: list[HandState] = []
        self.hand_count = 0

    def _new_hand(self) -> HandState:
        hero_seat = self.rng.randrange(MAX_PLAYERS)
        stacks = [self.start_stack] * MAX_PLAYERS
        s = _start_hand(MAX_PLAYERS, stacks, self.sb, self.bb, hero_seat, self.rng)
        return s

    def reset(self) -> tuple[dict, np.ndarray]:
        self.states = [self._new_hand() for _ in range(self.n_envs)]
        self._advance_until_hero_or_done()
        return self._observe()

    def _advance_until_hero_or_done(self) -> None:
        """For each env, advance opponents until it's hero's turn or hand ends."""
        # Safety: a single hand has at most ~60 opponent actions even in worst case;
        # 2000 iters is a hard ceiling to break any deadlock caused by pathological state.
        max_iters = 2000
        for _ in range(max_iters):
            opp_idx = [i for i, s in enumerate(self.states)
                       if not s.done and s.to_act >= 0 and s.to_act != s.hero_seat]
            if not opp_idx:
                return
            obs_batch = [get_observation(self.states[i], self.states[i].to_act) for i in opp_idx]
            actions = self.opponent_fn(obs_batch)
            for i, a in zip(opp_idx, actions):
                s = self.states[i]
                mask = legal_action_mask(s)
                if not mask[a]:
                    a = CHECK_CALL if mask[CHECK_CALL] else int(np.flatnonzero(mask)[0])
                apply_action(s, a)
        # Deadlock fallback: forcibly mark stuck envs as done with zero reward.
        import sys
        stuck = [i for i, s in enumerate(self.states)
                 if not s.done and s.to_act != s.hero_seat]
        print(f"[WARN] _advance_until_hero_or_done hit {max_iters}-iter cap; "
              f"force-resetting {len(stuck)} stuck env(s)", flush=True, file=sys.stderr)
        for i in stuck:
            s = self.states[i]
            s.done = True
            s.hero_reward_chips = 0.0

    def _observe(self) -> tuple[dict, np.ndarray]:
        """Build hero-perspective observation batch for all envs. Terminal envs get dummy obs."""
        obs_list: list[dict] = []
        masks: list[np.ndarray] = []
        for s in self.states:
            if s.done or s.to_act != s.hero_seat:
                # terminal or anomaly; return placeholder obs
                obs_list.append(get_observation(s, s.hero_seat))
                masks.append(np.ones(NUM_ACTIONS, dtype=np.bool_))
            else:
                obs_list.append(get_observation(s, s.hero_seat))
                masks.append(legal_action_mask(s))
        # batch stack
        batch = {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}
        return batch, np.stack(masks)

    def step(self, hero_actions: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, dict]:
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        dones = np.zeros(self.n_envs, dtype=np.bool_)
        for i, a in enumerate(hero_actions):
            s = self.states[i]
            if s.done:
                dones[i] = True
                continue
            if s.to_act != s.hero_seat:
                # shouldn't happen — still advance opponents (safety net)
                continue
            mask = legal_action_mask(s)
            if not mask[a]:
                a = CHECK_CALL if mask[CHECK_CALL] else int(np.flatnonzero(mask)[0])
            apply_action(s, int(a))

        # advance opponents
        self._advance_until_hero_or_done()

        # compute rewards for envs that ended; reset them
        for i, s in enumerate(self.states):
            if s.done:
                reward = s.players[s.hero_seat].stack - s.initial_stacks[s.hero_seat]
                rewards[i] = reward
                dones[i] = True
                self.hand_count += 1
                self.states[i] = self._new_hand()
        if dones.any():
            self._advance_until_hero_or_done()

        obs, mask = self._observe()
        return obs, rewards, dones, mask, {"hands": self.hand_count}
