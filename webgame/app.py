"""Flask web UI: play 6-max NLHE as HERO against 5 BC bots.

Run:
    python -m webgame.app --bc-ckpt checkpoints/bc_full/best.pt --host 127.0.0.1 --port 5000

Then open http://127.0.0.1:5000 in a browser.

State lives in-memory in a single global table — this is a local dev tool,
not a multi-user server.
"""
from __future__ import annotations

import argparse
import random
import sys
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request

# Make repo root importable when run as "python -m webgame.app"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from poker_agent.data.features import MAX_PLAYERS  # noqa: E402
from poker_agent.env.actions import (  # noqa: E402
    ACTION_NAMES,
    ALL_IN,
    CHECK_CALL,
    FOLD,
    RAISE_POT_FRACTIONS,
    RAISE_START,
    bucketize_raise,
)
from poker_agent.env.cards import int_to_card  # noqa: E402
from poker_agent.env.holdem import (  # noqa: E402
    _start_hand,
    apply_action,
    get_observation,
    legal_action_mask,
)
from poker_agent.models.ppo import load_bc_checkpoint, masked_logits  # noqa: E402
from poker_agent.training.train_ppo import obs_to_tensor  # noqa: E402


POS_NAMES = ["SB", "BB", "UTG", "HJ", "CO", "BTN"]
START_STACK_BB = 100.0
SB_AMT = 0.5
BB_AMT = 1.0
SB_SEAT = 0
BB_SEAT = 1
BTN_SEAT = 5


# ----------------------------------------------------------------------------
# Global game state
# ----------------------------------------------------------------------------
class Game:
    """Single-table 6-max hand manager. Human always plays one rotating seat."""

    def __init__(self, bc_ckpt: str, device: str = "cpu", seed: int | None = None):
        self.device = torch.device(device)
        self.model = load_bc_checkpoint(bc_ckpt, map_location=str(self.device))
        self.model.to(self.device)
        self.model.eval()
        self.rng = random.Random(seed if seed is not None else 0xC0FFEE)
        self.hand_idx = 0
        self.state = None            # HandState or None
        self.hero_seat = 0
        self.stacks = [START_STACK_BB] * MAX_PLAYERS
        self.log: list[str] = []     # text log for the current hand
        self.lock = threading.Lock()
        # Cached at hand end so UI can replay showdown.
        self.last_settle = None      # dict: winners, revealed_holes, pnl
        # Session-wide stats (persist across new_hand calls).
        self.session = {
            "hands": 0,
            "pnl": 0.0,
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "best": 0.0,
            "worst": 0.0,
        }
        # Debug: when True, reveal all bot hole cards even mid-hand.
        # Default on — the task has full visibility for study/debug purposes.
        self.reveal_all = True

    # ---------------- hand lifecycle ----------------
    def new_hand(self) -> None:
        # rotate hero seat each hand for variety
        self.hero_seat = self.hand_idx % MAX_PLAYERS
        # reset stacks to 100bb every hand (cash-game-style, no progressive)
        self.stacks = [START_STACK_BB] * MAX_PLAYERS
        self.state = _start_hand(
            MAX_PLAYERS,
            self.stacks,
            SB_AMT,
            BB_AMT,
            hero_seat=self.hero_seat,
            rng=self.rng,
        )
        self.hand_idx += 1
        self.log = [
            f"=== Hand #{self.hand_idx} — HERO seat {self.hero_seat} "
            f"({POS_NAMES[self.hero_seat]}) ==="
        ]
        self.log.append(
            f"Blinds: SB {SB_AMT} / BB {BB_AMT}. Starting stacks 100bb."
        )
        self.last_settle = None
        self._advance_opponents()

    @torch.no_grad()
    def _bot_action(self, seat: int) -> int:
        obs = get_observation(self.state, seat)
        mask = legal_action_mask(self.state)
        batch = {k: np.stack([obs[k]]) for k in obs}
        tb = obs_to_tensor(batch, self.device)
        out = self.model(tb)
        logits = out["policy_logits"].float()
        mt = torch.from_numpy(np.stack([mask])).to(self.device)
        ml = masked_logits(logits, mt)
        probs = F.softmax(ml, dim=-1)
        a = int(torch.multinomial(probs, num_samples=1).item())
        if not mask[a]:
            a = CHECK_CALL if mask[CHECK_CALL] else int(np.flatnonzero(mask)[0])
        return a

    def _advance_opponents(self) -> None:
        """Advance bot actions until it's human's turn or hand ends."""
        guard = 0
        while (
            self.state is not None
            and not self.state.done
            and self.state.to_act >= 0
            and self.state.to_act != self.hero_seat
        ):
            guard += 1
            if guard > 500:
                # Defensive: should never hit.
                break
            seat = self.state.to_act
            a = self._bot_action(seat)
            self._record_action(seat, a)
            apply_action(self.state, a)

        if self.state is not None and self.state.done:
            self._finalize_hand()

    def human_act(self, bucket: int, raise_to_bb: float | None = None) -> None:
        """Apply human action. If raise_to_bb given, bucketize; otherwise use bucket."""
        if self.state is None or self.state.done:
            return
        if self.state.to_act != self.hero_seat:
            return
        mask = legal_action_mask(self.state)

        if raise_to_bb is not None and bucket >= RAISE_START and bucket != ALL_IN:
            # Snap user's "raise to" to the closest legal bucket.
            p = self.state.players[self.hero_seat]
            to_call = max(0.0, self.state.current_bet - p.committed_street)
            snapped = bucketize_raise(
                raise_to_chips=float(raise_to_bb),
                pot_before_call=self.state.pot,
                to_call=to_call,
                stack_remaining=p.stack,
            )
            bucket = snapped

        if not mask[bucket]:
            # Clamp to a legal fallback.
            bucket = CHECK_CALL if mask[CHECK_CALL] else int(np.flatnonzero(mask)[0])

        self._record_action(self.hero_seat, bucket)
        apply_action(self.state, bucket)
        self._advance_opponents()

    # ---------------- helpers ----------------
    def _record_action(self, seat: int, bucket: int) -> None:
        s = self.state
        p = s.players[seat]
        to_call_before = max(0.0, s.current_bet - p.committed_street)
        stack_before = p.stack
        pot_before = s.pot
        street_name = ("Preflop", "Flop", "Turn", "River")[min(s.street, 3)]
        who = POS_NAMES[seat] + (" [YOU]" if seat == self.hero_seat else "")

        if bucket == FOLD:
            label = "folds"
        elif bucket == CHECK_CALL:
            if to_call_before <= 1e-9:
                label = "checks"
            else:
                pay = min(to_call_before, stack_before)
                label = f"calls {pay:.2f}"
        elif bucket == ALL_IN:
            label = f"ALL-IN ({stack_before:.2f})"
        else:
            frac = RAISE_POT_FRACTIONS[bucket - RAISE_START]
            label = f"raises ({frac:g}x pot)"

        self.log.append(f"[{street_name}] {who}: {label} (pot {pot_before:.1f})")

    def _finalize_hand(self) -> None:
        s = self.state
        hero = s.players[self.hero_seat]
        pnl = hero.stack - s.initial_stacks[self.hero_seat]
        non_folded = [p for p in s.players if not p.folded]
        revealed: dict[int, list[str]] = {}
        # Reveal all contested holes if multiple players reached showdown (i.e. board got to river).
        if len(non_folded) >= 2:
            for p in non_folded:
                revealed[p.pos] = [int_to_card(c) for c in p.hole]
        # Always reveal hero hole for reference.
        revealed[self.hero_seat] = [int_to_card(c) for c in hero.hole]

        winners = [POS_NAMES[i] for i in s.winner_seats]
        self.log.append(
            f"Winners: {', '.join(winners) if winners else '(none)'}."
        )
        if revealed:
            lines = []
            for pos_i in sorted(revealed):
                lines.append(f"{POS_NAMES[pos_i]}={' '.join(revealed[pos_i])}")
            self.log.append("Revealed: " + "  |  ".join(lines))
        self.log.append(f"Final board: {' '.join(int_to_card(c) for c in s.board)}")
        self.log.append(f"HERO PnL: {pnl:+.2f} bb")

        self.last_settle = {
            "winners": s.winner_seats,
            "revealed": revealed,
            "pnl": pnl,
            "board": [int_to_card(c) for c in s.board],
            "went_to_showdown": len(non_folded) >= 2,
        }

        self.session["hands"] += 1
        self.session["pnl"] += pnl
        if pnl > 1e-6:
            self.session["wins"] += 1
        elif pnl < -1e-6:
            self.session["losses"] += 1
        else:
            self.session["ties"] += 1
        self.session["best"] = max(self.session["best"], pnl)
        self.session["worst"] = min(self.session["worst"], pnl)

    def reset_session(self) -> None:
        self.session = {
            "hands": 0,
            "pnl": 0.0,
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "best": 0.0,
            "worst": 0.0,
        }

    # ---------------- serialization ----------------
    def as_dict(self) -> dict:
        s = self.state
        if s is None:
            return {"started": False}
        hero = s.players[self.hero_seat]
        board = [int_to_card(c) for c in s.board]
        pot = s.pot
        to_call = max(0.0, s.current_bet - hero.committed_street) if not s.done else 0.0
        mask = legal_action_mask(s) if not s.done else np.zeros(len(ACTION_NAMES), dtype=bool)

        # Which seats should have their hole cards shown?
        # - hero always
        # - any seat when reveal_all (debug) is on
        # - at showdown: every seat recorded in last_settle.revealed
        revealed_seats: set[int] = {self.hero_seat}
        if self.reveal_all:
            revealed_seats.update(p.pos for p in s.players)
        if s.done and self.last_settle is not None:
            revealed_seats.update(self.last_settle["revealed"].keys())

        players = []
        for p in s.players:
            show_hole = (p.pos in revealed_seats) and not (
                p.folded and p.pos != self.hero_seat and not self.reveal_all
            )
            hole_out = (
                [int_to_card(c) for c in p.hole] if show_hole else ["??", "??"]
            )
            players.append({
                "seat": p.pos,
                "pos": POS_NAMES[p.pos],
                "stack": round(p.stack, 2),
                "committed_street": round(p.committed_street, 2),
                "committed_total": round(p.committed_total, 2),
                "folded": p.folded,
                "all_in": p.all_in,
                "is_hero": p.pos == self.hero_seat,
                "is_to_act": (not s.done) and (p.pos == s.to_act),
                "is_winner": p.pos in s.winner_seats,
                "is_button": p.pos == BTN_SEAT,
                "is_sb": p.pos == SB_SEAT,
                "is_bb": p.pos == BB_SEAT,
                "hole": hole_out,
                "hole_revealed": show_hole,
            })

        # Raise slider bounds in bb.
        min_raise_to = s.current_bet + s.min_raise
        max_raise_to = hero.committed_street + hero.stack
        can_raise = (not s.done) and bool(mask[RAISE_START:ALL_IN].any()) or bool(mask[ALL_IN])

        street_name = (
            "Preflop" if s.street == 0
            else "Flop" if s.street == 1
            else "Turn" if s.street == 2
            else "River" if s.street == 3
            else "Showdown"
        )

        pot_frac_preview = []
        for i, frac in enumerate(RAISE_POT_FRACTIONS):
            b = RAISE_START + i
            post_call_pot = s.pot + to_call
            raise_to = to_call + frac * post_call_pot + hero.committed_street
            raise_to = max(raise_to, min_raise_to)
            raise_to = min(raise_to, max_raise_to)
            pot_frac_preview.append({
                "bucket": b,
                "label": f"{frac:g}x pot",
                "raise_to_bb": round(raise_to, 2),
                "legal": bool(mask[b]) if not s.done else False,
            })

        return {
            "started": True,
            "hand_idx": self.hand_idx,
            "hero_seat": self.hero_seat,
            "hero_pos": POS_NAMES[self.hero_seat],
            "street": s.street,
            "street_name": street_name,
            "board": board,
            "pot": round(pot, 2),
            "current_bet": round(s.current_bet, 2),
            "min_raise": round(s.min_raise, 2),
            "min_raise_to": round(min_raise_to, 2),
            "max_raise_to": round(max_raise_to, 2),
            "to_call": round(to_call, 2),
            "to_act": s.to_act,
            "to_act_pos": POS_NAMES[s.to_act] if s.to_act >= 0 else "",
            "players": players,
            "done": s.done,
            "waiting_for_human": (not s.done) and (s.to_act == self.hero_seat),
            "legal": {
                "fold": bool(mask[FOLD]),
                "check_call": bool(mask[CHECK_CALL]),
                "allin": bool(mask[ALL_IN]),
                "any_raise": can_raise,
            },
            "raise_buttons": pot_frac_preview,
            "log": list(self.log),
            "settle": (
                {
                    "winners": [POS_NAMES[i] for i in self.last_settle["winners"]],
                    "winner_seats": list(self.last_settle["winners"]),
                    "pnl": round(self.last_settle["pnl"], 2),
                    "went_to_showdown": self.last_settle.get("went_to_showdown", False),
                }
                if self.last_settle else None
            ),
            "btn_seat": BTN_SEAT,
            "sb_seat": SB_SEAT,
            "bb_seat": BB_SEAT,
            "reveal_all": self.reveal_all,
            "session": {
                "hands": self.session["hands"],
                "pnl": round(self.session["pnl"], 2),
                "wins": self.session["wins"],
                "losses": self.session["losses"],
                "ties": self.session["ties"],
                "best": round(self.session["best"], 2),
                "worst": round(self.session["worst"], 2),
                "avg_bb_per_hand": (
                    round(self.session["pnl"] / self.session["hands"], 3)
                    if self.session["hands"] > 0 else 0.0
                ),
            },
        }


# ----------------------------------------------------------------------------
# Flask app
# ----------------------------------------------------------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")
GAME: Game | None = None


@app.after_request
def _no_cache(resp):
    # Local dev tool — never let the browser cache stale JS/CSS/HTML.
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/state")
def api_state():
    return jsonify(GAME.as_dict())


@app.post("/api/new_hand")
def api_new_hand():
    with GAME.lock:
        GAME.new_hand()
    return jsonify(GAME.as_dict())


@app.post("/api/reveal")
def api_reveal():
    body = request.get_json(silent=True) or {}
    with GAME.lock:
        GAME.reveal_all = bool(body.get("on", False))
    return jsonify(GAME.as_dict())


@app.post("/api/reset_session")
def api_reset_session():
    with GAME.lock:
        GAME.reset_session()
    return jsonify(GAME.as_dict())


@app.post("/api/act")
def api_act():
    body = request.get_json(silent=True) or {}
    bucket = int(body.get("bucket", CHECK_CALL))
    raise_to = body.get("raise_to_bb", None)
    raise_to = float(raise_to) if raise_to is not None else None
    with GAME.lock:
        GAME.human_act(bucket, raise_to_bb=raise_to)
    return jsonify(GAME.as_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc-ckpt", default="checkpoints/bc_full/best.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    global GAME
    GAME = Game(args.bc_ckpt, device=args.device, seed=args.seed)
    print(f"Loaded BC model from {args.bc_ckpt}")
    print(f"Open http://{args.host}:{args.port} to play.")
    # Flask's dev server is fine for a local single-user tool.
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
