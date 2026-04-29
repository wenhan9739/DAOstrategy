"""Evaluate a hero checkpoint vs a BC opponent pool by playing many hands.

Reports mbb/hand with a 95% confidence interval.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from ..env.actions import NUM_ACTIONS
from ..env.holdem import PokerVecEnv
from ..models.bc import BCModel
from ..models.ppo import load_bc_checkpoint, masked_logits
from ..training.train_ppo import BCOpponent, obs_to_tensor


@torch.no_grad()
def hero_act(model: BCModel, obs_t: dict, mask_t: torch.Tensor,
             deterministic: bool = False) -> torch.Tensor:
    out = model(obs_t)
    ml = masked_logits(out["policy_logits"].float(), mask_t)
    if deterministic:
        return ml.argmax(dim=-1)
    return torch.distributions.Categorical(logits=ml).sample()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hero-ckpt", required=True)
    ap.add_argument("--opp-ckpt", required=True)
    ap.add_argument("--n-hands", type=int, default=100_000)
    ap.add_argument("--n-envs", type=int, default=256)
    ap.add_argument("--start-stack-bb", type=float, default=100.0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hero = load_bc_checkpoint(args.hero_ckpt, map_location=device).to(device).eval()
    opp_bc = load_bc_checkpoint(args.opp_ckpt, map_location=device).to(device).eval()
    opp_fn = BCOpponent(opp_bc, device=device, temperature=1.0)

    env = PokerVecEnv(n_envs=args.n_envs, opponent_fn=opp_fn,
                      start_stack_bb=args.start_stack_bb,
                      sb_bb=0.5, bb_bb=1.0, seed=args.seed)
    obs_np, mask_np = env.reset()
    obs_t = obs_to_tensor(obs_np, device)
    mask_t = torch.from_numpy(mask_np).to(device)

    rewards: list[float] = []
    t0 = time.time()
    step = 0
    while len(rewards) < args.n_hands:
        with torch.amp.autocast(device if device == "cuda" else "cpu",
                                 dtype=torch.bfloat16, enabled=(device == "cuda")):
            action = hero_act(hero, obs_t, mask_t, args.deterministic)
        acts_np = action.cpu().numpy()
        obs_np, rew_np, done_np, mask_np, info = env.step(acts_np)
        obs_t = obs_to_tensor(obs_np, device)
        mask_t = torch.from_numpy(mask_np).to(device)
        if done_np.any():
            rewards.extend(rew_np[done_np].tolist())
        step += 1
        if step % 500 == 0:
            mean = np.mean(rewards) * 1000.0 if rewards else 0.0
            print(f"  hands={len(rewards):6d}  mbb/hand={mean:+.1f}  elapsed={time.time()-t0:.0f}s",
                  flush=True)

    arr = np.array(rewards[: args.n_hands], dtype=np.float64)
    mean_bb = arr.mean()
    sem = arr.std(ddof=1) / np.sqrt(len(arr))
    mbb = mean_bb * 1000.0
    ci = 1.96 * sem * 1000.0
    print("\n=== Evaluation ===")
    print(f"Hero        : {args.hero_ckpt}")
    print(f"Opponent BC : {args.opp_ckpt}")
    print(f"Hands       : {len(arr):,}")
    print(f"Mean        : {mbb:+.2f} mbb/hand  (95% CI ±{ci:.1f})")
    print(f"Win rate    : {(arr > 0).mean() * 100:.1f}% of hands profitable")


if __name__ == "__main__":
    main()
