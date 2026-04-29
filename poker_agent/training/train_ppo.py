"""PPO training loop: hero vs frozen BC opponents in vectorized env."""
from __future__ import annotations

import argparse
import copy
import math
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter

from ..env.actions import CHECK_CALL, NUM_ACTIONS
from ..env.holdem import PokerVecEnv
from ..env.subproc_vec import SubprocPokerVecEnv
from ..models.bc import BCModel
from ..models.ppo import ActorCritic, load_bc_checkpoint, masked_logits, sample_action


# ---------------------------------------------------------------------------
# Observation utilities (numpy <-> torch)
# ---------------------------------------------------------------------------
OBS_KEYS_LONG = ("ctx_int", "board", "hole", "hist_actor", "hist_street", "hist_bucket")
OBS_KEYS_FLOAT = ("ctx_float", "hist_rfrac")


def obs_to_tensor(obs: dict, device: str) -> dict:
    out = {}
    for k in OBS_KEYS_LONG:
        out[k] = torch.from_numpy(np.asarray(obs[k], dtype=np.int64)).to(device, non_blocking=True)
    for k in OBS_KEYS_FLOAT:
        out[k] = torch.from_numpy(np.asarray(obs[k], dtype=np.float32)).to(device, non_blocking=True)
    out["active"] = torch.from_numpy(np.asarray(obs["active"], dtype=np.bool_)).to(device, non_blocking=True)
    return out


# ---------------------------------------------------------------------------
# Opponent callable
# ---------------------------------------------------------------------------
class BCOpponent:
    """Opponent decision fn driven by a frozen BC model."""

    def __init__(self, model: BCModel, device: str, temperature: float = 1.0):
        self.model = model.eval()
        self.device = device
        self.temperature = temperature

    @torch.no_grad()
    def __call__(self, obs_list: list[dict]) -> list[int]:
        if not obs_list:
            return []
        # batch
        batch = {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}
        tb = obs_to_tensor(batch, self.device)
        with torch.amp.autocast(self.device if self.device == "cuda" else "cpu",
                                 dtype=torch.bfloat16, enabled=(self.device == "cuda")):
            out = self.model(tb)
        logits = out["policy_logits"].float() / max(1e-3, self.temperature)
        # no mask info here; sample from full distribution. Env will sanitize illegal actions.
        probs = F.softmax(logits, dim=-1)
        actions = torch.multinomial(probs, num_samples=1).squeeze(-1).cpu().numpy()
        return actions.tolist()


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------
class RolloutBuffer:
    def __init__(self, n_steps: int, n_envs: int, obs_sample: dict, device: str):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.device = device
        self.ptr = 0
        self.full = False

        def alloc(shape, dtype):
            return torch.zeros((n_steps, n_envs) + tuple(shape), dtype=dtype, device=device)

        self.obs = {k: alloc(v.shape[1:], torch.long if v.dtype in (torch.int64, torch.int32) else torch.float32)
                    for k, v in obs_sample.items()}
        # active bool
        self.obs["active"] = alloc(obs_sample["active"].shape[1:], torch.bool)
        self.masks = alloc((NUM_ACTIONS,), torch.bool)
        self.actions = alloc((), torch.long)
        self.logp = alloc((), torch.float32)
        self.values = alloc((), torch.float32)
        self.rewards = alloc((), torch.float32)
        self.dones = alloc((), torch.bool)
        self.ref_logits = alloc((NUM_ACTIONS,), torch.float32)  # frozen BC logits for KL penalty

    def add(self, obs_t, mask, action, logp, value, reward, done, ref_logits):
        i = self.ptr
        for k, v in obs_t.items():
            self.obs[k][i] = v
        self.masks[i] = mask
        self.actions[i] = action
        self.logp[i] = logp
        self.values[i] = value
        self.rewards[i] = reward
        self.dones[i] = done
        self.ref_logits[i] = ref_logits
        self.ptr += 1
        if self.ptr >= self.n_steps:
            self.full = True

    def reset(self):
        self.ptr = 0
        self.full = False

    def compute_gae(self, last_values, last_dones, gamma: float = 0.99, lam: float = 0.95):
        adv = torch.zeros_like(self.rewards)
        lastgaelam = torch.zeros(self.n_envs, device=self.device)
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                nonterm = ~last_dones
                nextvals = last_values
            else:
                nonterm = ~self.dones[t + 1]
                nextvals = self.values[t + 1]
            delta = self.rewards[t] + gamma * nextvals * nonterm.float() - self.values[t]
            lastgaelam = delta + gamma * lam * nonterm.float() * lastgaelam
            adv[t] = lastgaelam
        returns = adv + self.values
        self.adv = adv
        self.returns = returns

    def iter_minibatches(self, mb_size: int):
        total = self.n_steps * self.n_envs
        idx = torch.randperm(total, device=self.device)
        for start in range(0, total, mb_size):
            sel = idx[start:start + mb_size]
            t = sel // self.n_envs
            e = sel % self.n_envs
            batch = {k: v[t, e] for k, v in self.obs.items()}
            yield {
                "batch": batch,
                "mask": self.masks[t, e],
                "action": self.actions[t, e],
                "logp_old": self.logp[t, e],
                "value_old": self.values[t, e],
                "adv": self.adv[t, e],
                "ret": self.returns[t, e],
                "ref_logits": self.ref_logits[t, e],
            }


# ---------------------------------------------------------------------------
# Main PPO loop
# ---------------------------------------------------------------------------
def _save_resume_ckpt(path: Path, *, hero, opt, it: int, bc_arch: dict, cfg: dict,
                      hands_done: int, running_reward: float, running_count: int,
                      elapsed: float, device: str) -> None:
    """Atomic save of a full-state resume checkpoint."""
    payload = {
        "model": hero.bc.state_dict(),
        "optimizer": opt.state_dict(),
        "iter": it,
        "hands_done": hands_done,
        "running_reward": running_reward,
        "running_count": running_count,
        "elapsed": elapsed,
        "cfg": {**cfg, **bc_arch},
        "np_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" and torch.cuda.is_available() else None,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic on same filesystem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--bc-ckpt", required=True, help="path to BC checkpoint (.pt)")
    ap.add_argument("--resume", default="auto",
                    help="path to resume ckpt; 'auto' = <ckpt_dir>/latest.pt if present; 'none' to force fresh start")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    device = cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    # Optional VRAM cap (leave headroom for the user's desktop apps).
    gpu_frac = float(cfg.get("gpu_mem_fraction", os.environ.get("POKER_GPU_FRAC", "0")) or 0)
    if device == "cuda" and gpu_frac > 0:
        torch.cuda.set_per_process_memory_fraction(gpu_frac, device=0)
        print(f"Capped GPU memory fraction to {gpu_frac:.2f}", flush=True)

    # ---- build models
    bc_opp_model = load_bc_checkpoint(args.bc_ckpt, map_location=device).to(device)
    for p in bc_opp_model.parameters():
        p.requires_grad_(False)
    bc_opp_model.eval()
    opp_fn = BCOpponent(bc_opp_model, device=device,
                        temperature=cfg.get("opp_temperature", 1.0))

    hero_bc = load_bc_checkpoint(args.bc_ckpt, map_location=device).to(device)
    hero = ActorCritic(hero_bc).to(device)
    # remember BC architecture so we can reload hero later without the bc_ckpt
    bc_ckpt_obj = torch.load(args.bc_ckpt, map_location="cpu", weights_only=False)
    bc_arch = {k: bc_ckpt_obj.get("cfg", {}).get(k)
               for k in ("d_model", "n_heads", "n_layers", "dropout")
               if bc_ckpt_obj.get("cfg", {}).get(k) is not None}

    # frozen reference for KL penalty (same as opp snapshot)
    ref_model = copy.deepcopy(bc_opp_model).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # ---- build env (single-process or subprocess-parallel)
    n_workers = int(cfg.get("n_workers", 0) or 0)
    if n_workers > 0:
        envs_per_worker = int(cfg["envs_per_worker"])
        n_envs = n_workers * envs_per_worker
        env = SubprocPokerVecEnv(
            n_workers=n_workers,
            envs_per_worker=envs_per_worker,
            bc_ckpt_path=args.bc_ckpt,
            start_stack_bb=cfg.get("start_stack_bb", 100.0),
            sb_bb=cfg.get("sb_bb", 0.5),
            bb_bb=cfg.get("bb_bb", 1.0),
            opp_temperature=cfg.get("opp_temperature", 1.0),
            seed=cfg.get("seed", 0),
            torch_threads=int(cfg.get("worker_torch_threads", 2)),
        )
        print(f"SubprocVecEnv: {n_workers} workers x {envs_per_worker} envs = {n_envs} total",
              flush=True)
    else:
        n_envs = cfg["n_envs"]
        env = PokerVecEnv(
            n_envs=n_envs,
            opponent_fn=opp_fn,
            start_stack_bb=cfg.get("start_stack_bb", 100.0),
            sb_bb=cfg.get("sb_bb", 0.5),
            bb_bb=cfg.get("bb_bb", 1.0),
            seed=cfg.get("seed", 0),
        )
    obs_np, mask_np = env.reset()
    obs_t = obs_to_tensor(obs_np, device)
    mask_t = torch.from_numpy(mask_np).to(device)

    buffer = RolloutBuffer(n_steps=cfg["n_steps"], n_envs=n_envs,
                           obs_sample=obs_t, device=device)

    opt = torch.optim.AdamW(
        [p for p in hero.parameters() if p.requires_grad],
        lr=cfg["lr"],
        betas=(0.9, 0.95),
        weight_decay=cfg.get("weight_decay", 0.0),
    )

    log_dir = Path(cfg.get("log_dir", "runs/ppo"))
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)
    ckpt_dir = Path(cfg.get("ckpt_dir", "checkpoints/ppo"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    gamma = cfg.get("gamma", 0.99)
    lam = cfg.get("gae_lambda", 0.95)
    clip_eps = cfg.get("clip_eps", 0.2)
    vf_coef = cfg.get("vf_coef", 0.5)
    ent_coef = cfg.get("ent_coef", 0.01)
    kl_coef = cfg.get("kl_to_bc_coef", 0.02)
    epochs = cfg.get("epochs", 4)
    mb_size = cfg.get("minibatch_size", 4096)
    max_grad = cfg.get("grad_clip", 0.5)

    total_iters = cfg["total_iters"]
    t0 = time.time()
    hands_done = 0
    running_reward = 0.0
    running_count = 0
    start_iter = 1
    elapsed_offset = 0.0

    # ---- resume from latest.pt if requested ----
    resume_path: Path | None = None
    if args.resume == "auto":
        cand = ckpt_dir / "latest.pt"
        if cand.exists():
            resume_path = cand
    elif args.resume and args.resume != "none":
        resume_path = Path(args.resume)
    if resume_path is not None and resume_path.exists():
        print(f"Resuming from {resume_path}", flush=True)
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        hero.bc.load_state_dict(ck["model"])
        try:
            opt.load_state_dict(ck["optimizer"])
        except Exception as e:
            print(f"[WARN] optimizer state load failed ({e}); continuing with fresh optimizer", flush=True)
        start_iter = int(ck.get("iter", 0)) + 1
        hands_done = int(ck.get("hands_done", 0))
        running_reward = float(ck.get("running_reward", 0.0))
        running_count = int(ck.get("running_count", 0))
        elapsed_offset = float(ck.get("elapsed", 0.0))
        try:
            np.random.set_state(ck["np_rng"])
            torch.set_rng_state(ck["torch_rng"].cpu() if isinstance(ck["torch_rng"], torch.Tensor) else ck["torch_rng"])
            if device == "cuda" and ck.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all(ck["cuda_rng"])
        except Exception as e:
            print(f"[WARN] RNG state restore failed ({e}); continuing", flush=True)
        print(f"Resumed at iter {start_iter-1}, hands={hands_done}, elapsed={elapsed_offset:.0f}s", flush=True)
    elif args.resume == "auto":
        print("No latest.pt found; starting fresh.", flush=True)

    # ---- graceful shutdown: save latest.pt on SIGINT/SIGTERM/SIGBREAK ----
    _shutdown_state = {"requested": False}

    def _request_shutdown(signum, frame):
        if _shutdown_state["requested"]:
            print("[signal] second signal received; hard-exiting", flush=True, file=sys.stderr)
            os._exit(1)
        _shutdown_state["requested"] = True
        print(f"[signal] caught {signum}; will save & exit after current iter", flush=True, file=sys.stderr)

    for sig_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, sig_name):
            try:
                signal.signal(getattr(signal, sig_name), _request_shutdown)
            except (ValueError, OSError):
                pass

    for it in range(start_iter, total_iters + 1):
        buffer.reset()
        # -------- rollout collection --------
        hero.eval()
        for t in range(cfg["n_steps"]):
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                    out = hero.bc(obs_t)
                    ref_out = ref_model(obs_t)
                logits = out["policy_logits"].float()
                values = out["value"].float()
                ref_logits = ref_out["policy_logits"].float()
                ml = masked_logits(logits, mask_t)
                dist = torch.distributions.Categorical(logits=ml)
                action = dist.sample()
                logp = dist.log_prob(action)

            acts_np = action.cpu().numpy()
            next_obs_np, rew_np, done_np, next_mask_np, info = env.step(acts_np)

            buffer.add(
                obs_t, mask_t,
                action, logp, values,
                torch.from_numpy(rew_np).to(device),
                torch.from_numpy(done_np).to(device),
                ref_logits,
            )

            running_reward += float(rew_np[done_np].sum())
            running_count += int(done_np.sum())

            obs_t = obs_to_tensor(next_obs_np, device)
            mask_t = torch.from_numpy(next_mask_np).to(device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                last_out = hero.bc(obs_t)
            last_values = last_out["value"].float()
            last_dones = torch.zeros(n_envs, dtype=torch.bool, device=device)
        buffer.compute_gae(last_values=last_values, last_dones=last_dones,
                           gamma=gamma, lam=lam)
        # normalize advantages
        adv_mean = buffer.adv.mean()
        adv_std = buffer.adv.std().clamp(min=1e-6)
        buffer.adv = (buffer.adv - adv_mean) / adv_std

        # -------- PPO update --------
        hero.train()
        stats_accum = {"policy": 0.0, "value": 0.0, "ent": 0.0, "kl": 0.0, "clip": 0.0, "n": 0}
        for _ in range(epochs):
            for mb in buffer.iter_minibatches(mb_size):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                    out = hero.bc(mb["batch"])
                logits = out["policy_logits"].float()
                values = out["value"].float()
                ml = masked_logits(logits, mb["mask"])
                dist = torch.distributions.Categorical(logits=ml)
                new_logp = dist.log_prob(mb["action"])
                entropy = dist.entropy().mean()

                ratio = (new_logp - mb["logp_old"]).exp()
                pg1 = ratio * mb["adv"]
                pg2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb["adv"]
                policy_loss = -torch.min(pg1, pg2).mean()

                v_clipped = mb["value_old"] + (values - mb["value_old"]).clamp(-clip_eps, clip_eps)
                vl1 = (values - mb["ret"]).pow(2)
                vl2 = (v_clipped - mb["ret"]).pow(2)
                value_loss = torch.max(vl1, vl2).mean()

                # KL penalty to the frozen BC reference
                ref_ml = masked_logits(mb["ref_logits"], mb["mask"])
                ref_dist = torch.distributions.Categorical(logits=ref_ml)
                kl = torch.distributions.kl.kl_divergence(dist, ref_dist).mean()

                loss = (policy_loss
                        + vf_coef * value_loss
                        - ent_coef * entropy
                        + kl_coef * kl)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(hero.parameters(), max_grad)
                opt.step()

                clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean().item()
                stats_accum["policy"] += policy_loss.item()
                stats_accum["value"]  += value_loss.item()
                stats_accum["ent"]    += entropy.item()
                stats_accum["kl"]     += kl.item()
                stats_accum["clip"]   += clip_frac
                stats_accum["n"]      += 1

        # -------- logging --------
        n = max(1, stats_accum["n"])
        hands_done = info["hands"]
        mbb_per_hand = (running_reward / max(1, running_count)) * 1000.0  # mbb/hand
        elapsed = time.time() - t0 + elapsed_offset
        print(
            f"iter {it:5d} | hands {hands_done:7d} | mbb/hand {mbb_per_hand:+7.1f} | "
            f"pol {stats_accum['policy']/n:+.4f} val {stats_accum['value']/n:.4f} "
            f"ent {stats_accum['ent']/n:.3f} kl {stats_accum['kl']/n:.4f} "
            f"clip {stats_accum['clip']/n:.3f} | {elapsed:.0f}s",
            flush=True,
        )
        writer.add_scalar("train/mbb_per_hand", mbb_per_hand, it)
        writer.add_scalar("train/policy_loss", stats_accum["policy"]/n, it)
        writer.add_scalar("train/value_loss", stats_accum["value"]/n, it)
        writer.add_scalar("train/entropy", stats_accum["ent"]/n, it)
        writer.add_scalar("train/kl_to_bc", stats_accum["kl"]/n, it)
        writer.add_scalar("train/clip_frac", stats_accum["clip"]/n, it)
        writer.add_scalar("env/hands", hands_done, it)
        running_reward = 0.0
        running_count = 0

        if it % cfg.get("save_every", 50) == 0:
            torch.save({"model": hero.bc.state_dict(), "iter": it,
                        "cfg": {**cfg, **bc_arch}},
                       ckpt_dir / f"iter_{it:05d}.pt")
            _save_resume_ckpt(
                ckpt_dir / "latest.pt",
                hero=hero, opt=opt, it=it, bc_arch=bc_arch, cfg=cfg,
                hands_done=hands_done, running_reward=running_reward,
                running_count=running_count, elapsed=elapsed, device=device,
            )

        if _shutdown_state["requested"]:
            print("[signal] saving latest.pt and exiting cleanly", flush=True, file=sys.stderr)
            _save_resume_ckpt(
                ckpt_dir / "latest.pt",
                hero=hero, opt=opt, it=it, bc_arch=bc_arch, cfg=cfg,
                hands_done=hands_done, running_reward=running_reward,
                running_count=running_count, elapsed=elapsed, device=device,
            )
            writer.close()
            return

    torch.save({"model": hero.bc.state_dict(), "iter": total_iters,
                "cfg": {**cfg, **bc_arch}},
               ckpt_dir / "final.pt")
    _save_resume_ckpt(
        ckpt_dir / "latest.pt",
        hero=hero, opt=opt, it=total_iters, bc_arch=bc_arch, cfg=cfg,
        hands_done=hands_done, running_reward=running_reward,
        running_count=running_count, elapsed=time.time() - t0 + elapsed_offset,
        device=device,
    )
    writer.close()


if __name__ == "__main__":
    main()
