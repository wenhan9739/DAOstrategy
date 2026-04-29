"""Subprocess-parallel wrapper around ``PokerVecEnv``.

The native ``PokerVecEnv`` steps a pool of poker hands in a tight Python
loop: this is the wall-clock bottleneck during PPO rollouts and keeps
the GPU at 20-30% utilization. This wrapper shards the envs across K
worker processes so env stepping runs in parallel (Python GIL bypassed
via multiprocessing).

Each worker loads its own BC opponent model **on CPU** (model is tiny,
~1.8M params; CPU forward on batch ~32 is ~5-15 ms). This keeps the
main GPU free for hero forward/backward and for the reference model
used in the KL penalty.
"""
from __future__ import annotations

import multiprocessing as mp
from typing import Any

import numpy as np


def _worker(
    remote,
    parent_remote,
    n_envs_local: int,
    bc_ckpt_path: str,
    seed: int,
    start_stack_bb: float,
    sb_bb: float,
    bb_bb: float,
    opp_temperature: float,
    torch_threads: int,
) -> None:
    """Entry point for a worker process."""
    parent_remote.close()

    import torch

    torch.set_num_threads(max(1, torch_threads))
    torch.set_num_interop_threads(1)

    from ..models.ppo import load_bc_checkpoint
    from ..training.train_ppo import BCOpponent
    from .holdem import PokerVecEnv

    bc = load_bc_checkpoint(bc_ckpt_path, map_location="cpu").eval()
    for p in bc.parameters():
        p.requires_grad_(False)
    opp = BCOpponent(bc, device="cpu", temperature=opp_temperature)

    env = PokerVecEnv(
        n_envs=n_envs_local,
        opponent_fn=opp,
        start_stack_bb=start_stack_bb,
        sb_bb=sb_bb,
        bb_bb=bb_bb,
        seed=seed,
    )
    obs, mask = env.reset()
    remote.send(("ready", (obs, mask)))

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                out = env.step(data)
                remote.send(("step", out))
            elif cmd == "close":
                break
            else:
                raise RuntimeError(f"unknown cmd {cmd}")
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        try:
            remote.close()
        except Exception:
            pass


class SubprocPokerVecEnv:
    """Drop-in replacement for ``PokerVecEnv`` that shards envs across workers.

    API matches ``PokerVecEnv``:
      - ``reset() -> (obs_batch, mask_batch)``
      - ``step(hero_actions) -> (obs, reward, done, mask, info)``
    """

    def __init__(
        self,
        n_workers: int,
        envs_per_worker: int,
        bc_ckpt_path: str,
        start_stack_bb: float = 100.0,
        sb_bb: float = 0.5,
        bb_bb: float = 1.0,
        opp_temperature: float = 1.0,
        seed: int = 0,
        torch_threads: int = 2,
    ) -> None:
        self.n_workers = n_workers
        self.envs_per_worker = envs_per_worker
        self.n_envs = n_workers * envs_per_worker

        ctx = mp.get_context("spawn")
        self.remotes = []
        self.processes = []

        for w in range(n_workers):
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(
                target=_worker,
                args=(
                    child_conn,
                    parent_conn,
                    envs_per_worker,
                    bc_ckpt_path,
                    seed + w * 10_000,
                    start_stack_bb,
                    sb_bb,
                    bb_bb,
                    opp_temperature,
                    torch_threads,
                ),
                daemon=True,
            )
            proc.start()
            child_conn.close()
            self.remotes.append(parent_conn)
            self.processes.append(proc)

        # Wait for all workers to initialize + deliver first obs
        initial = []
        for r in self.remotes:
            tag, payload = r.recv()
            assert tag == "ready"
            initial.append(payload)

        obss, masks = zip(*initial)
        self._last_obs = self._stack_obs(list(obss))
        self._last_mask = np.concatenate(list(masks), axis=0)
        self.total_hands = 0
        self._closed = False

    @staticmethod
    def _stack_obs(obss: list[dict]) -> dict:
        keys = list(obss[0].keys())
        return {k: np.concatenate([o[k] for o in obss], axis=0) for k in keys}

    def reset(self) -> tuple[dict, np.ndarray]:
        return self._last_obs, self._last_mask

    def step(self, hero_actions: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, dict]:
        n = self.envs_per_worker
        for i, r in enumerate(self.remotes):
            r.send(("step", hero_actions[i * n:(i + 1) * n]))
        outs = []
        for r in self.remotes:
            tag, payload = r.recv()
            assert tag == "step"
            outs.append(payload)

        obss, rews, dones, masks, infos = zip(*outs)
        obs = self._stack_obs(list(obss))
        rew = np.concatenate(list(rews), axis=0)
        done = np.concatenate(list(dones), axis=0)
        mask = np.concatenate(list(masks), axis=0)
        self.total_hands = sum(i["hands"] for i in infos)
        return obs, rew, done, mask, {"hands": self.total_hands}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for r in self.remotes:
            try:
                r.send(("close", None))
            except Exception:
                pass
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass
