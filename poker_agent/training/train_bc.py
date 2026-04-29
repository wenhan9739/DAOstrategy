"""Behaviour cloning training loop. CUDA + bf16 AMP."""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..data.dataset import BCListDataset, ShardLocalSampler, make_splits
from ..models.bc import BCModel, bc_loss


def _move(batch: dict, device: str) -> dict:
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()}


def build_loaders(shard_glob: str, val_frac: float, batch_size: int, workers: int,
                  only_num_players: int | None = None, suit_permute: bool = False):
    train_paths, val_paths = make_splits(shard_glob, val_frac=val_frac)
    print(f"train shards: {len(train_paths)}, val shards: {len(val_paths)}")
    train_ds = BCListDataset(train_paths, only_num_players=only_num_players,
                             suit_permute=suit_permute)
    # Never augment the validation set (keeps metric comparable across runs).
    val_ds = BCListDataset(val_paths, only_num_players=only_num_players,
                           suit_permute=False)
    # Keep shard cache very small (1-2 shards) to bound RAM.
    train_ds._CACHE_CAP = 2
    val_ds._CACHE_CAP = 2
    print(f"train samples: {len(train_ds):,}, val samples: {len(val_ds):,}"
          f"  (filter num_players={only_num_players}, suit_permute={suit_permute})")

    # Shard-local sampler: shuffles shards + samples within shards so each
    # minibatch stays inside a single shard (no random-access disk I/O).
    train_sampler = ShardLocalSampler(train_ds, seed=0)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler,
        num_workers=workers, pin_memory=True, drop_last=True,
        persistent_workers=(workers > 0), prefetch_factor=2 if workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=max(1, workers // 2),
        pin_memory=True, drop_last=False,
        persistent_workers=(workers > 0),
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, loader, device, loss_cfg: dict, max_batches: int = 200):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = _move(batch, device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch)
            stats = bc_loss(out, batch, **loss_cfg)
        total_loss += stats["loss"].item()
        total_acc += stats["acc"].item()
        n += 1
    model.train()
    return total_loss / max(1, n), total_acc / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    device = cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        device = "cpu"
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    # Optional VRAM cap (leave room for user's desktop apps).
    gpu_frac = float(cfg.get("gpu_mem_fraction", os.environ.get("POKER_GPU_FRAC", "0")) or 0)
    if device == "cuda" and gpu_frac > 0:
        torch.cuda.set_per_process_memory_fraction(gpu_frac, device=0)
        print(f"Capped GPU memory fraction to {gpu_frac:.2f}")

    train_loader, val_loader = build_loaders(
        shard_glob=cfg["shard_glob"],
        val_frac=cfg.get("val_frac", 0.05),
        batch_size=cfg["batch_size"],
        workers=cfg.get("num_workers", 4),
        only_num_players=cfg.get("only_num_players", None),
        suit_permute=cfg.get("suit_permute", False),
    )

    loss_cfg = dict(
        rfrac_weight=cfg.get("rfrac_weight", 0.1),
        label_smoothing=cfg.get("label_smoothing", 0.05),
        raise_neighbor_weight=cfg.get("raise_neighbor_weight", 0.15),
    )

    model = BCModel(
        d_model=cfg.get("d_model", 192),
        n_layers=cfg.get("n_layers", 4),
        n_heads=cfg.get("n_heads", 4),
        dropout=cfg.get("dropout", 0.1),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.2f}M params")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.01),
        betas=(0.9, 0.95),
    )

    total_steps = cfg["total_steps"]
    warmup = cfg.get("warmup_steps", 1000)
    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(0.05, 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    log_dir = Path(cfg.get("log_dir", "runs/bc"))
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)

    ckpt_dir = Path(cfg.get("ckpt_dir", "checkpoints/bc"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    grad_clip = cfg.get("grad_clip", 1.0)
    log_every = cfg.get("log_every", 50)
    eval_every = cfg.get("eval_every", 2000)
    save_every = cfg.get("save_every", 5000)

    best_val_loss = float("inf")
    step = 0
    model.train()
    t0 = time.time()
    running = {"loss": 0.0, "acc": 0.0, "pol": 0.0, "rf": 0.0, "n": 0}

    data_iter = iter(train_loader)
    while step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        batch = _move(batch, device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch)
            stats = bc_loss(out, batch, **loss_cfg)
        loss = stats["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        scheduler.step()
        step += 1

        running["loss"] += loss.item()
        running["acc"] += stats["acc"].item()
        running["pol"] += stats["policy_loss"].item()
        running["rf"]  += stats["rfrac_loss"].item()
        running["n"]   += 1

        if step % log_every == 0:
            n = running["n"]
            lr = scheduler.get_last_lr()[0]
            dt = time.time() - t0
            msg = (f"step {step:6d} | lr {lr:.2e} | loss {running['loss']/n:.4f} "
                   f"pol {running['pol']/n:.4f} rf {running['rf']/n:.4f} "
                   f"acc {running['acc']/n*100:.1f}% | {step/dt:.0f} it/s")
            print(msg, flush=True)
            writer.add_scalar("train/loss", running["loss"]/n, step)
            writer.add_scalar("train/acc", running["acc"]/n, step)
            writer.add_scalar("train/policy_loss", running["pol"]/n, step)
            writer.add_scalar("train/rfrac_loss", running["rf"]/n, step)
            writer.add_scalar("lr", lr, step)
            running = {"loss": 0.0, "acc": 0.0, "pol": 0.0, "rf": 0.0, "n": 0}

        if step % eval_every == 0:
            val_loss, val_acc = evaluate(model, val_loader, device, loss_cfg,
                                          max_batches=cfg.get("eval_batches", 200))
            print(f"[eval] step {step} val_loss {val_loss:.4f} val_acc {val_acc*100:.1f}%", flush=True)
            writer.add_scalar("val/loss", val_loss, step)
            writer.add_scalar("val/acc", val_acc, step)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({"model": model.state_dict(), "step": step,
                            "cfg": cfg, "val_loss": val_loss},
                           ckpt_dir / "best.pt")
                print(f"  -> saved best ({val_loss:.4f})", flush=True)

        if step % save_every == 0:
            torch.save({"model": model.state_dict(), "step": step, "cfg": cfg},
                       ckpt_dir / f"step_{step:06d}.pt")

    torch.save({"model": model.state_dict(), "step": step, "cfg": cfg},
               ckpt_dir / "final.pt")
    writer.close()


if __name__ == "__main__":
    main()
