"""PyTorch Dataset for BC training.

Reads .npz shards produced by preprocess.py and serves them memory-mapped.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .features import HIST_LEN


def _permute_suits_np(cards: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Apply a suit permutation to card-index array.

    Card encoding: rank = card // 4, suit = card % 4 (see env/cards.py).
    52 is the unknown/pad sentinel and must be preserved.
    """
    out = cards.copy()
    known = cards < 52
    if known.any():
        c = cards[known].astype(np.int64)
        rank = c // 4
        suit = c % 4
        out[known] = (rank * 4 + perm[suit]).astype(cards.dtype)
    return out


class BCShardDataset(Dataset):
    """Each item is a dict of tensors for a single decision sample.

    Options
    -------
    only_num_players: if set (e.g. 6), skip samples whose ``num_players`` field
        is not equal to this value. Used to keep the training distribution pure
        6-max (heads-up / 3-max hands pollute positional statistics because
        ``_assign_positions`` maps HU players onto the 6-max position space).
    suit_permute: if True, apply a random 24-way suit permutation to each
        sample's board / hole cards (history carries no card info). This is a
        free data-augmentation that preserves all hand-evaluation semantics.
    """

    ARRAY_KEYS = (
        "ctx_float", "ctx_int", "board", "hole", "active",
        "hist_actor", "hist_street", "hist_bucket", "hist_rfrac",
        "label_bucket", "label_rfrac", "num_players", "actor_idx",
    )

    def __init__(self, shard_glob: str,
                 only_num_players: int | None = None,
                 suit_permute: bool = False):
        self.shard_paths = sorted(glob.glob(shard_glob))
        if not self.shard_paths:
            raise FileNotFoundError(f"no shards match {shard_glob}")
        self.only_num_players = only_num_players
        self.suit_permute = suit_permute
        self._build_index()
        self._cache: dict[int, dict] = {}

    def _build_index(self) -> None:
        """Build per-shard index arrays of kept local indices."""
        self._sizes: list[int] = []
        self._cum: list[int] = []
        self._local_indices: list[np.ndarray] = []
        tot = 0
        for p in self.shard_paths:
            with np.load(p, mmap_mode="r") as z:
                if self.only_num_players is not None:
                    keep = np.flatnonzero(
                        z["num_players"][:].astype(np.int64) == int(self.only_num_players)
                    ).astype(np.int64)
                else:
                    keep = np.arange(len(z["label_bucket"]), dtype=np.int64)
            self._local_indices.append(keep)
            self._sizes.append(len(keep))
            tot += len(keep)
            self._cum.append(tot)
        self._total = tot

    def __len__(self) -> int:
        return self._total

    def _get_shard(self, shard_idx: int) -> dict:
        if shard_idx not in self._cache:
            if len(self._cache) >= getattr(self, "_CACHE_CAP", 3):
                k = next(iter(self._cache))
                del self._cache[k]
            with np.load(self.shard_paths[shard_idx]) as z:
                self._cache[shard_idx] = {k: np.ascontiguousarray(z[k]) for k in self.ARRAY_KEYS}
        return self._cache[shard_idx]

    def __getitem__(self, idx: int) -> dict:
        # binary search for shard
        lo, hi = 0, len(self._cum) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._cum[mid] <= idx:
                lo = mid + 1
            else:
                hi = mid
        shard_idx = lo
        off = idx - (self._cum[shard_idx - 1] if shard_idx > 0 else 0)
        local = int(self._local_indices[shard_idx][off])
        shard = self._get_shard(shard_idx)

        board = shard["board"][local]
        hole = shard["hole"][local]
        if self.suit_permute:
            perm = np.random.permutation(4).astype(np.int64)
            board = _permute_suits_np(np.ascontiguousarray(board), perm)
            hole = _permute_suits_np(np.ascontiguousarray(hole), perm)

        return {
            "ctx_float": torch.from_numpy(shard["ctx_float"][local].astype(np.float32)),
            "ctx_int": torch.from_numpy(shard["ctx_int"][local].astype(np.int64)),
            "board": torch.from_numpy(board.astype(np.int64)),
            "hole": torch.from_numpy(hole.astype(np.int64)),
            "active": torch.from_numpy(shard["active"][local].astype(np.bool_)),
            "hist_actor": torch.from_numpy(shard["hist_actor"][local].astype(np.int64)),
            "hist_street": torch.from_numpy(shard["hist_street"][local].astype(np.int64)),
            "hist_bucket": torch.from_numpy(shard["hist_bucket"][local].astype(np.int64)),
            "hist_rfrac": torch.from_numpy(shard["hist_rfrac"][local].astype(np.float32)),
            "label_bucket": torch.tensor(int(shard["label_bucket"][local]), dtype=torch.long),
            "label_rfrac": torch.tensor(float(shard["label_rfrac"][local]), dtype=torch.float32),
            "num_players": torch.tensor(int(shard["num_players"][local]), dtype=torch.long),
            "actor_idx": torch.tensor(int(shard["actor_idx"][local]), dtype=torch.long),
        }


def make_splits(shard_glob: str, val_frac: float = 0.05, seed: int = 0):
    """Split shards deterministically into train/val."""
    paths = sorted(glob.glob(shard_glob))
    n_val = max(1, int(len(paths) * val_frac))
    return paths[:-n_val], paths[-n_val:]


class BCListDataset(BCShardDataset):
    """Same as BCShardDataset but takes an explicit list of shard paths.

    Placed at module level so it can be pickled by the DataLoader workers
    on Windows (spawn start method).
    """

    def __init__(self, paths: list[str],
                 only_num_players: int | None = None,
                 suit_permute: bool = False):
        self.shard_paths = list(paths)
        assert self.shard_paths, "empty shard list"
        self.only_num_players = only_num_players
        self.suit_permute = suit_permute
        self._build_index()
        self._cache = {}


class ShardLocalSampler(Sampler[int]):
    """Shard-local shuffle.

    Produces indices in shard-contiguous batches: shard order is permuted
    every epoch, and samples *within* a shard are shuffled. As long as
    ``batch_size <= samples_per_shard``, each minibatch only touches ONE
    shard, giving perfect cache locality and avoiding random disk I/O
    across the full dataset.
    """

    def __init__(self, dataset: BCShardDataset, seed: int = 0):
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        shard_order = rng.permutation(len(self.dataset.shard_paths))
        for sidx in shard_order:
            start = self.dataset._cum[sidx - 1] if sidx > 0 else 0
            end = self.dataset._cum[sidx]
            local = rng.permutation(end - start)
            for l in local:
                yield int(start + l)
        self.epoch += 1

    def __len__(self) -> int:
        return len(self.dataset)
