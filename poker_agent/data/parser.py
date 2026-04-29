"""Parser for PHH single-hand and PHHS multi-hand files.

We do NOT use pokerkit in the hot path (too slow for 8M hands). We parse the
TOML-lite subset we actually need. PHH fields we care about:

    variant = 'NT'
    antes = [..]
    blinds_or_straddles = [..]
    min_bet = 100
    starting_stacks = [..]
    actions = ['d dh p1 TcQc', 'p3 f', 'p4 cbr 210', ...]
    seat_count = 6              # optional
    hand = 123                  # optional
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

_SECTION_RE = re.compile(r"(?m)^\[(\d+)\]\s*$")


@dataclass
class HandRecord:
    """Raw parsed hand (still a bag of strings; replay.py turns it into states)."""

    variant: str
    antes: list[float]
    blinds: list[float]
    min_bet: float
    starting_stacks: list[float]
    actions: list[str]
    seat_count: int = 0
    src_file: str = ""
    src_idx: int = 0  # [N] index inside a phhs; 0 for single phh
    extra: dict = field(default_factory=dict)

    @property
    def num_players(self) -> int:
        return len(self.starting_stacks)


def _parse_value(v: str):
    v = v.strip()
    # Handle TOML-style time literal (HH:MM:SS) -> keep as string
    if re.fullmatch(r"\d{1,2}:\d{2}:\d{2}", v):
        return v
    try:
        return ast.literal_eval(v)
    except Exception:
        return v


def _parse_section(body: str, src_file: str, src_idx: int) -> HandRecord | None:
    kv: dict = {}
    buf = ""
    in_multi = False
    bracket_depth = 0
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            if not in_multi:
                continue
        if not in_multi:
            if "=" not in stripped:
                continue
            key, _, rest = stripped.partition("=")
            key = key.strip()
            rest = rest.strip()
            # Check if value starts a multi-line list/dict
            if rest.startswith(("[", "{")):
                bracket_depth = rest.count("[") + rest.count("{") - rest.count("]") - rest.count("}")
                if bracket_depth > 0:
                    buf = rest
                    in_multi = True
                    current_key = key
                    continue
            kv[key] = _parse_value(rest)
        else:
            buf += "\n" + line
            bracket_depth += line.count("[") + line.count("{") - line.count("]") - line.count("}")
            if bracket_depth <= 0:
                kv[current_key] = _parse_value(buf)
                buf = ""
                in_multi = False
    if kv.get("variant") != "NT":
        return None
    try:
        return HandRecord(
            variant=kv["variant"],
            antes=list(kv.get("antes", [])),
            blinds=list(kv["blinds_or_straddles"]),
            min_bet=float(kv["min_bet"]),
            starting_stacks=list(kv["starting_stacks"]),
            actions=list(kv["actions"]),
            seat_count=int(kv.get("seat_count", len(kv["starting_stacks"]))),
            src_file=src_file,
            src_idx=src_idx,
            extra={k: kv[k] for k in kv if k not in {
                "variant", "antes", "blinds_or_straddles", "min_bet",
                "starting_stacks", "actions", "seat_count",
                "ante_trimming_status",
            }},
        )
    except (KeyError, TypeError, ValueError):
        return None


def parse_phh(path: str | Path) -> HandRecord | None:
    path = Path(path)
    return _parse_section(path.read_text(encoding="utf-8", errors="ignore"),
                          str(path), 0)


def iter_phhs(path: str | Path) -> Iterator[HandRecord]:
    """Yield each hand inside a multi-hand .phhs file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = [(m.start(), m.end(), int(m.group(1)))
               for m in _SECTION_RE.finditer(text)]
    if not matches:
        return
    for i, (start, hdr_end, idx) in enumerate(matches):
        body_start = hdr_end
        body_end = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        rec = _parse_section(text[body_start:body_end], str(path), idx)
        if rec is not None:
            yield rec
