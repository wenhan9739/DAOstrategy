"""Filter 6-max NLHE hands from phh-dataset.

Rules:
  - variant == 'NT' (No-limit Texas Hold'em)
  - seat_count == 6 (table max seats = 6)
    * For single .phh files without seat_count, fall back to len(starting_stacks) == 6.

Output: copies/writes filtered hands under OUT_ROOT, preserving subdirectory layout.
"""
from __future__ import annotations

import re
import shutil
import sys
import time
from pathlib import Path

SRC_ROOT = Path(r"d:\Poker\DAOstrategy\phh-dataset\data")
OUT_ROOT = Path(r"d:\Poker\DAOstrategy\holdem6max")

VARIANT_RE = re.compile(r"^variant\s*=\s*['\"]([^'\"]+)['\"]", re.MULTILINE)
SEAT_COUNT_RE = re.compile(r"^seat_count\s*=\s*(\d+)", re.MULTILINE)
STARTING_STACKS_RE = re.compile(r"^starting_stacks\s*=\s*\[([^\]]*)\]", re.MULTILINE)
SECTION_SPLIT_RE = re.compile(r"(?m)^\[(\d+)\]\s*$")


def count_stacks(text: str) -> int:
    m = STARTING_STACKS_RE.search(text)
    if not m:
        return 0
    body = m.group(1).strip()
    if not body:
        return 0
    return len([x for x in body.split(",") if x.strip()])


def is_6max_nlhe_single(text: str) -> bool:
    mv = VARIANT_RE.search(text)
    if not mv or mv.group(1) != "NT":
        return False
    ms = SEAT_COUNT_RE.search(text)
    if ms:
        return int(ms.group(1)) == 6
    return count_stacks(text) == 6


def process_single_phh(src: Path, out: Path) -> bool:
    try:
        text = src.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    if not is_6max_nlhe_single(text):
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out)
    return True


def split_phhs_sections(text: str):
    """Yield (header_idx:int, body:str) for each [N] section."""
    positions = [(m.start(), m.end(), int(m.group(1))) for m in SECTION_SPLIT_RE.finditer(text)]
    if not positions:
        return
    for i, (start, hdr_end, idx) in enumerate(positions):
        body_start = hdr_end
        body_end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        yield idx, text[body_start:body_end]


def process_phhs(src: Path, out: Path) -> tuple[int, int]:
    """Return (kept, total)."""
    try:
        text = src.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return 0, 0
    kept_chunks: list[str] = []
    total = 0
    kept = 0
    for idx, body in split_phhs_sections(text):
        total += 1
        mv = VARIANT_RE.search(body)
        if not mv or mv.group(1) != "NT":
            continue
        ms = SEAT_COUNT_RE.search(body)
        seat_ok = False
        if ms:
            seat_ok = int(ms.group(1)) == 6
        else:
            seat_ok = count_stacks(body) == 6
        if not seat_ok:
            continue
        kept += 1
        kept_chunks.append(f"[{kept}]" + body.rstrip() + "\n\n")
    if kept_chunks:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(kept_chunks), encoding="utf-8")
    return kept, total


def main() -> None:
    t0 = time.time()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    stats = {
        "pluribus_files_kept": 0,
        "pluribus_files_total": 0,
        "wsop_files_kept": 0,
        "wsop_files_total": 0,
        "root_files_kept": 0,
        "root_files_total": 0,
        "handhq_phhs_files": 0,
        "handhq_hands_kept": 0,
        "handhq_hands_total": 0,
    }

    # Root-level sample .phh files
    for src in sorted(SRC_ROOT.glob("*.phh")):
        stats["root_files_total"] += 1
        rel = src.relative_to(SRC_ROOT)
        out = OUT_ROOT / rel
        if process_single_phh(src, out):
            stats["root_files_kept"] += 1

    # Pluribus (all NT 6-handed by design; still validate)
    pluribus_dir = SRC_ROOT / "pluribus"
    if pluribus_dir.exists():
        for src in pluribus_dir.rglob("*.phh"):
            stats["pluribus_files_total"] += 1
            rel = src.relative_to(SRC_ROOT)
            out = OUT_ROOT / rel
            if process_single_phh(src, out):
                stats["pluribus_files_kept"] += 1
            if stats["pluribus_files_total"] % 2000 == 0:
                print(f"  pluribus scanned {stats['pluribus_files_total']}...", flush=True)

    # WSOP (mixed-game; filter)
    wsop_dir = SRC_ROOT / "wsop"
    if wsop_dir.exists():
        for src in wsop_dir.rglob("*.phh"):
            stats["wsop_files_total"] += 1
            rel = src.relative_to(SRC_ROOT)
            out = OUT_ROOT / rel
            if process_single_phh(src, out):
                stats["wsop_files_kept"] += 1

    # HandHQ (.phhs bundles)
    handhq_dir = SRC_ROOT / "handhq"
    if handhq_dir.exists():
        phhs_list = list(handhq_dir.rglob("*.phhs"))
        print(f"HandHQ: {len(phhs_list)} .phhs bundles to scan...", flush=True)
        for i, src in enumerate(phhs_list, 1):
            rel = src.relative_to(SRC_ROOT)
            out = OUT_ROOT / rel
            kept, total = process_phhs(src, out)
            stats["handhq_hands_kept"] += kept
            stats["handhq_hands_total"] += total
            if kept > 0:
                stats["handhq_phhs_files"] += 1
            if i % 200 == 0 or i == len(phhs_list):
                dt = time.time() - t0
                print(
                    f"  handhq {i}/{len(phhs_list)} files | kept={stats['handhq_hands_kept']:,}"
                    f" / total={stats['handhq_hands_total']:,} | {dt:.1f}s",
                    flush=True,
                )

    dt = time.time() - t0
    print("\n=== FILTER SUMMARY (6-max NLHE) ===")
    print(f"Root sample .phh     kept/total: {stats['root_files_kept']}/{stats['root_files_total']}")
    print(f"Pluribus .phh        kept/total: {stats['pluribus_files_kept']}/{stats['pluribus_files_total']}")
    print(f"WSOP .phh            kept/total: {stats['wsop_files_kept']}/{stats['wsop_files_total']}")
    print(
        f"HandHQ hands kept/total: {stats['handhq_hands_kept']:,}/{stats['handhq_hands_total']:,}"
        f"  ({stats['handhq_phhs_files']} non-empty .phhs files)"
    )
    total_kept = (
        stats["root_files_kept"]
        + stats["pluribus_files_kept"]
        + stats["wsop_files_kept"]
        + stats["handhq_hands_kept"]
    )
    print(f"TOTAL 6-max NLHE hands: {total_kept:,}")
    print(f"Output directory: {OUT_ROOT}")
    print(f"Elapsed: {dt:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
