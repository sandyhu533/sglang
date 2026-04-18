#!/usr/bin/env python3
"""Anchor-based in-place patcher for scheduler.py (issue #22831 ablation).

Why not unified diff: line drift across commits breaks `git apply`.
This script locates anchor strings and inserts new code next to them.
Idempotent: detects existing markers and refuses to double-patch.

Usage:
    python3 ttft_ablation/patches/apply_patches.py              # apply
    python3 ttft_ablation/patches/apply_patches.py --revert     # revert

Both env vars must be set on server launch:
    SGLANG_TTFT_DEBUG=1       -> emits [TTFT_DEBUG] log lines
    SGLANG_TTFT_SMOKE_FIX=1   -> enables F1 smoke fix (reset sticky flag)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TARGET = REPO / "python/sglang/srt/managers/scheduler.py"

MARK_START = "# >>> TTFT_ABLATION_PATCH START"
MARK_END = "# <<< TTFT_ABLATION_PATCH END"


def wrap(block: str, tag: str) -> str:
    return f"{MARK_START} [{tag}]\n{block.rstrip()}\n{MARK_END} [{tag}]\n"


# ----------------------------------------------------------------------------
# PATCH 1 — smoke fix F1: reset batch_is_full at top of _get_new_batch_prefill_raw.
# Anchor: line just before the fast-path read.
# ----------------------------------------------------------------------------
ANCHOR_SMOKE_FIX = (
    "        if (\n"
    "            self.running_batch.batch_is_full or len(self.waiting_queue) == 0\n"
    "        ) and self.chunked_req is None:\n"
    "            return None\n"
)
INSERT_SMOKE_FIX = wrap(
    (
        "        # [TTFT SMOKE FIX F1] Demote sticky cross-iteration flag to per-iter\n"
        '        if os.environ.get("SGLANG_TTFT_SMOKE_FIX", "0") == "1":\n'
        "            self.running_batch.batch_is_full = False\n"
    ),
    "smoke_fix",
)

# ----------------------------------------------------------------------------
# PATCH 2 — debug log at ENTRY + EARLY_RETURN of _get_new_batch_prefill_raw.
# Same anchor as above; we insert BOTH before the `if (...) return None` block.
# ----------------------------------------------------------------------------
INSERT_DEBUG_ENTRY = wrap(
    (
        "        # [TTFT DEBUG] entry log\n"
        '        if os.environ.get("SGLANG_TTFT_DEBUG", "0") == "1":\n'
        "            logger.info(\n"
        '                f"[TTFT_DEBUG] ev=ENTRY t={time.perf_counter():.6f} "\n'
        '                f"running_bs={len(self.running_batch.reqs)} "\n'
        '                f"waiting={len(self.waiting_queue)} "\n'
        '                f"batch_is_full={self.running_batch.batch_is_full} "\n'
        '                f"chunked_req={self.chunked_req is not None} "\n'
        '                f"kv_avail={self.token_to_kv_pool_allocator.available_size()}"\n'
        "            )\n"
    ),
    "debug_entry",
)

# Early-return log: inject inside the `if (...) return None` branch.
ANCHOR_EARLY_RETURN = (
    "        if (\n"
    "            self.running_batch.batch_is_full or len(self.waiting_queue) == 0\n"
    "        ) and self.chunked_req is None:\n"
    "            return None\n"
)
REPLACE_EARLY_RETURN = (
    "        if (\n"
    "            self.running_batch.batch_is_full or len(self.waiting_queue) == 0\n"
    "        ) and self.chunked_req is None:\n"
    f"{MARK_START} [debug_early_return]\n"
    '            if os.environ.get("SGLANG_TTFT_DEBUG", "0") == "1":\n'
    "                logger.info(\n"
    '                    f"[TTFT_DEBUG] ev=EARLY_RETURN t={time.perf_counter():.6f} "\n'
    '                    f"batch_is_full={self.running_batch.batch_is_full} "\n'
    '                    f"waiting={len(self.waiting_queue)}"\n'
    "                )\n"
    f"{MARK_END} [debug_early_return]\n"
    "            return None\n"
)

# ----------------------------------------------------------------------------
# PATCH 3 — SET_A log (allocatable==0 path).
# ----------------------------------------------------------------------------
ANCHOR_SET_A = (
    "        if (\n"
    "            self.get_num_allocatable_reqs(running_bs) <= 0\n"
    "            and self.chunked_req is None\n"
    "            and not self.enable_priority_preemption\n"
    "        ):\n"
    "            self.running_batch.batch_is_full = True\n"
    "            return None\n"
)
REPLACE_SET_A = (
    "        if (\n"
    "            self.get_num_allocatable_reqs(running_bs) <= 0\n"
    "            and self.chunked_req is None\n"
    "            and not self.enable_priority_preemption\n"
    "        ):\n"
    "            self.running_batch.batch_is_full = True\n"
    f"{MARK_START} [debug_set_a]\n"
    '            if os.environ.get("SGLANG_TTFT_DEBUG", "0") == "1":\n'
    "                logger.info(\n"
    '                    f"[TTFT_DEBUG] ev=SET_A_ALLOCATABLE_ZERO "\n'
    '                    f"t={time.perf_counter():.6f} running_bs={running_bs}"\n'
    "                )\n"
    f"{MARK_END} [debug_set_a]\n"
    "            return None\n"
)

# ----------------------------------------------------------------------------
# PATCH 4 — SET_B log (can_run_list saturated).
# ----------------------------------------------------------------------------
ANCHOR_SET_B = (
    "            running_bs = len(self.running_batch.reqs)\n"
    "            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):\n"
    "                self.running_batch.batch_is_full = True\n"
)
REPLACE_SET_B = (
    "            running_bs = len(self.running_batch.reqs)\n"
    "            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):\n"
    "                self.running_batch.batch_is_full = True\n"
    f"{MARK_START} [debug_set_b]\n"
    '                if os.environ.get("SGLANG_TTFT_DEBUG", "0") == "1":\n'
    "                    logger.info(\n"
    '                        f"[TTFT_DEBUG] ev=SET_B_CAN_RUN_SATURATED "\n'
    '                        f"t={time.perf_counter():.6f} running_bs={running_bs} "\n'
    '                        f"can_run={len(adder.can_run_list)} "\n'
    '                        f"req_input_len={len(req.origin_input_ids)}"\n'
    "                    )\n"
    f"{MARK_END} [debug_set_b]\n"
)

# ----------------------------------------------------------------------------
# PATCH 5 — CLEAR_D log (batch shrunk in update_running_batch).
# ----------------------------------------------------------------------------
ANCHOR_CLEAR_D = (
    "        if batch.batch_size() < initial_bs:\n"
    "            batch.batch_is_full = False\n"
)
REPLACE_CLEAR_D = (
    "        if batch.batch_size() < initial_bs:\n"
    "            batch.batch_is_full = False\n"
    f"{MARK_START} [debug_clear_d]\n"
    '            if os.environ.get("SGLANG_TTFT_DEBUG", "0") == "1":\n'
    "                logger.info(\n"
    '                    f"[TTFT_DEBUG] ev=CLEAR_D_BATCH_SHRUNK "\n'
    '                    f"t={time.perf_counter():.6f} initial_bs={initial_bs} "\n'
    '                    f"new_bs={batch.batch_size()}"\n'
    "                )\n"
    f"{MARK_END} [debug_clear_d]\n"
)


PATCHES = [
    # Must apply smoke fix BEFORE early-return replace (same anchor).
    ("smoke_fix_insert_before", ANCHOR_SMOKE_FIX, INSERT_SMOKE_FIX + ANCHOR_SMOKE_FIX),
    ("debug_entry_insert_before", ANCHOR_SMOKE_FIX, INSERT_DEBUG_ENTRY + ANCHOR_SMOKE_FIX),
    ("debug_early_return", ANCHOR_EARLY_RETURN, REPLACE_EARLY_RETURN),
    ("debug_set_a", ANCHOR_SET_A, REPLACE_SET_A),
    ("debug_set_b", ANCHOR_SET_B, REPLACE_SET_B),
    ("debug_clear_d", ANCHOR_CLEAR_D, REPLACE_CLEAR_D),
]


def apply() -> int:
    src = TARGET.read_text()
    if MARK_START in src:
        print("refusing to re-apply: MARK_START already present. Run --revert first.")
        return 1

    for name, anchor, replacement in PATCHES:
        count = src.count(anchor)
        if count == 0:
            print(f"!! [{name}] anchor not found — aborting, no changes written")
            return 2
        if count > 1:
            print(f"!! [{name}] anchor matched {count} times — aborting (ambiguous)")
            return 2
        src = src.replace(anchor, replacement, 1)
        print(f"ok [{name}]")

    TARGET.write_text(src)
    print(f"\npatched: {TARGET}")
    print("\nenable at launch:")
    print("  SGLANG_TTFT_DEBUG=1 SGLANG_TTFT_SMOKE_FIX=1 python -m sglang.launch_server ...")
    return 0


def revert() -> int:
    """Strip every MARK_START..MARK_END block (inclusive)."""
    src = TARGET.read_text()
    if MARK_START not in src:
        print("no markers found — nothing to revert")
        return 0

    lines = src.splitlines(keepends=True)
    out: list[str] = []
    skip = False
    removed = 0
    for ln in lines:
        if MARK_START in ln:
            skip = True
            removed += 1
            continue
        if MARK_END in ln:
            skip = False
            continue
        if skip:
            removed += 1
            continue
        out.append(ln)
    TARGET.write_text("".join(out))
    print(f"reverted: {removed} lines removed from {TARGET}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()
    return revert() if args.revert else apply()


if __name__ == "__main__":
    sys.exit(main())
