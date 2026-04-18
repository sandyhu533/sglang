#!/usr/bin/env python3
"""Parse Phase 1B ablation logs for SGLang issue #22831.

Extracts from each <label>.server.log:
  - batch_is_full set/clear event counts and timestamps
  - Duration the flag stayed True (per episode + cumulative)
  - Max consecutive iterations with flag=True
  - Retraction count ("KV cache pool is full" warnings)
  - Entry-to-first-admit latency

Extracts from each <label>.repro.log:
  - small p50 / p99 TTFT
  - large p50 / p99 TTFT

Prints a markdown table comparing all configs.

Usage:  python3 parse_logs.py <results_dir>
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

TTFT_DEBUG_RE = re.compile(
    r"\[TTFT_DEBUG\] ev=(?P<ev>\w+) t=(?P<t>[\d.]+)"
)
RETRACT_RE = re.compile(r"(?:retract|Retract|KV cache pool is full)")
SMALL_P99_RE = re.compile(r"[Ss]mall\s+p99[:\s=]+(?P<v>[\d.]+)\s*ms", re.IGNORECASE)
LARGE_P99_RE = re.compile(r"[Ll]arge\s+p99[:\s=]+(?P<v>[\d.]+)\s*ms", re.IGNORECASE)
SMALL_P50_RE = re.compile(r"[Ss]mall.*?[Bb]urst\s+p50[:\s=]+(?P<v>[\d.]+)\s*ms")
BASELINE_RE = re.compile(r"[Bb]aseline[:\s]+(?P<v>[\d.]+)\s*ms")


@dataclass
class ServerStats:
    set_events: List[Tuple[float, str]] = field(default_factory=list)   # (t, kind)
    clear_events: List[Tuple[float, str]] = field(default_factory=list) # (t, kind)
    entries: List[float] = field(default_factory=list)
    early_returns: List[float] = field(default_factory=list)
    retractions: int = 0

    @property
    def total_iters(self) -> int:
        return len(self.entries)

    @property
    def early_return_frac(self) -> float:
        return len(self.early_returns) / max(1, self.total_iters)

    def flag_true_episodes(self) -> List[float]:
        """Return list of durations (sec) that batch_is_full stayed True."""
        # Merge set+clear events sorted by time. Interval starts at a SET,
        # ends at the next CLEAR. If no clear after a set, use last_t.
        events = sorted(
            [(t, "SET") for t, _ in self.set_events]
            + [(t, "CLEAR") for t, _ in self.clear_events],
            key=lambda x: x[0],
        )
        episodes: List[float] = []
        cur_set: Optional[float] = None
        for t, kind in events:
            if kind == "SET" and cur_set is None:
                cur_set = t
            elif kind == "CLEAR" and cur_set is not None:
                episodes.append(t - cur_set)
                cur_set = None
        if cur_set is not None and events:
            episodes.append(events[-1][0] - cur_set)
        return episodes


def parse_server_log(path: Path) -> ServerStats:
    s = ServerStats()
    for line in path.read_text(errors="ignore").splitlines():
        if "[TTFT_DEBUG]" in line:
            m = TTFT_DEBUG_RE.search(line)
            if not m:
                continue
            ev = m.group("ev")
            t = float(m.group("t"))
            if ev == "ENTRY":
                s.entries.append(t)
            elif ev == "EARLY_RETURN":
                s.early_returns.append(t)
            elif ev.startswith("SET_"):
                s.set_events.append((t, ev))
            elif ev.startswith("CLEAR_"):
                s.clear_events.append((t, ev))
        elif RETRACT_RE.search(line):
            s.retractions += 1
    return s


@dataclass
class TTFTStats:
    baseline_ms: Optional[float] = None
    small_p50_ms: Optional[float] = None
    small_p99_ms: Optional[float] = None
    large_p99_ms: Optional[float] = None


def parse_repro_log(path: Path) -> TTFTStats:
    t = TTFTStats()
    text = path.read_text(errors="ignore")
    if m := BASELINE_RE.search(text):
        t.baseline_ms = float(m.group("v"))
    if m := SMALL_P50_RE.search(text):
        t.small_p50_ms = float(m.group("v"))
    # Take the LAST p99 match (final aggregate line in repro output)
    small_matches = SMALL_P99_RE.findall(text)
    if small_matches:
        t.small_p99_ms = float(small_matches[-1])
    large_matches = LARGE_P99_RE.findall(text)
    if large_matches:
        t.large_p99_ms = float(large_matches[-1])
    return t


def fmt_ms(v: Optional[float]) -> str:
    return f"{v:>8.0f}" if v is not None else "      —"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    results_dir = Path(sys.argv[1])
    if not results_dir.is_dir():
        print(f"error: {results_dir} is not a directory", file=sys.stderr)
        return 1

    labels = sorted(
        {p.stem.rsplit(".", 1)[0] for p in results_dir.glob("*.server.log")}
    )
    if not labels:
        print(f"no *.server.log in {results_dir}", file=sys.stderr)
        return 1

    rows = []
    for label in labels:
        srv = parse_server_log(results_dir / f"{label}.server.log")
        repro_path = results_dir / f"{label}.repro.log"
        ttft = parse_repro_log(repro_path) if repro_path.exists() else TTFTStats()
        episodes = srv.flag_true_episodes()
        max_ep = max(episodes) if episodes else 0.0
        sum_ep = sum(episodes)
        rows.append((label, srv, ttft, max_ep, sum_ep))

    print()
    print("| config | small p99 ms | large p99 ms | baseline ms | early-return % | flag-true episodes | max episode (s) | total flag-true (s) | retract # |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, srv, ttft, max_ep, sum_ep in rows:
        print(
            f"| {label} "
            f"| {fmt_ms(ttft.small_p99_ms)} "
            f"| {fmt_ms(ttft.large_p99_ms)} "
            f"| {fmt_ms(ttft.baseline_ms)} "
            f"| {srv.early_return_frac * 100:>6.1f} "
            f"| {len(srv.flag_true_episodes()):>6} "
            f"| {max_ep:>8.2f} "
            f"| {sum_ep:>8.2f} "
            f"| {srv.retractions:>6} |"
        )

    print()
    print("Interpretation guide:")
    print("  - H1 confirmed if A has large 'max episode' (≥ several seconds) AND")
    print("    F1_smoke_fix has small p99 << A small p99.")
    print("  - If F1 helps but B (no mixed chunk) does not, H2 is secondary.")
    print("  - High retract # in A suggests H3/H4 is also contributing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
