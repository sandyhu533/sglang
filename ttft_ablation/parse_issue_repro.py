#!/usr/bin/env python3
"""Paired stats for the issue-repro sweep (#22831).

Reads <OUTDIR>/{A,E3}_s<seed>.repro.log produced by run_issue_repro.sh,
extracts aggregate metrics (small p99, small mean, large p99, large mean),
and reports paired A vs E3 differences with Wilcoxon signed-rank p-values.

Usage:  python3 parse_issue_repro.py <OUTDIR>
"""

from __future__ import annotations

import math
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from scipy.stats import wilcoxon
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


METRICS = [
    ("small_p99_ms", re.compile(r"Small p99:\s*(?P<v>[\d.]+)\s*ms")),
    ("small_p50_ms", re.compile(r"Small\s+burst\s+p50:\s*(?P<v>[\d.]+)\s*ms")),
    ("small_mean_ms", re.compile(r"Small mean:\s*(?P<v>[\d.]+)\s*ms")),
    ("large_p99_ms", re.compile(r"Large p99:\s*(?P<v>[\d.]+)\s*ms")),
    ("large_mean_ms", re.compile(r"Large mean:\s*(?P<v>[\d.]+)\s*ms")),
    ("output_tok_s", re.compile(r"Output tok/s:\s*(?P<v>[\d.]+)")),
]

LOWER_IS_BETTER = {
    "small_p99_ms": True, "small_p50_ms": True, "small_mean_ms": True,
    "large_p99_ms": True, "large_mean_ms": True,
    "output_tok_s": False,
}


def parse_log(path: Path) -> Dict[str, float]:
    text = path.read_text(errors="replace")
    out: Dict[str, float] = {}
    for key, rx in METRICS:
        m = rx.search(text)
        if m:
            out[key] = float(m.group("v"))
    return out


def collect(outdir: Path, label: str) -> Dict[int, Dict[str, float]]:
    by_seed: Dict[int, Dict[str, float]] = {}
    for p in sorted(outdir.glob(f"{label}_s*.repro.log")):
        m = re.search(r"_s(\d+)\.repro\.log$", p.name)
        if not m:
            continue
        seed = int(m.group(1))
        by_seed[seed] = parse_log(p)
    return by_seed


def paired_row(
    metric: str, a_by_seed: Dict[int, Dict[str, float]],
    e_by_seed: Dict[int, Dict[str, float]],
) -> Optional[Tuple[str, float, float, float, float, float, int, int, Optional[float]]]:
    """Return (metric, mu_a, sd_a, mu_e, sd_e, pct_delta, improvements, n, p)."""
    seeds = sorted(set(a_by_seed) & set(e_by_seed))
    a_vals, e_vals = [], []
    for s in seeds:
        if metric in a_by_seed[s] and metric in e_by_seed[s]:
            a_vals.append(a_by_seed[s][metric])
            e_vals.append(e_by_seed[s][metric])
    if len(a_vals) < 2:
        return None

    mu_a = statistics.mean(a_vals)
    mu_e = statistics.mean(e_vals)
    sd_a = statistics.stdev(a_vals) if len(a_vals) > 1 else 0.0
    sd_e = statistics.stdev(e_vals) if len(e_vals) > 1 else 0.0
    pct = 100.0 * (mu_e - mu_a) / mu_a if mu_a > 0 else 0.0

    lower_better = LOWER_IS_BETTER.get(metric, True)
    improvements = sum(
        1 for a, e in zip(a_vals, e_vals)
        if ((e < a) if lower_better else (e > a))
    )

    p = None
    if HAVE_SCIPY and len(a_vals) >= 2:
        diffs = [e - a for a, e in zip(a_vals, e_vals)]
        if any(abs(d) > 1e-9 for d in diffs):
            try:
                res = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
                p = float(res.pvalue)
            except ValueError:
                p = None

    return (metric, mu_a, sd_a, mu_e, sd_e, pct, improvements, len(a_vals), p)


def fmt_ms(v: float) -> str:
    if v >= 1000:
        return f"{v/1000:.2f} s"
    return f"{v:.0f} ms"


def fmt_val(metric: str, v: float) -> str:
    if metric == "output_tok_s":
        return f"{v:.0f} tok/s"
    return fmt_ms(v)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 1

    outdir = Path(sys.argv[1]).resolve()
    a = collect(outdir, "A")
    e = collect(outdir, "E3")

    if not a or not e:
        print(f"No A or E3 logs found under {outdir}", file=sys.stderr)
        return 1

    print(f"# Issue-repro paired stats — {outdir}")
    print(f"# A seeds: {sorted(a)}   E3 seeds: {sorted(e)}")
    print()

    # Per-seed table (raw).
    seeds = sorted(set(a) & set(e))
    print("## Per-seed raw")
    print()
    print("| seed | small_p99 A | small_p99 E3 | Δ | small_mean A | small_mean E3 | large_p99 A | large_p99 E3 |")
    print("| ---: | --- | --- | --- | --- | --- | --- | --- |")
    for s in seeds:
        sa = a[s]; se = e[s]
        def g(d, k): return d.get(k, float("nan"))
        sp99_a = g(sa, "small_p99_ms"); sp99_e = g(se, "small_p99_ms")
        delta = ""
        if sp99_a == sp99_a and sp99_e == sp99_e and sp99_a > 0:  # not NaN
            delta = f"{(sp99_e - sp99_a) / sp99_a * 100:+.1f}%"
        print(
            f"| {s} | {fmt_ms(sp99_a)} | {fmt_ms(sp99_e)} | {delta} | "
            f"{fmt_ms(g(sa, 'small_mean_ms'))} | {fmt_ms(g(se, 'small_mean_ms'))} | "
            f"{fmt_ms(g(sa, 'large_p99_ms'))} | {fmt_ms(g(se, 'large_p99_ms'))} |"
        )
    print()

    # Aggregate paired table.
    print("## Paired aggregate (Wilcoxon signed-rank)")
    print()
    print("| metric         | A (μ±σ)        | E3 (μ±σ)        | Δ%         | ↓ / n | p (Wilcoxon) |")
    print("| -------------- | -------------- | --------------- | ---------: | ----: | -----------: |")
    for metric, _ in METRICS:
        row = paired_row(metric, a, e)
        if row is None:
            continue
        (_, mu_a, sd_a, mu_e, sd_e, pct, imp, n, p) = row
        p_fmt = f"{p:.3f}" if p is not None else "—"
        sp_label = metric.replace("_ms", "").replace("_", " ")
        print(
            f"| {sp_label:<14} | {fmt_val(metric, mu_a):>12} ± {fmt_val(metric, sd_a):>8} | "
            f"{fmt_val(metric, mu_e):>12} ± {fmt_val(metric, sd_e):>8} | "
            f"{pct:+6.1f}% | {imp}/{n} | {p_fmt} |"
        )
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
