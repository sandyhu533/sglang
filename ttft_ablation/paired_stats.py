#!/usr/bin/env python3
"""Paired-seed analysis for A vs E3 throughput sweep.

Computes per-cell: mean, std, paired Δ, paired t-test, sign split.
Also flags anomalous (likely-HOL-hit) baseline seeds by JSON-only
anomaly detection (no server-log dependency), since interim runs
overwrote some server logs.

Usage:  python3 paired_stats.py results/throughput/sweep
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as stats
from pathlib import Path

from scipy import stats as sstats  # type: ignore

CELL_RE = re.compile(r"^(?P<label>[^_]+)_c(?P<conc>\d+)_s(?P<seed>\d+)\.json$")

METRICS = [
    ("output_throughput", "output tok/s", False),  # (key, label, lower_is_better)
    ("input_throughput", "input tok/s", False),
    ("mean_ttft_ms", "mean TTFT (ms)", True),
    ("median_ttft_ms", "median TTFT (ms)", True),
    ("p99_ttft_ms", "p99 TTFT (ms)", True),
    ("median_itl_ms", "median ITL (ms)", True),
    ("p99_itl_ms", "p99 ITL (ms)", True),
    ("duration", "bench duration (s)", True),
]


def load(outdir: Path) -> dict:
    """{label: {conc: {seed: row_dict}}}"""
    table: dict = {}
    for f in sorted(outdir.glob("*.json")):
        m = CELL_RE.match(f.name)
        if not m:
            continue
        label, conc, seed = m["label"], int(m["conc"]), int(m["seed"])
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        table.setdefault(label, {}).setdefault(conc, {})[seed] = d
    return table


def paired_analysis(a_vals, e3_vals, lower_better):
    """Returns dict with μ_A, μ_E3, Δ%, paired t, p, sign_ratio, n_seeds."""
    paired = list(zip(a_vals, e3_vals))
    diffs = [e - a for a, e in paired]  # E3 - A (so negative = E3 lower)
    mu_a = stats.mean(a_vals)
    mu_e3 = stats.mean(e3_vals)
    delta_pct = (mu_e3 - mu_a) / mu_a * 100 if mu_a else 0.0

    t, p = sstats.ttest_rel(e3_vals, a_vals)
    if lower_better:
        improvements = sum(1 for d in diffs if d < 0)
    else:
        improvements = sum(1 for d in diffs if d > 0)
    n = len(paired)
    return {
        "mu_a": mu_a, "sd_a": stats.pstdev(a_vals),
        "mu_e3": mu_e3, "sd_e3": stats.pstdev(e3_vals),
        "delta_pct": delta_pct,
        "t": t, "p": p,
        "improvements": improvements, "n": n,
    }


def fmt_delta(r, lower_better):
    arrow = "↓" if r["delta_pct"] < 0 else "↑"
    d = r["delta_pct"]
    s = f"{d:+.1f}%"
    p = r["p"]
    ptag = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
    n = r["n"]
    imp = r["improvements"]
    # bold if significant (p<0.05) and in the favourable direction
    favorable = (d < 0) if lower_better else (d > 0)
    body = f"{s} ({ptag}, {imp}/{n}{arrow})"
    if p < 0.05 and favorable:
        return f"**{body}**"
    return body


def anomaly_flag(seed_vals, seed_idx_map):
    """Return list of (seed, value) where value is > μ + 2σ (TTFT p99)."""
    vals = list(seed_vals.values())
    if len(vals) < 3:
        return []
    mu = stats.mean(vals)
    sd = stats.pstdev(vals)
    flagged = []
    for seed, v in seed_vals.items():
        if sd > 0 and (v - mu) / sd > 2.0:
            flagged.append((seed, v, mu, sd))
    return flagged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", type=Path)
    args = ap.parse_args()
    table = load(args.outdir)

    a = table.get("A", {})
    e3 = table.get("E3", {})
    concs = sorted(set(a.keys()) & set(e3.keys()))

    print(f"# Paired analysis: A vs E3 (n per cell below)\n")
    print(f"(Δ% reports E3 relative to A; p is paired t-test; `X/N↓` = # seeds where E3 is better)\n")

    for key, label, lower in METRICS:
        print(f"## {label}\n")
        print("| concurrency | n | μ_A | μ_E3 | Δ% | paired t | p | sign |")
        print("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for c in concs:
            seeds = sorted(set(a[c].keys()) & set(e3[c].keys()))
            a_vals = [a[c][s].get(key) for s in seeds]
            e3_vals = [e3[c][s].get(key) for s in seeds]
            if any(v is None for v in a_vals + e3_vals):
                continue
            r = paired_analysis(a_vals, e3_vals, lower)
            print(f"| {c} | {r['n']} | {r['mu_a']:.1f} | {r['mu_e3']:.1f} | "
                  f"{r['delta_pct']:+.1f}% | {r['t']:+.2f} | "
                  f"{r['p']:.3f} | {r['improvements']}/{r['n']} |")
        print()

    # Anomaly detection: baseline p99_ttft_ms outliers per (conc)
    print("## Baseline (A) anomaly detection: p99 TTFT > μ + 2σ per concurrency\n")
    print("These are candidate HOL-blocking events (no server-log required).\n")
    for c in concs:
        p99_ttft = {s: a[c][s]["p99_ttft_ms"] for s in a[c]}
        flagged = anomaly_flag(p99_ttft, None)
        if not flagged:
            print(f"- c={c}: no outliers (n={len(p99_ttft)})")
            continue
        print(f"- **c={c}**: {len(flagged)} outlier(s) / {len(p99_ttft)} seeds")
        for seed, v, mu, sd in flagged:
            print(f"  - seed={seed}: p99_ttft={v:.0f} ms  (μ={mu:.0f}, σ={sd:.0f}, z={(v-mu)/sd:.1f})")
            # Also show that seed's duration as secondary evidence
            dur = a[c][seed].get("duration", 0)
            all_dur = [a[c][s].get("duration", 0) for s in a[c]]
            dur_mu = stats.mean(all_dur)
            print(f"    duration={dur:.1f}s (avg={dur_mu:.1f}s)")


if __name__ == "__main__":
    main()
