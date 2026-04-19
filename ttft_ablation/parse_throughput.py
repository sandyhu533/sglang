#!/usr/bin/env python3
"""Aggregate bench_serving JSONs from a throughput sweep into a markdown table.

Usage:  python3 parse_throughput.py results/throughput/sweep [--md OUT.md]

Expects one JSON per cell named `<label>_c<conc>_s<seed>.json`. Reports
mean ± std across seeds for each (label, concurrency) cell.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as stats
from pathlib import Path

# Metrics to surface. Order matters — first one is "primary" headline.
METRICS = [
    ("output_throughput", "out tok/s"),
    ("input_throughput", "in tok/s"),
    ("mean_ttft_ms", "TTFT mean"),
    ("p99_ttft_ms", "TTFT p99"),
    ("median_itl_ms", "ITL p50"),
    ("p99_itl_ms", "ITL p99"),
]

CELL_RE = re.compile(r"^(?P<label>[^_]+)_c(?P<conc>\d+)_s(?P<seed>\d+)\.json$")


def fmt(mu: float, sd: float, metric: str) -> str:
    if "throughput" in metric:
        return f"{mu:.0f} ± {sd:.0f}"
    return f"{mu:.0f} ± {sd:.0f}"


def aggregate(outdir: Path) -> dict:
    """{label: {conc: {metric: [seed_values]}}}"""
    table: dict = {}
    for f in sorted(outdir.glob("*.json")):
        m = CELL_RE.match(f.name)
        if not m:
            continue
        label = m["label"]
        conc = int(m["conc"])
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            print(f"!! skipping malformed JSON: {f.name}")
            continue
        cell = table.setdefault(label, {}).setdefault(conc, {})
        for key, _ in METRICS:
            if key in d and d[key] is not None:
                cell.setdefault(key, []).append(d[key])
    return table


def render(table: dict) -> str:
    out = []
    labels = sorted(table.keys())
    concs = sorted({c for lab in labels for c in table[lab].keys()})

    for metric, header in METRICS:
        out.append(f"\n### {header}\n")
        # header row
        cols = ["concurrency"] + labels
        out.append("| " + " | ".join(cols) + " |")
        out.append("|" + "|".join(["---:"] * len(cols)) + "|")
        for c in concs:
            row = [str(c)]
            for lab in labels:
                vals = table.get(lab, {}).get(c, {}).get(metric, [])
                if not vals:
                    row.append("—")
                    continue
                mu = stats.mean(vals)
                sd = stats.pstdev(vals) if len(vals) > 1 else 0.0
                row.append(fmt(mu, sd, metric))
            out.append("| " + " | ".join(row) + " |")

        # baseline-relative deltas (vs A) — compact, one row per label
        if "A" in labels:
            out.append("")
            out.append(f"_Δ vs A:_")
            delta_cols = ["concurrency"] + [l for l in labels if l != "A"]
            out.append("| " + " | ".join(delta_cols) + " |")
            out.append("|" + "|".join(["---:"] * len(delta_cols)) + "|")
            for c in concs:
                base = table.get("A", {}).get(c, {}).get(metric, [])
                if not base:
                    continue
                base_mu = stats.mean(base)
                row = [str(c)]
                for lab in delta_cols[1:]:
                    vals = table.get(lab, {}).get(c, {}).get(metric, [])
                    if not vals or base_mu == 0:
                        row.append("—")
                        continue
                    mu = stats.mean(vals)
                    pct = (mu - base_mu) / base_mu * 100
                    sign = "+" if pct >= 0 else ""
                    row.append(f"{sign}{pct:.1f}%")
                out.append("| " + " | ".join(row) + " |")

    # Coverage report
    out.append("\n### Coverage\n")
    out.append("| label | concurrency | seeds |")
    out.append("|---|---:|---:|")
    for lab in labels:
        for c in concs:
            vals = table.get(lab, {}).get(c, {}).get(METRICS[0][0], [])
            out.append(f"| {lab} | {c} | {len(vals)} |")

    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--md", type=Path, default=None,
                    help="if given, write rendered markdown here")
    args = ap.parse_args()

    table = aggregate(args.outdir)
    md = render(table)
    print(md)
    if args.md:
        args.md.write_text(md + "\n")
        print(f"\nwrote {args.md}", file=__import__("sys").stderr)


if __name__ == "__main__":
    main()
