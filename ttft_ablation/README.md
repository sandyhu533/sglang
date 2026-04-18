# TTFT Ablation — Issue #22831 Phase 1B

Work directory, not committed. Run on the GPU box after cloning the branch.

## One-time setup (GPU box)

```bash
cd /path/to/sglang
git checkout scheduler/ttft-asymmetric-prefill
pip install -e "python[all]"

# Apply ablation patches (anchor-based in-place edit, survives line drift).
# Both debug-log and smoke-fix instrumentation are env-gated, so the patched
# file is a no-op unless SGLANG_TTFT_DEBUG or SGLANG_TTFT_SMOKE_FIX is set.
python3 ttft_ablation/patches/apply_patches.py

chmod +x ttft_ablation/run_ablation.sh
```

To remove instrumentation cleanly:
```bash
python3 ttft_ablation/patches/apply_patches.py --revert
```

## Run all 6 configs

```bash
cd ttft_ablation
./run_ablation.sh all
# ~15-25 min total on a single 4090
```

Individual:
```bash
./run_ablation.sh A_baseline
./run_ablation.sh F1_smoke_fix
```

## Collect evidence

```bash
python3 parse_logs.py results/<timestamp>/
```

Prints a markdown table of small/large p99 TTFT + flag-true episode stats per config. Paste directly into the issue comment or PR description.

## Config map

**Updated 2026-04-18**: A / F1 / C already run, all show p99 ≈ 27-30s. This
refutes H1 (sticky flag) and H7 (decode super-linear / KV pressure). Log
analysis points to **H11: FCFS HOL blocking + early-break on NO_TOKEN** —
admission loop breaks at the first large req that can't fit, never tries
smaller reqs queued behind it. E2 is the decisive experiment for H11.

| Label | Tests | Expected outcome |
|---|---|---|
| A_baseline | reporter's config | small p99 ≈ 22-27s (confirms repro) — ✅ DONE |
| F1_smoke_fix | SGLANG_TTFT_SMOKE_FIX=1, clears flag at ENTRY | no change (flag not root cause) — ✅ DONE, H1 refuted |
| C_mrr8 | `--max-running-requests 8` | no change (not KV bandwidth bound) — ✅ DONE, H7 refuted |
| B_no_mixed_chunk | `--disable-mixed-chunk` | probably no change |
| D_chunk32k | `--chunked-prefill-size 32768` | large req fits in one chunk, sidesteps chunking budget contention; may help as workaround |
| E_priority | `--enable-priority-scheduling` | no change (clients don't set priority) |
| **E2_hol_fix** | **SGLANG_TTFT_HOL_FIX=1** | **small p99 drops to < 5s → H11 confirmed + candidate fix validated** |

## Skipping already-run configs

```bash
# comma-separated list of labels to skip on a fresh OUTDIR
SKIP="A_baseline,F1_smoke_fix,C_mrr8" ./run_ablation.sh all

# or: re-point OUTDIR at a prior run — any config with an existing
# *.repro.log in OUTDIR is auto-skipped
OUTDIR=./results/20260418-1830 ./run_ablation.sh all
```

## Cleanup

```bash
python3 ttft_ablation/patches/apply_patches.py --revert
# or
git checkout -- python/sglang/srt/managers/scheduler.py
```
