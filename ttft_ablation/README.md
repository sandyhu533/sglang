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

| Label | Tests | Expected outcome if H1 is root cause |
|---|---|---|
| A_baseline | reporter's config | small p99 ≈ 22s, flag-true episode ≈ 22s, confirms repro |
| B_no_mixed_chunk | `--disable-mixed-chunk` | similar to A (H2 is secondary) |
| C_mrr8 | `--max-running-requests 8` | p99 DROPS (forces earlier shrink → flag clears sooner) |
| D_chunk32k | `--chunked-prefill-size 32768` | mild improvement; still sticky |
| E_priority | `--enable-priority-scheduling` | small p99 stays high (no priority assigned by client) |
| F1_smoke_fix | SGLANG_TTFT_SMOKE_FIX=1 | **small p99 < 2s** → H1 confirmed |

## Cleanup

```bash
python3 ttft_ablation/patches/apply_patches.py --revert
# or
git checkout -- python/sglang/srt/managers/scheduler.py
```
