# TTFT Ablation — Issue #22831

Research directory for the small-request TTFT regression reported in
[sgl-project/sglang#22831](https://github.com/sgl-project/sglang/issues/22831).

**Status**: root cause identified, fix proposed, empirical validation done.
For the TL;DR and the patch, see [CONCLUSION.md](CONCLUSION.md).

## Layout

```
ttft_ablation/
├── README.md              ← you are here (navigation)
├── CONCLUSION.md          ← proposed PR: root cause, fix, diff, validation
├── EXPERIMENT_LOG.md      ← full journal: all hypotheses, all runs, analysis
│
├── repro_bench.py         ← self-contained TTFT load generator (no fuzzer dep)
├── run_ablation.sh        ← harness: runs A/E2/D/H/G/E3/G3 configs end-to-end
├── parse_logs.py          ← extracts TTFT + flag-episode stats → markdown table
├── patches/
│   └── apply_patches.py   ← idempotent anchor-based patcher for scheduler.py
│
└── results/
    ├── primary/           ← final runs (A, E2, D, H, G, E3, G3) on a common server
    ├── replicate/         ← outer-loop replicate (A, E3) for variance estimate
    └── refuted_configs/   ← supporting data for refuted hypotheses (F1, C)
```

Pod-level setup lives outside this dir at `/workspace/pod_bootstrap.sh` —
it provisions the venv and HF cache on /workspace so the 20 GB ephemeral
overlay doesn't fill up. Not part of the project itself.

## Quick start (reproducing the result)

```bash
# 1. Apply the instrumentation + both fix variants (all env-gated; no-op otherwise)
python3 ttft_ablation/patches/apply_patches.py

# 2. Run the full sweep (~30 min on a single 4090)
cd ttft_ablation
./run_ablation.sh all

# 3. Get the comparison table
python3 parse_logs.py results/<timestamp>/
```

Environment flags the patches respect:

| flag | effect |
|---|---|
| `SGLANG_TTFT_DEBUG=1` | emit `[TTFT_DEBUG]` per-iter trace events |
| `SGLANG_TTFT_SMOKE_FIX=1` | F1: clear `batch_is_full` at each admission entry (refuted) |
| `SGLANG_TTFT_HOL_FIX=1` | E2: unconditionally continue on NO_TOKEN (works, but regresses large) |
| `SGLANG_TTFT_HOL_SMART=1` | E3: continue on NO_TOKEN only when `rem_total_tokens > 0` (**proposed fix**) |

Revert:
```bash
python3 ttft_ablation/patches/apply_patches.py --revert
```

## Reading order

1. [`CONCLUSION.md`](CONCLUSION.md) — root cause + 5-line patch.
2. [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) — every config tested, all
   hypotheses, full result matrix, variance replicate.
3. `repro_bench.py` + `run_ablation.sh` — to re-run.
