# Phase 1B Results — Issue #22831

Reproduced on **1× RTX 4090** (24 GB), Qwen3-0.6B, `mem-fraction-static=0.82`.
Workload from `repro_bench.py` (self-contained, no fuzzer deps):
40 large (~3500-word prompts) + 120 small (~40-word prompts) per run, 3 runs.

## Headline numbers

| config | small p50 | small p99 | large p99 | baseline | notes |
|---|---:|---:|---:|---:|---|
| A_baseline | 25 011 ms | **26 935 ms** (106× baseline) | 26 365 ms | 254 ms | bug reproduces |
| F1_smoke_fix | 25 935 ms | **28 720 ms** (113× baseline) | 27 611 ms | 253 ms | no improvement |

Reporter's original numbers on 3× 4090: p99 TTFT = 22.9 s (454× baseline).
Our 1-GPU reproduction is in the same order of magnitude and same shape
(small + large both elevated, uniform p50 / p99).

## Server-side event counts (from `TTFT_DEBUG` instrumentation)

| metric | A_baseline | F1_smoke_fix |
|---|---:|---:|
| `ev=ENTRY` | 176 151 | 15 781 |
| `ev=EARLY_RETURN` | 175 458 | 14 710 |
| `ev=CLEAR_D_BATCH_SHRUNK` | 10 | 15 |
| `batch_is_full=True` occurrences | **772** | **0** |
| peak `running_bs` | 100 | 114 |
| peak `waiting` | 89 | 99 |
| retractions | 0 | 0 |

## Interpretation

**H1 (sticky `batch_is_full` flag) is NOT the root cause.** The smoke fix
in F1 resets the flag at the top of every scheduler iteration — the event
counts confirm it works (flag goes from 772 → 0 True-observations). Yet
`small p99` is unchanged (~27 s). If the bug were "small requests refused
admission because the flag was stuck True", F1 would have collapsed the TTFT
delta; it did not.

The peak `running_bs` in F1 (114) is actually higher than A (100), meaning
small requests **do** reach the running batch — just late. The bottleneck is
admission-ordering, not admission-refusal.

Two hypotheses remain consistent with the data:

1. **Decode-batch-size scaling** (the reporter's framing). Once ~100 requests
   are simultaneously in decode, each forward step processes all of them;
   step time scales with batch size, so a small request that *is* admitted
   still takes many slow decode steps to emit its first token.
2. **FCFS head-of-line blocking**. Large requests arrive at t=0, small
   stream at t=300 ms. In FCFS, smalls wait behind the 40 large prefills.

Ablations yet to run: **C_mrr8** (`--max-running-requests 8`) tests (1) by
capping decode batch size; **D_chunk32k** (`--chunked-prefill-size 32768`)
tests whether faster large-prefill throughput reduces small queue time;
**E_priority** tests whether priority scheduling helps (likely no, since
the client doesn't set priority headers).

## Patch coverage caveat

`patches/01_debug_log.patch` instruments only two `batch_is_full = True`
sites (`SET_A` at L2488, `SET_B` at L2560). Two other set-sites in
`scheduler.py` (L2574, L2611) are uninstrumented, which is why A's log shows
`batch_is_full=True` 772 times but `ev=SET_*` counts are 0. The pairing-based
"flag-true episodes" column in `parse_logs.py` is therefore not trustworthy
for A. This does **not** affect the A-vs-F1 p99 comparison (which uses the
client-side TTFT numbers directly).
