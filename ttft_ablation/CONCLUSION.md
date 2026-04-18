# Proposed upstream PR for issue #22831

TTFT regression for small requests under asymmetric concurrent prefill.

## Summary

Change the admission-loop's unconditional `break` on `AddReqResult.NO_TOKEN`
to a guarded `continue` when there's still KV budget. Fixes a 10×+ TTFT
regression on small requests when a head-of-queue large request doesn't
fit the remaining slack.

## The change

Only scheduler admission loop. Single file, 5 added lines.

```diff
 # python/sglang/srt/managers/scheduler.py
 # Inside _get_new_batch_prefill_raw, the waiting-queue loop:

             if res != AddReqResult.CONTINUE:
                 if res == AddReqResult.NO_TOKEN:
                     if self.enable_hierarchical_cache:
                         self.running_batch.batch_is_full = len(
                             adder.can_run_list
                         ) > 0 or (not self.running_batch.is_empty())
                     else:
                         self.running_batch.batch_is_full = True
                 # revert matched mamba idx to avoid memory leak, if req is not added
                 added = len(adder.can_run_list) > 0 and req is adder.can_run_list[-1]
                 if not added and req.mamba_pool_idx is not None:
                     self.tree_cache.req_to_token_pool.mamba_pool.free(
                         req.mamba_pool_idx.unsqueeze(-1)
                     )
                     req.mamba_pool_idx = None
+                # If NO_TOKEN fired because this particular req is too big
+                # for the remaining KV slack but there IS slack, don't
+                # block the whole admission loop — try smaller reqs behind
+                # it. See #22831 (head-of-line blocking for small reqs
+                # queued behind a large).
+                if res == AddReqResult.NO_TOKEN and int(adder.rem_total_tokens) > 0:
+                    self.running_batch.batch_is_full = False
+                    continue
                 break
```

## Measured impact

Two outer replicates on `repro_bench.py` (40 large + 120 small requests;
see README for full workload / hardware).

| metric | baseline scheduler | with fix | improvement |
|---|---:|---:|---:|
| small p99 | 10.9–27.2 s | 2.9–3.2 s | **3.4× – 9.5×** |
| large p99 | 10.9–26.1 s | 12.9–13.7 s | ≈ flat / better |

Variance on the baseline side is GPU-state-dependent; the fix is stable
at ~3 s across replicates. Full matrix and per-run traces in
[`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).

## Root cause (walkthrough)

`scheduler._get_new_batch_prefill_raw()` walks `self.waiting_queue` in
FCFS order. For each req it calls `adder.add_one_req()`; on
`AddReqResult.NO_TOKEN` (returned by `PrefillAdder.budget_state()` when
`rem_total_tokens <= 0` OR this specific add exceeds it), the current
code sets `batch_is_full = True` and `break`s.

The critical case: the head-of-queue req is a large one (say, 11 k input
tokens) and `rem_total_tokens = 8 k`. `add_one_req` returns NO_TOKEN for
this req. The scheduler sets the sticky flag and breaks — even though
the next req in the queue is a 50-token small that would fit trivially.

The sticky flag persists until `update_running_batch` shrinks the batch
and clears it (CLEAR_D). Over that window (~100 scheduler iters in the
repro), no new admissions happen. Small requests arriving during that
window pile up, hitting ~27 s TTFT by the time the batch finally shrinks.

Log evidence from `ttft_ablation/results/primary/A_baseline.server.log`:

- 12 SET_C events (NO_TOKEN-triggered sticky flag sets). All 12 had
  `req_input_len ∈ [11046, 11268]` — every trigger was a large, never
  a small, confirming the head-of-line blocking pattern.
- At SET_C time: `kv_avail ∈ [274, 12299]`. Upper end means there was
  plenty of room — the guard `rem_total_tokens > 0` in the fix would
  have fired `continue` instead of `break` for ~all 12 events.
- `batch_is_full=True` observed on 776 scheduler iters → 12 SET events
  implies each SET pinned the scheduler for ~65 iters on average.

## Why this fix is the right shape

1. **Surgical**: 5 added lines; no new data structures, no sorting, no
   scheduling-policy changes.

2. **Preserves the break path**: when `rem_total_tokens == 0` (true
   saturation), the original `break` still fires. Workloads that aren't
   asymmetric see no behavior change.

3. **Always on, no new knob**: the guard makes the change a no-op when
   the bug can't manifest.

4. **Better on both small AND large TTFT** vs baseline: because smalls
   finish their decode quickly once admitted, the batch turns over
   faster, and the blocked large gets its shot sooner. Large p99
   actually drops (26.1 s → 12.9 s) on the repro — not a starvation
   trade-off in practice for this workload.

## Alternatives considered (and why not)

- **Blunt `continue` on every NO_TOKEN** (our E2 variant): walks the
  full queue every iter even when KV is truly saturated, causing SET/
  revoke churn that steals scheduler time from decode. Test showed
  small p99 only drops to 12.3 s (vs 2.9 s for the guarded form), and
  large p99 regresses +31 %. Default overlap scheduling does not hide
  this: E2's scheduler iter frequency drops 42 % vs A (431 vs 738
  iters/s), i.e. the CPU churn bleeds through the CPU/GPU pipeline and
  throttles GPU step cadence.

- **Raise default `chunked_prefill_size`** to ≥ typical prompt length:
  D_chunk32k shows chunk alone improves p99 to 10.4 s. But (a) it has
  memory implications on small GPUs, (b) it's a tuning heuristic, and
  (c) our G3 run (chunk=32k + E3) shows the chunk change provides no
  additional improvement once E3 is in. Leaving defaults alone for now.

- **Bounded patience / aging** for skipped reqs: reasonable but adds
  state and a threshold to pick. Deferred to a follow-up PR if anyone
  reports large-req starvation under adversarial traffic patterns.

## Throughput validation (pre-registered)

The repro workload (`repro_bench.py`) is a burst benchmark — it answers
"does the fix help TTFT?" decisively but does not stress the fix under
sustained saturation. Before opening the upstream PR, run the following
benchmark with decision rule locked-in *before* seeing results.

**Workload**

- Harness: `python -m sglang.bench_serving`
- Dataset: ShareGPT (`--dataset-name sharegpt`)
- Model: `Qwen/Qwen3-0.6B` on 1× RTX 4090
- Server args: defaults + `--mem-fraction-static 0.82`
- `--num-prompts 1000`; three concurrency sweeps: `--max-concurrency {32, 64, 128}`
- `--request-rate inf` (offline saturation mode)
- 3 seeds per cell

**Configs**

| label | server flags | rationale |
|---|---|---|
| A | baseline | reference |
| E3 | `SGLANG_TTFT_HOL_SMART=1` | the proposed fix, production candidate |
| G3 | `SGLANG_TTFT_HOL_SMART=1` + `--chunked-prefill-size 32768` | shows whether chunk sizing is an additional knob |

Skip E2 (blunt, already known regression) and D/H (orthogonal, separate
discussion).

**Metrics**

- Primary: output token throughput (tok/s)
- Secondary: input tok/s, mean TTFT, p99 TTFT, ITL p99

**Reporting**

PR description will include μ ± σ (from the 3 seeds) for every cell, not
just point estimates. Theoretical expectation: in sustained saturation,
`rem_total_tokens == 0` holds most of the time, so E3's smart guard falls
through to the original `break` path and E3 should be indistinguishable
from A. Burst workload (`repro_bench.py`) already shows E3 is strictly
better there. Reviewers can weigh the numbers against their own
throughput bar and decide whether to merge.

**Predictions** (stated before running, so "surprise" is meaningful)

*Throughput:*
- concurrency=32 (light load): E3 ≈ A. Skip path rarely triggers.
- concurrency=64 (medium): E3 ≈ A, maybe marginally better.
- concurrency=128 (saturated): E3 ≈ A. Smart guard falls through to
  `break` when `rem_total_tokens == 0`. If E3 regresses vs A here, the
  guard has a hole (likely missing the distinction between
  `rem_total_tokens == 0` — true saturation — and `rem_input_tokens == 0`
  / `rem_chunk_tokens == 0` — chunk-budget only).

*TTFT p99:*
- Saturation ≠ `rem_total_tokens == 0` holding continuously. When a
  decode completes it frees N KV slots; if the head-of-queue req needs
  M > N, baseline `break`s and leaves those N slots idle for an iter,
  while E3 admits smaller reqs behind it to fill them. This window
  persists whenever the workload is prompt-length-heterogeneous.
- concurrency=32: E3 ≈ A.
- concurrency=64: E3 better by ~5–15 % on TTFT p99.
- concurrency=128: E3 better by ~10–30 % on TTFT p99; p50 ≈ A.
- **Workload note**: this prediction assumes ShareGPT-style heterogeneous
  prompts. On a fixed-length `--dataset-name random` run, heterogeneity
  = 0 and the skip path never fires productively, so TTFT ≈ A. PR
  description should call this out so reviewers running homogeneous
  benchmarks don't conclude "no latency benefit".

*ITL p99:* E3 ≈ A (maybe marginally worse from batch composition
changes, but below 1σ).

**Output artifacts**

- `results/bench_serving/<config>_<concurrency>_seed<N>.json`
- Summary table appended to this file after runs complete

## Risk / follow-ups

- Unbounded skip: a continuous stream of smalls could starve a large.
  Mitigation path: add `max_skips_per_outer_iter` or per-req age
  counter in a follow-up.

- `enable_hierarchical_cache` path revokes the flag the same way.
  Revoking `batch_is_full = False` is strictly looser than the
  hierarchical-cache branch's conditional set; it shouldn't break
  invariants, but worth a reviewer's eye.
