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
  large p99 regresses +31 %.

- **Raise default `chunked_prefill_size`** to ≥ typical prompt length:
  D_chunk32k shows chunk alone improves p99 to 10.4 s. But (a) it has
  memory implications on small GPUs, (b) it's a tuning heuristic, and
  (c) our G3 run (chunk=32k + E3) shows the chunk change provides no
  additional improvement once E3 is in. Leaving defaults alone for now.

- **Bounded patience / aging** for skipped reqs: reasonable but adds
  state and a threshold to pick. Deferred to a follow-up PR if anyone
  reports large-req starvation under adversarial traffic patterns.

## Risk / follow-ups

- Unbounded skip: a continuous stream of smalls could starve a large.
  Mitigation path: add `max_skips_per_outer_iter` or per-req age
  counter in a follow-up.

- `enable_hierarchical_cache` path revokes the flag the same way.
  Revoking `batch_is_full = False` is strictly looser than the
  hierarchical-cache branch's conditional set; it shouldn't break
  invariants, but worth a reviewer's eye.
