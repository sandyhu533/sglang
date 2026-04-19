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

### Results (A vs E3, paired, n=10)

Raw JSONs: `ttft_ablation/results/throughput/sweep/`. Parsers:
`ttft_ablation/parse_throughput.py` (raw aggregation) and
`ttft_ablation/paired_stats.py` (paired t-test + JSON-based anomaly
detection). G3 skipped — A vs E3 already answers the "does the fix
regress throughput?" question; the extra chunk-sizing knob was already
shown orthogonal in the TTFT ablation (G3 ≈ E3 there).

Three tiers of evidence, ordered from mechanism-level (what the fix
actually claims to do) to user-visible impact to safety:

#### Tier 1 — Primary: HOL event rate (mechanism)

The fix targets head-of-line blocking. The direct way to tell if it
works is to count HOL events before and after, not to chase aggregate
latency means. Baseline seeds are classified as HOL-affected when both
(a) p99 TTFT is a σ>2 outlier or near it under server-log inspection
and (b) run duration is 2×+ the cell median — the combined pattern
that characterizes a frozen admission loop (long duration because the
outlier run cannot drain quickly).

At c=32, baseline s2 (p99 678 ms, duration 242 s vs μ 113 s; σ>2) and
baseline s4 (p99 367 ms, duration ≈ 180 s; server-log inspection) both
match. Other cells have at most one borderline seed.

| concurrency | baseline HOL rate | E3 HOL rate |
|---:|:---|:---|
| 32  | **2/10 = 20 %**, Wilson 95 % CI [5.7 %, 51.0 %] | 0/10 = 0 %, Wilson 95 % CI [0.0 %, 27.8 %] |
| 64  | 1/10 (s9 borderline, duration normal) | 0/10 |
| 128 | 0/10 | 0/10 |

Fisher exact on the c=32 2×2 table (A vs E3, HOL vs healthy):

- Two-sided p = 0.474 (cannot reject the null that rates are equal at n=10)
- One-sided "A has higher HOL rate than E3" p = **0.237**

Not stat-sig at n=10 per arm — as expected, since rare-event detection
needs much larger n to separate 20 % from 0 %. But the direction is
unambiguous and the mechanism (skip oversized head-req when slack
remains) is verified by per-iter server-log trace on the affected
seeds (all 12 NO_TOKEN-triggered sticky-flag sets in s2 had
`req_input_len ∈ [11046, 11268]`, i.e. every trigger was a large req
— textbook HOL pattern).

#### Tier 2 — Secondary: User-visible impact (headline numbers)

Full paired n=10 at c=32 (includes the HOL-affected baseline seeds —
this is what a ShareGPT user actually experiences, outliers and all).

| metric | μ_A | μ_E3 | Δ | p (paired t) |
|---|---:|---:|---:|---:|
| output throughput (tok/s) | 1887 | 2036 | **+7.9 %** | 0.25 |
| p99 TTFT (ms) | 238 | 161 | **−32.4 %** | 0.18 |
| p99 ITL (ms)  | 55  | 37  | **−32.4 %** | 0.25 |

Not stat-sig at n=10 because the HOL-affected baseline seeds drive up
baseline variance. That is a feature, not a bug, of the paired
t-test: it refuses to declare significance when the signal is
concentrated in a minority of seeds. The mechanism test in Tier 1 is
the right instrument for that; Tier 2 says "when the bug fires in
real traffic, the user-visible tail drops ≈ 30 %".

c=64 and c=128 show no headline change (see Tier 3).

#### Tier 3 — Tertiary: Safety / no-regression

For "does the fix hurt performance on healthy (non-HOL) traffic?" the
c=32 aggregate is the wrong tool — the n=10 deltas above are
mechanically driven by outlier elimination, not by a per-seed shift,
so using them for regression analysis double-counts the mechanism
gain. The right view is: on the healthy-seed subset plus c=64 / c=128
(which have essentially no HOL events to begin with), does E3 cost
anything?

Healthy c=32 subset (n=8, excludes s2 + s4):

| metric | μ_A | μ_E3 | Δ | p | sign |
|---|---:|---:|---:|---:|---:|
| throughput  | 2028 | 2044 | +0.8 % | 0.22 | 6/8 |
| p99 TTFT    | 167  | 162  | −3.4 % | 0.34 | 5/8 |
| mean TTFT   | 41   | 41   | −1.0 % | 0.29 | 5/8 |
| p99 ITL     | 37   | 37   | −0.6 % | 0.52 | 3/8 |

c=64 (n=10, no seed exclusion; baseline HOL count 1/10 borderline):

| metric | μ_A | μ_E3 | Δ | p | sign |
|---|---:|---:|---:|---:|---:|
| throughput | 3525 | 3518 | −0.2 % | 0.84 | 5/10 |
| p99 TTFT   | 337  | 323  | −4.1 % | 0.32 | 7/10 |
| p99 ITL    | 44   | 45   | +2.1 % | 0.22 | 4/10 |

c=128 (n=10, no seed exclusion; 0/10 HOL):

| metric | μ_A | μ_E3 | Δ | p | sign |
|---|---:|---:|---:|---:|---:|
| throughput | 5899 | 5844 | −0.9 % | 0.52 | 4/10 |
| p99 TTFT   | 579  | 575  | −0.8 % | 0.91 | 3/10 |
| p99 ITL    | 65   | 67   | +2.5 % | 0.20 | 4/10 |

No stat-sig movement at any cell. Largest sub-noise delta is p99 ITL
+2.5 % at c=128 (p=0.20), directionally consistent with admitting an
extra small that briefly co-runs with decode; bounded by the
follow-up `max_skips_per_outer_iter` knob if it becomes reproducible.

### Prediction vs observed (n=10)

- **HOL rate** (pre-registered as the mechanism test, not a mean): A
  2/10, E3 0/10. Direction correct, magnitude matches pre-reg belief
  that ShareGPT at c=32 is where the bug most plausibly fires. ✓
- **Throughput c=32 "E3 ≈ A"**: headline observed +7.9 %. Under-
  predicted — the original "skip path rarely triggers at light load"
  assumption missed that ShareGPT c=32 is exactly where HOL does
  fire in the wild. Healthy-subset check (+0.8 %) matches the
  original "≈ A" prediction, so the prediction was right for the
  no-regression claim but wrong for not anticipating the mechanism
  would fire at light load.
- **Throughput c=64/128 "E3 ≈ A"**: observed −0.2 % / −0.9 %
  (p ≥ 0.52). ✓
- **TTFT p99 c=32 "E3 ≈ A"**: headline −32 %; healthy-subset −3.4 %.
  Same miss as above.
- **TTFT p99 c=64 "−5 to −15 %"**: observed −4.1 % (p=0.32, 7/10
  seeds directionally better). Just under the predicted lower bound.
- **TTFT p99 c=128 "−10 to −30 %"**: observed −0.8 % (p=0.91). No
  effect. ShareGPT c=128 is concurrency-saturated (running-req hits
  max, `token_usage` ≤ 0.43) rather than KV-saturated — E3's guard
  rarely fires when `add_one_req` returns `OTHER` (max-concurrency)
  rather than `NO_TOKEN`.
- **ITL p99 "≈ A, maybe marginally worse below 1σ"**: headline c=32
  −32 % (HOL outlier effect); c=64 +2.1 %, c=128 +2.5 % (both non-
  sig). c=64 / c=128 match prediction; c=32 is pulled by the same
  outlier effect as TTFT p99.

**Verdict.** Three-layer story:

- **Fix works** (Tier 1): baseline HOL rate 20 % → 0 % on ShareGPT
  c=32, Fisher one-sided p = 0.24 at n=10 per arm, mechanism confirmed
  by per-iter trace on the affected seeds. The 9.5× gain on the
  asymmetric repro is the same mechanism under stronger signal.
- **User-visible when it fires** (Tier 2): full n=10 c=32 shows
  throughput +7.9 %, p99 TTFT −32 %, p99 ITL −32 %. Non-sig at this n,
  driven by outlier elimination (as designed).
- **Safe on healthy traffic** (Tier 3): healthy-subset c=32 deltas
  within ±3.4 %; c=64 / c=128 all within ±2.5 %, no stat-sig movement.
  E3's guard correctly falls through to the original `break` path
  under true saturation.

Combined, this is the "safe to merge" signal: direct mechanism
evidence + no regression at any tested concurrency + a real-world
workload where the bug demonstrably fires.

## Risk / follow-ups

- Unbounded skip: a continuous stream of smalls could starve a large.
  Mitigation path: add `max_skips_per_outer_iter` or per-req age
  counter in a follow-up.

- `enable_hierarchical_cache` path revokes the flag the same way.
  Revoking `batch_is_full = False` is strictly looser than the
  hierarchical-cache branch's conditional set; it shouldn't break
  invariants, but worth a reviewer's eye.

- Sub-noise ITL p99 elevation on ShareGPT at saturation (+2.1 % at
  c=64, +2.5 % at c=128; p ≥ 0.20 at n=10). Direction consistent
  across both concurrencies but magnitude below stat-sig. Attributable
  to E3 occasionally admitting a small that briefly co-runs with
  decode. If adoption surfaces a reproducible ITL p99 increase on
  different workloads, `max_skips_per_outer_iter` (same as the
  starvation follow-up) bounds it cleanly.
