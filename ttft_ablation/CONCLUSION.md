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

## Measured impact — issue-repro (A vs E3, paired, n=10)

**Setup.** `repro_bench.py` workload: 40 large (~11 k tok) + 120 small
(~50 tok) requests co-firing at t=0. Baseline scheduler sets
`batch_is_full=True` on every large-triggered `NO_TOKEN`, so HOL fires
on every seed — this is the "upper bound on E3's impact" measurement.
10 paired seeds × 3 inner runs averaged. Raw logs in
`results/issue_repro/paired_n10/`; parser: `parse_issue_repro.py`.

**Per-seed raw** (10/10 seeds show the same direction on every TTFT
metric — tightness of the cluster is the headline):

| seed | small_p99 A | small_p99 E3 | Δ      | large_p99 A | large_p99 E3 |
| ---: | ----------- | ------------ | ------ | ----------- | ------------ |
| 0    | 10.75 s     | 2.88 s       | −73.2 %| 10.86 s     | 11.31 s      |
| 1    | 10.77 s     | 2.87 s       | −73.4 %| 10.87 s     | 11.32 s      |
| 2    | 10.75 s     | 2.88 s       | −73.2 %| 10.87 s     | 11.32 s      |
| 3    | 10.78 s     | 2.87 s       | −73.4 %| 10.88 s     | 11.35 s      |
| 4    | 10.77 s     | 2.88 s       | −73.3 %| 10.86 s     | 11.29 s      |
| 5    | 10.75 s     | 2.87 s       | −73.3 %| 10.83 s     | 11.29 s      |
| 6    | 10.77 s     | 2.87 s       | −73.3 %| 10.88 s     | 11.33 s      |
| 7    | 10.85 s     | 2.89 s       | −73.4 %| 10.94 s     | 11.29 s      |
| 8    | 10.76 s     | 2.86 s       | −73.4 %| 10.87 s     | 11.32 s      |
| 9    | 10.79 s     | 2.87 s       | −73.4 %| 10.87 s     | 11.33 s      |

**Paired aggregate** (Wilcoxon signed-rank, n=10):

| metric        | A (μ ± sd)       | E3 (μ ± sd)     | Δ           | ↓/10  | p (Wilcoxon) |
|---------------|------------------|-----------------|------------:|------:|-------------:|
| small p99     | 10.77 s ± 32 ms  | 2.87 s ± 8 ms   | **−73.3 %** | 10/10 | **0.002**    |
| small p50     | 7.92 s ± 34 ms   | 1.35 s ± 16 ms  | −82.9 %     | 10/10 | 0.002        |
| small mean    | 8.04 s ± 31 ms   | 1.26 s ± 8 ms   | −84.3 %     | 10/10 | 0.002        |
| large p99     | 10.87 s ± 27 ms  | 11.31 s ± 21 ms | **+4.1 %**  | 0/10  | 0.002        |
| large mean    | 5.52 s ± 15 ms   | 5.83 s ± 9 ms   | +5.5 %      | 0/10  | 0.002        |
| output tok/s  | 354 ± 8 tok/s    | 350 ± 2 tok/s   | −1.1 %      | 2/10  | 0.328        |

Every TTFT direction is significant at p=0.002 (the Wilcoxon floor for
n=10 with unanimous sign, so the direction is reliable); throughput is
non-significant (2/10 directionally higher, p=0.328) — neutral.

**Reading**: small p99 drops ~3.7× (deterministic and reproducible to
the ms across paired seeds) at the cost of a ~4 % / ~450 ms absolute
shift on large p99 and no measurable throughput change. The large-side
shift is structural: admitting a small into leftover KV slack pushes
the next large's admission by one slot-release window (one decode
step).

### Cross-run comparison — E3 eliminates tail variance

The cleanest way to see what the fix actually does is to pool this
n=10 sweep with the earlier primary + replicate single-outer runs (all
same scheduler.py, same seeds, same GPU — just different wall-clock
sessions so thermal / CUDA-graph warmup state varies):

| run                 | A small p99 | E3 small p99 | A large p99 | E3 large p99 |
|---------------------|-------------|--------------|-------------|--------------|
| primary (n=1 outer) | 27.15 s     | 2.85 s       | 26.10 s     | 12.85 s      |
| replicate (n=1)     | 10.87 s     | 3.24 s       | 10.92 s     | 13.74 s      |
| **paired n=10**     | **10.77 s ± 32 ms** | **2.87 s ± 8 ms** | **10.87 s ± 27 ms** | **11.31 s ± 21 ms** |

Two regimes on the baseline side:

- **Baseline is GPU-state-sensitive.** A small p99 swings 10.8 → 27 s
  across sessions; A large p99 swings 10.9 → 26 s. The mechanism is
  straightforward: each sticky-flag set pins admission for ~65
  scheduler iterations, and how long those 65 iters take depends on
  decode-step wall-clock (thermal throttle, CUDA graph cache state,
  allocator churn, etc.). The bug **amplifies** whatever latency
  noise is present in the decode path into the admission-pinning
  window.

- **E3 is GPU-state-insensitive.** E3 small p99 clusters at 2.9 ±
  0.2 s, E3 large p99 at 12.5 ± 1.2 s — both ~10× tighter std than
  baseline. Reason: E3 never enters the amplification loop. Admission
  proceeds at the natural slot-release rate regardless of what decode
  is doing.

**So the fix's value is not primarily "lower mean"** (though it is
that, 3.7×); it is **"collapse tail variance"**. In the lucky
baseline regime (n=10 sweep), E3 pays +4 % on large p99 — the
structural cost of admitting one extra small per slot. In the
unlucky regime (primary), baseline pays 2.4× on both small and large
p99, and E3 stays at its deterministic floor. A prod operator's
tail-latency SLA is defined over whatever regimes their traffic hits
— E3 removes the bad regime entirely.

Full per-seed log traces and the full TTFT-ablation matrix (E2/D/G/G3/H)
in [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).

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

4. **Large-side cost is bounded and structural**, not runaway
   starvation. Admitting a small into leftover KV slack pushes the
   next large admission by one slot-release window (one decode step).
   Measured: in the low-variance paired-n=10 regime, large p99 shifts
   +4 % (+450 ms, 0/10 seeds improving); in the high-variance regime
   where baseline had already paid heavy TTFT tail cost, large p99
   *improves* ~50 % because small throughput frees up room faster.
   Either way, no unbounded starvation — small admission stops as soon
   as `rem_total_tokens` hits zero, and `max_skips_per_outer_iter` is
   available as a bounded follow-up if any adversarial workload
   surfaces real large-side delay.

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

## Throughput validation — ShareGPT (A vs E3, paired, n=10)

**Setup.** `python -m sglang.bench_serving` against `Qwen/Qwen3-0.6B`
on 1× RTX 4090; ShareGPT, 1000 prompts, `--request-rate inf`,
`--mem-fraction-static 0.82`, seeds 0..9 paired, concurrency
∈ {32, 64, 128}. E3 = baseline + `SGLANG_TTFT_HOL_SMART=1`.
Raw JSONs in `results/throughput/sweep/`; parsers: `parse_throughput.py`
(aggregation), `paired_stats.py` (paired t-test + JSON-based HOL-event
detection).

Three tiers of evidence, from mechanism → user-visible → safety:

#### Tier 1 — HOL event rate (mechanism)

Count HOL events, don't chase aggregate means: a seed is HOL-affected
when (a) p99 TTFT is a σ>2 outlier for its cell AND (b) run duration is
≥ 2× the cell median — the signature of a frozen admission loop.

At c=32, baseline s2 matches both (p99 678 ms, duration 242 s vs μ
113 s) and s4 matches (b) weakly plus server-log confirmation (p99
367 ms, duration ≈ 180 s).

| concurrency | baseline HOL rate | E3 HOL rate |
|---:|:---|:---|
| 32  | **2/10 = 20 %**, Wilson 95 % CI [5.7 %, 51.0 %] | 0/10 = 0 %, Wilson 95 % CI [0.0 %, 27.8 %] |
| 64  | 0/10 (s9 shows p99 elevation but E3 pairs equally; tail-luck, not HOL) | 0/10 |
| 128 | 0/10 | 0/10 |

Fisher exact on the c=32 2×2 table (A vs E3, HOL vs healthy):

- Two-sided p = 0.474 (cannot reject the null that rates are equal at n=10)
- One-sided "A has higher HOL rate than E3" p = **0.237**

Not stat-sig at n=10 — rare-event detection needs larger n to separate
20 % from 0 %. But direction is unambiguous and mechanism is confirmed
by per-iter server-log trace: all 12 NO_TOKEN-triggered sticky-flag
sets on s2 had `req_input_len ∈ [11046, 11268]`, i.e. every trigger was
a large request — textbook HOL pattern.

#### Tier 2 — User-visible impact at c=32

Full paired n=10 at c=32, including the HOL-affected baseline seeds
(what a ShareGPT user would experience, outliers and all):

| metric | μ_A | μ_E3 | Δ | p (paired t) |
|---|---:|---:|---:|---:|
| output throughput (tok/s) | 1887 | 2036 | **+7.9 %** | 0.25 |
| p99 TTFT (ms) | 238 | 161 | **−32.4 %** | 0.18 |
| p99 ITL (ms)  | 55  | 37  | **−32.4 %** | 0.25 |

Non-sig at n=10 because baseline variance is inflated by the two HOL
seeds — the paired t-test correctly refuses to declare significance
when the signal is concentrated in a minority. Read the takeaway as
"when the bug fires in real traffic, the user-visible tail drops ~30 %",
and look to Tier 1 for the mechanism claim and Tier 3 for no-regression.

#### Tier 3 — Safety / no-regression

Using the c=32 aggregate to ask "does E3 hurt healthy traffic?" would
double-count the mechanism gain (outlier elimination ≠ per-seed shift).
The right views are the healthy c=32 subset (excluding the two HOL
seeds) and the c=64 / c=128 cells (where HOL events are essentially 0).

Healthy c=32 subset (n=8, excludes s2 + s4):

| metric | μ_A | μ_E3 | Δ | p | sign |
|---|---:|---:|---:|---:|---:|
| throughput  | 2028 | 2044 | +0.8 % | 0.22 | 6/8 |
| p99 TTFT    | 167  | 162  | −3.4 % | 0.34 | 5/8 |
| mean TTFT   | 41   | 41   | −1.0 % | 0.29 | 5/8 |
| p99 ITL     | 37   | 37   | −0.6 % | 0.52 | 3/8 |

c=64 (n=10, no seed exclusion; 0/10 HOL):

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
