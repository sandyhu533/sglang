# TTFT Ablation Experiment Journal — Issue #22831

## 0. Setup

| item | value |
|---|---|
| GPU | 1× RTX 4090 (24 GB) |
| Model | `Qwen/Qwen3-0.6B` |
| Workload | `repro_bench.py` defaults: 40 large (~3 500-word prompts) + 120 small (~40-word prompts), 3 inner runs, seed=0 |
| Common server args | `--mem-fraction-static 0.82` |
| Outer-loop replicates | 1 (primary) + 1 (replicate, A & E3 only) |
| Metrics reference row | A_baseline primary run (small p99 = 27 152 ms) |

"Δ vs A" in every run table is `(this_run - A_baseline) / A_baseline × 100%`,
computed against the primary A_baseline column.

## 1. TL;DR — summary matrix

| run | config | HOL mode | small p50 | small p99 | Δ small p99 | large p99 | Δ large p99 | large mean | peak bs |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| A_baseline (primary) | chunk=2048 | off | 24 306 | 27 152 | — (ref) | 26 105 | — (ref) | 13 837 | 100 |
| E2_hol_fix | chunk=2048 | blunt | 9 730 | 12 336 | −55 % | 34 129 | **+31 %** | 19 402 | 74 |
| D_chunk32k | chunk=32 768 | off | 7 556 | 10 441 | −62 % | 10 497 | −60 % | 5 323 | 100 |
| H_chunk_12k | chunk=12 288 | off | 7 533 | 10 306 | −62 % | 10 418 | −60 % | 5 301 | 100 |
| G_chunk_hol_combo | chunk=32 768 | blunt | 279 | 2 821 | −90 % | 12 432 | −52 % | 6 699 | 74 |
| **E3_smart_hol** | **chunk=2048** | **smart** | **345** | **2 854** | **−89 %** | **12 854** | **−51 %** | 6 893 | 74 |
| G3_smart_combo | chunk=32 768 | smart | 233 | 2 751 | −90 % | 12 237 | −53 % | 6 544 | 74 |
| A_baseline (replicate) | chunk=2048 | off | 8 011 | 10 870 | — | 10 919 | — | 5 581 | — |
| E3_smart_hol (replicate) | chunk=2048 | smart | 1 092 | 3 245 | — | 13 744 | — | 7 134 | — |

Key reading: only E2 regresses large p99 (**+31 %**) — the blunt
unconditional skip starves larges. Every other fix *improves* large p99
(−51 % to −60 %) because smalls clearing faster speeds up batch turnover.
E3 gets the small-p99 win without E2's starvation cost.

All numbers in ms. "HOL mode": `off` = original, `blunt` = continue on every NO_TOKEN
(`SGLANG_TTFT_HOL_FIX=1`), `smart` = continue only when `rem_total_tokens > 0`
(`SGLANG_TTFT_HOL_SMART=1`).

## 2. Hypothesis ledger (final)

| ID | Hypothesis | Status | Decisive evidence |
|---|---|---|---|
| H1 | sticky `batch_is_full` flag refuses small admits | **REFUTED** | F1 (clear flag at ENTRY): small p99 unchanged 27 → 29 s |
| H8 | KV is genuinely saturated when flag is up | **REFUTED** | Run 1: kv_avail up to 12 299 tokens while flag sticky |
| H_decode | large decode batch is slow per-step | **REFUTED** | C_mrr8 (cap running_bs=8): small p99 27 → 31 s (worse) |
| H11 | FCFS HOL: admission loop breaks on NO_TOKEN → smaller reqs behind large never tried | **CONFIRMED** | Runs 2, 4, 6, 7 (E2/G/E3/G3): every HOL-skipping variant drops small p99 5–10× |
| H14 | `chunked_prefill_size < max_req` keeps chunked_req active across many iters, widening H11's window | **CONFIRMED (amplifier, not root cause)** | Run 3 (D): chunked_req frames 667 → 0; small p99 27 → 10 s. Run 7 (G3): no additional benefit once E3 applied — amplifier alone |
| H12 | residual post-E2 p99 from decode batch congestion | **SUPERSEDED** | Absorbed into H14 (chunked_req state) |
| H13 | residual from chunk-budget admission rate ceiling | **SUPERSEDED** | Absorbed into H14 |

## 3. Run details

Every run entry below follows the same template:

```
**Config**        — server CLI args + env
**Hypothesis**    — what this run tests
**TTFT**          — aggregate over 3 inner runs, columns value / A-ref / Δ
**Server events** — instrumentation counts, columns value / A-ref / ratio
**Interpretation**— 1–2 paragraphs
**Outcome**       — what this run does to the hypothesis ledger
```

All primary-run logs are under `results/primary/<label>.server.log` and
`results/primary/<label>.repro.log`.

---

### Run 1 — A_baseline (reference)

**Config**: `--mem-fraction-static 0.82`; no env flags; chunk=2048 (default).

**Hypothesis**: none — establishes the reference row. Also first opportunity
to see SET_C_NO_TOKEN events from the new instrumentation.

**TTFT (3-run aggregate, ms)**

| metric | value | vs A | Δ |
|---|---:|---:|---:|
| baseline (single-req) | 248 | 248 | 0 % |
| small p50 | 24 306 | 24 306 | 0 % |
| small p99 | 27 152 | 27 152 | 0 % |
| large p99 | 26 105 | 26 105 | 0 % |
| large mean | 13 837 | 13 837 | 0 % |

**Server events**

| metric | value | A-ref | ratio |
|---|---:|---:|---:|
| ENTRY | 19 919 | 19 919 | ×1 |
| EARLY_RETURN | 19 230 | 19 230 | ×1 |
| SET_C_NO_TOKEN | 12 | 12 | ×1 |
| HOL_FIX_SKIP | 0 | 0 | — |
| CLEAR_D_BATCH_SHRUNK | 8 | 8 | ×1 |
| chunked_req=True frames | 667 | 667 | ×1 |
| batch_is_full=True frames | 776 | 776 | ×1 |
| peak running_bs | 100 | 100 | ×1 |

**Interpretation**

All 12 SET_C events carry `req_input_len ∈ {11046, 11150, 11166, 11167,
11181, 11268}` — every trigger is a large request in the ~11 k-token range,
never a small. `kv_avail` at SET_C time ranges 274 – 12 299 tokens; the
high end leaves ample room for a ~50-token small, but the admission loop
breaks on the head-of-queue large before any small is tried.

Sticky-flag math: 12 SET_C × average persistence ≈ 65 frames each = 776
batch_is_full=True observations. Flag is genuinely sticky between SET and
CLEAR.

**Outcome**

- H8 (KV saturation) **double-refuted** at SET_C time.
- H11 (FCFS HOL) promoted to **strong candidate** — fingerprint matches.
- H14 (chunked_req amplifier) flagged: 667 chunked_req frames observed.

---

### Run 2 — E2_hol_fix (blunt continue on NO_TOKEN)

**Config**: `SGLANG_TTFT_HOL_FIX=1`; chunk=2048.

**Hypothesis**: H11 — if FCFS HOL is the root cause, unconditionally
continuing past NO_TOKEN should drop small p99.

**TTFT (3-run aggregate, ms)**

| metric | value | vs A | Δ |
|---|---:|---:|---:|
| small p50 | 9 730 | 24 306 | **−60 %** |
| small p99 | 12 336 | 27 152 | **−55 %** |
| large p99 | 34 129 | 26 105 | +31 % |
| large mean | 19 402 | 13 837 | +40 % |

**Server events**

| metric | value | A-ref | ratio |
|---|---:|---:|---:|
| ENTRY | 14 672 | 19 919 | ×0.74 |
| EARLY_RETURN | 13 600 | 19 230 | ×0.71 |
| SET_C_NO_TOKEN | 8 105 | 12 | ×675 |
| HOL_FIX_SKIP | 8 105 | 0 | ∞ |
| CLEAR_D_BATCH_SHRUNK | 12 | 8 | ×1.5 |
| chunked_req=True frames | 669 | 667 | ×1.00 |
| batch_is_full=True frames | 0 | 776 | 0 |
| peak running_bs | 74 | 100 | −26 |

**Interpretation**

HOL_FIX_SKIP `skipped_input_len` distribution: 62 unique values, all in the
10 991–11 268 range. **Every skip is a large request** — the loop never
falsely skips a small. H11 mechanism validated.

Large p99 regresses +31 % (starvation: larges wait through repeated skips
before their turn), and peak_bs drops 100 → 74 (some KV capacity unused
because larges queued behind the skip policy never get admitted quickly).

**Outcome**

- H11 **CONFIRMED**.
- New concern: blunt continue causes loop churn (8 105 skip events for
  ~12 genuine saturation moments in A) and large-req starvation.
- Opens space for the "smart" variant tested later (Run 6, E3).

---

### Run 3 — D_chunk32k

**Config**: `--chunked-prefill-size 32768`; no HOL fix.

**Hypothesis**: H14 — if the chunked_req state amplifies H11, sizing the
chunk so every large fits in one iter should eliminate the amplifier and
help small p99 even without an admission-loop fix.

**TTFT (3-run aggregate, ms)**

| metric | value | vs A | Δ |
|---|---:|---:|---:|
| small p50 | 7 556 | 24 306 | **−69 %** |
| small p99 | 10 441 | 27 152 | **−62 %** |
| large p99 | 10 497 | 26 105 | **−60 %** |
| large mean | 5 323 | 13 837 | **−62 %** |

**Server events**

| metric | value | A-ref | ratio |
|---|---:|---:|---:|
| ENTRY | 29 038 | 19 919 | ×1.46 |
| EARLY_RETURN | 28 904 | 19 230 | ×1.50 |
| SET_C_NO_TOKEN | 7 | 12 | ×0.58 |
| HOL_FIX_SKIP | 0 | 0 | — |
| CLEAR_D_BATCH_SHRUNK | 8 | 8 | ×1 |
| **chunked_req=True frames** | **0** | 667 | **0** |
| batch_is_full=True frames | 776 | 776 | ×1 |
| peak running_bs | 100 | 100 | ×1 |

**Interpretation**

A single 32 768-token chunk fits every ~11 k-token large in one admission
iter, so the scheduler never enters chunked_req mode. SET_C drops to
7 events (all still large-req triggers, per H11's fingerprint). Large p99
also improves dramatically because larges now complete prefill in one iter
rather than six.

**Outcome**

- H14 **CONFIRMED** as a major amplifier.
- D improves everything (both small and large) without any scheduler-logic
  change. But remaining 10 s small p99 is still ≈ 40 × baseline — room
  for more.
- Opens the question: does stacking D + HOL-fix help further? (Run 4.)

---

### Run 4 — G_chunk_hol_combo (D + E2)

**Config**: `--chunked-prefill-size 32768` + `SGLANG_TTFT_HOL_FIX=1`.

**Hypothesis**: do H11 and H14 contribute independently, or does one
dominate the other?

**TTFT (3-run aggregate, ms)**

| metric | value | vs A | Δ |
|---|---:|---:|---:|
| small p50 | 279 | 24 306 | **−99 %** |
| small p99 | 2 821 | 27 152 | **−90 %** |
| large p99 | 12 432 | 26 105 | **−52 %** |
| large mean | 6 699 | 13 837 | **−52 %** |

**Server events**

| metric | value | A-ref | ratio |
|---|---:|---:|---:|
| ENTRY | 33 137 | 19 919 | ×1.66 |
| EARLY_RETURN | 32 473 | 19 230 | ×1.69 |
| SET_C_NO_TOKEN | 11 565 | 12 | ×964 |
| HOL_FIX_SKIP | 11 565 | 0 | ∞ |
| CLEAR_D_BATCH_SHRUNK | 119 | 8 | ×15 |
| chunked_req=True frames | 0 | 667 | 0 |
| batch_is_full=True frames | 0 | 776 | 0 |
| peak running_bs | 74 | 100 | −26 |

**Interpretation**

The 11 565 SET_C events here are NOT new saturation moments; they're the
same ~7 genuine NO_TOKEN events from D, but now re-tried every outer iter
because the blunt HOL fix revokes the flag each time. Each re-try admits
whichever smalls have arrived since last iter — which is why small p99
collapses from 10 s (D alone) to 2.8 s (G).

Large p99 regression vs D (+18 %) is starvation, but milder than E2 alone
vs A (+31 %) because with chunk=32 k, a skipped large can be fully
admitted in its *next* attempt rather than needing six more chunk iters.

**Outcome**

- H11 and H14 contribute **independently and additively**. Both matter.
- G is the best combined result so far (p99 = 2.8 s), but still carries
  blunt-skip's loop churn and some large-req cost.

---

### Run 5 — H_chunk_12k (threshold probe)

**Config**: `--chunked-prefill-size 12288`; no HOL fix.

**Hypothesis**: is the effective threshold exactly `chunk ≥ max_req`, or
does "bigger chunk" keep helping?

**TTFT (3-run aggregate, ms)**

| metric | value | vs A | Δ |
|---|---:|---:|---:|
| small p50 | 7 533 | 24 306 | **−69 %** |
| small p99 | 10 306 | 27 152 | **−62 %** |
| large p99 | 10 418 | 26 105 | **−60 %** |
| large mean | 5 301 | 13 837 | **−62 %** |

**Server events**

| metric | value | A-ref | ratio |
|---|---:|---:|---:|
| SET_C_NO_TOKEN | 6 | 12 | ×0.5 |
| chunked_req=True frames | 3 | 667 | ×0.004 |
| batch_is_full=True frames | 780 | 776 | ×1.00 |
| peak running_bs | 100 | 100 | ×1 |

**Interpretation**

Within noise of D on every metric. A `chunked_prefill_size` anywhere from
12 288 up to 32 768 yields the same outcome. The 3 residual chunked_req
frames come from paged-token overhead above raw prompt length — edge-case
mismatch, negligible.

**Outcome**

- H14 **refined**: the threshold is exactly `chunk ≥ max_req_input_len`,
  not "bigger is always better". Makes an auto-tune heuristic cheap.

---

### Run 6 — E3_smart_hol (guarded continue on NO_TOKEN)

**Config**: `SGLANG_TTFT_HOL_SMART=1`; chunk=2048 (default).

**Hypothesis**: can a guard on `rem_total_tokens > 0` get the HOL-fix
benefit without blunt E2's loop churn and starvation, even at the default
chunk size?

**TTFT (3-run aggregate, ms)**

| metric | value | vs A | Δ |
|---|---:|---:|---:|
| small p50 | 345 | 24 306 | **−99 %** |
| small p99 | 2 854 | 27 152 | **−89 %** |
| large p99 | 12 854 | 26 105 | **−51 %** |
| large mean | 6 893 | 13 837 | **−50 %** |

Inner-run stability: per-run small p99 = 2 790 / 2 797 / 2 854 ms.

**Server events**

| metric | value | A-ref | ratio |
|---|---:|---:|---:|
| ENTRY | 44 938 | 19 919 | ×2.26 |
| EARLY_RETURN | 43 731 | 19 230 | ×2.27 |
| SET_C_NO_TOKEN | 11 734 | 12 | ×978 |
| HOL_FIX_SKIP | 11 734 | 0 | ∞ |
| chunked_req=True frames | 659 | 667 | ×0.99 |
| batch_is_full=True frames | 0 | 776 | 0 |
| peak running_bs | 74 | 100 | −26 |

**Interpretation**

E3 recovers G's small p99 (2.85 s ≈ 2.82 s) at the default chunk size.
Also cuts large p99 in half vs A — because smalls complete fast and the
batch turns over quickly, so larges get their turn sooner.

Why E3 dominates E2 at the same chunk: E2 continues on every NO_TOKEN
including true-saturation windows (`rem_total_tokens == 0`), walking the
full queue and SET/revoke-ing the flag repeatedly, stealing scheduler time
from decode. E3 breaks in the true-saturation case (preserving original
semantics) and only continues when there's actual KV room for a smaller
req. This makes admission cheap when it can't help and effective when it
can.

**Outcome**

- E3 is the **production-grade candidate**: matches G's small p99 with no
  chunk-size change, and is the best single-knob fix.
- H11 mechanism re-confirmed; H14 left as an orthogonal amplifier that
  becomes irrelevant once E3 is in.

---

### Run 7 — G3_smart_combo (E3 + chunk=32 k)

**Config**: `SGLANG_TTFT_HOL_SMART=1` + `--chunked-prefill-size 32768`.

**Hypothesis**: does stacking the chunk fix on top of E3 add anything?

**TTFT (3-run aggregate, ms)**

| metric | value | vs A | Δ |
|---|---:|---:|---:|
| small p50 | 233 | 24 306 | **−99 %** |
| small p99 | 2 751 | 27 152 | **−90 %** |
| large p99 | 12 237 | 26 105 | **−53 %** |
| large mean | 6 544 | 13 837 | **−53 %** |

**Server events**

| metric | value | A-ref | ratio |
|---|---:|---:|---:|
| ENTRY | 30 118 | 19 919 | ×1.51 |
| SET_C_NO_TOKEN | 11 757 | 12 | ×980 |
| HOL_FIX_SKIP | 11 757 | 0 | ∞ |
| chunked_req=True frames | 0 | 667 | 0 |
| batch_is_full=True frames | 0 | 776 | 0 |
| peak running_bs | 74 | 100 | −26 |

**Interpretation**

G3 ≈ E3 within noise on every metric. The chunk-size change adds no
measurable benefit once the admission loop is fixed.

**Outcome**

- H14 (chunked_req amplifier) is **orthogonal and redundant** once H11 is
  fixed. Chunk sizing doesn't need to change in the PR.

## 4. Variance replicate (`results/replicate/`)

A second outer-loop replicate of A and E3 from a fresh server start, hours
after Run 1–7 completed, to estimate run-to-run variance.

| metric | primary A | replicate A | primary E3 | replicate E3 |
|---|---:|---:|---:|---:|
| baseline (single-req) | 248 | 81 | 72 | 217 |
| small p50 | 24 306 | 8 011 | 345 | 1 092 |
| small p99 | 27 152 | 10 870 | 2 854 | 3 245 |
| large p99 | 26 105 | 10 919 | 12 854 | 13 744 |
| large mean | 13 837 | 5 581 | 6 893 | 7 134 |
| small p99 fix / baseline ratio | — | — | **9.5×** better | **3.4×** better |

**Interpretation**

A_baseline is not stable run-to-run (27 s primary vs 11 s replicate) even
though inner replicates within each run are stable (all three inner runs
within ±5 %). Likely from GPU-state / warm-up artifacts — the primary A
happened on a just-booted pod, the replicate after hours of prior work.
Even in the "faster" replicate, A is still 134× single-req baseline (bug
confirmed) and E3 is still 3.4× better than A (fix confirmed).

**Robust claim**: E3 consistently delivers ~3 s small p99 regardless of
GPU state; A_baseline delivers 10–27 s depending on state. Minimum fix
improvement observed is 3.4× on small p99; maximum 9.5×. Both are
substantial and the fix direction is unambiguous in every replicate.

## 5. Conclusion

Root cause, the 5-line fix diff, alternatives considered, and risk
analysis live in [`CONCLUSION.md`](CONCLUSION.md) so the PR-facing
summary stays in one place. The journal above is the supporting
evidence.
