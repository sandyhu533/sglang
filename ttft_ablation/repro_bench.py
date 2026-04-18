#!/usr/bin/env python3
"""Self-contained repro for SGLang issue #22831 — no fuzzer/trace deps.

Replicates the "asymmetric prefill" pattern that triggers the sticky
`batch_is_full` flag:

  - A burst of `--num-large` large-prompt requests fired at t=0.
    These reserve enough KV cache to push the scheduler into A_ALLOCATABLE_ZERO
    or B_CAN_RUN_SATURATED (i.e. sets batch_is_full=True).
  - A steady stream of `--num-small` short-prompt requests arriving during
    that saturation window. Under the bug, their TTFT stays elevated even
    after the batch has shrunk enough to admit them.

Output lines are intentionally shaped to be parsed by `parse_logs.py`:
  - `Baseline: <ms>`
  - `Small burst p50: <ms>`
  - `Small p99: <ms>`, `Large p99: <ms>` (last occurrence is authoritative)

CLI stays compatible with `run_ablation.sh`:
  python3 repro_bench.py --base-url http://localhost:30001 --runs 3
"""

from __future__ import annotations

import argparse
import asyncio
import random
import string
import sys
import time
from typing import List, Optional, Tuple

import httpx


def p99(values: List[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[max(0, int(len(values) * 0.99) - 1)]


def p50(values: List[float]) -> float:
    if not values:
        return 0.0
    return sorted(values)[len(values) // 2]


def random_prompt(target_words: int) -> str:
    # Word-ish tokens separated by spaces. Not real language — we only need
    # the tokenizer to emit ~target_words tokens and the model to prefill them.
    # Seeded per call so repeated runs aren't radix-cache hits.
    return " ".join(
        "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 8)))
        for _ in range(target_words)
    )


async def send_request(
    client: httpx.AsyncClient, url: str, prompt: str, max_tokens: int
) -> Tuple[Optional[float], Optional[str]]:
    """Send one completion request and measure TTFT. Returns (ttft_ms, err)."""
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    sent_at = time.time()
    first_token_at: Optional[float] = None
    try:
        async with client.stream("POST", f"{url}/v1/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    if first_token_at is None:
                        first_token_at = time.time()
    except Exception as exc:
        return None, str(exc)
    if first_token_at is None:
        return None, "no tokens received"
    return (first_token_at - sent_at) * 1000, None


async def run_burst(
    url: str,
    num_large: int,
    num_small: int,
    large_words: int,
    small_words: int,
    large_out: int,
    small_out: int,
    small_start_ms: int,
    small_interval_ms: int,
) -> Tuple[List[float], List[float], int]:
    """Fire one burst: large-burst at t=0, small-stream from small_start_ms."""
    small_ttfts: List[float] = []
    large_ttfts: List[float] = []
    errors = 0

    async with httpx.AsyncClient(timeout=180.0) as client:
        tasks: List[asyncio.Task] = []
        t0 = time.time()

        async def track(coro, bucket: List[float]):
            nonlocal errors
            ttft, err = await coro
            if err is None and ttft is not None:
                bucket.append(ttft)
            else:
                errors += 1

        # Large burst at t=0 — fire all at once to maximize KV pressure.
        for _ in range(num_large):
            prompt = random_prompt(large_words)
            tasks.append(
                asyncio.create_task(
                    track(send_request(client, url, prompt, large_out), large_ttfts)
                )
            )

        # Small stream staggered.
        for i in range(num_small):
            target = t0 + (small_start_ms + i * small_interval_ms) / 1000.0
            delay = target - time.time()
            if delay > 0:
                await asyncio.sleep(delay)
            prompt = random_prompt(small_words)
            tasks.append(
                asyncio.create_task(
                    track(send_request(client, url, prompt, small_out), small_ttfts)
                )
            )

        await asyncio.gather(*tasks, return_exceptions=True)

    return small_ttfts, large_ttfts, errors


async def flush_cache(url: str) -> None:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(f"{url}/flush_cache", params={"timeout": 5.0})
    except Exception:
        pass


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://localhost:30001")
    parser.add_argument("--runs", type=int, default=3)
    # Workload knobs — defaults tuned for a 24GB 4090 with mem_fraction=0.82.
    parser.add_argument("--num-large", type=int, default=40)
    parser.add_argument("--num-small", type=int, default=120)
    parser.add_argument("--large-words", type=int, default=3500)
    parser.add_argument("--small-words", type=int, default=40)
    parser.add_argument("--large-out", type=int, default=64)
    parser.add_argument("--small-out", type=int, default=16)
    parser.add_argument("--small-start-ms", type=int, default=300,
                        help="ms after t=0 before the small stream starts")
    parser.add_argument("--small-interval-ms", type=int, default=50,
                        help="ms between small requests")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    random.seed(args.seed)

    url = args.base_url.rstrip("/")

    # 1. Baseline TTFT — single small request, cold path.
    async with httpx.AsyncClient(timeout=30.0) as client:
        baseline_ms, err = await send_request(
            client, url, random_prompt(args.small_words), args.small_out
        )
    if baseline_ms is None:
        print(f"ERROR: baseline request failed ({err}). Is the server up?")
        return 1
    print(f"Server: {url}")
    print(f"Workload: {args.num_large} large ({args.large_words} words) + "
          f"{args.num_small} small ({args.small_words} words)")
    print(f"Baseline: {baseline_ms:.0f} ms (single small request)")
    print()

    all_small: List[float] = []
    all_large: List[float] = []
    total_errors = 0

    for run in range(args.runs):
        await flush_cache(url)
        await asyncio.sleep(1.0)
        small, large, errors = await run_burst(
            url,
            num_large=args.num_large,
            num_small=args.num_small,
            large_words=args.large_words,
            small_words=args.small_words,
            large_out=args.large_out,
            small_out=args.small_out,
            small_start_ms=args.small_start_ms,
            small_interval_ms=args.small_interval_ms,
        )
        all_small.extend(small)
        all_large.extend(large)
        total_errors += errors
        sp99, lp99 = p99(small), p99(large)
        err_str = f"  ({errors} errors)" if errors else ""
        print(f"  Run {run + 1}/{args.runs}: "
              f"Small p99={sp99:.0f}ms  Large p99={lp99:.0f}ms"
              f"  (small samples={len(small)}, large samples={len(large)})"
              f"{err_str}")
        await asyncio.sleep(2.0)

    # Aggregate summary — parse_logs.py takes the LAST p99 occurrences, so
    # these lines are the authoritative numbers for the ablation table.
    print()
    print("=" * 60)
    if all_small:
        sp50 = p50(all_small)
        sp99 = p99(all_small)
        smean = sum(all_small) / len(all_small)
        ratio = sp99 / max(baseline_ms, 1)
        print(f"Small requests ({len(all_small)} samples):")
        print(f"  Baseline : {baseline_ms:.0f} ms")
        print(f"  Small burst p50: {sp50:.0f} ms")
        print(f"  Small p99: {sp99:.0f} ms ({ratio:.1f}x baseline)")
        print(f"  Small mean: {smean:.0f} ms")
    if all_large:
        lp99 = p99(all_large)
        lmean = sum(all_large) / len(all_large)
        print(f"Large requests ({len(all_large)} samples):")
        print(f"  Large p99: {lp99:.0f} ms")
        print(f"  Large mean: {lmean:.0f} ms")
    if total_errors:
        print(f"Errors: {total_errors} requests failed")

    print()
    if all_small:
        sp99 = p99(all_small)
        if sp99 > 5000:
            print(f"BUG CONFIRMED: small p99={sp99:.0f}ms >> baseline={baseline_ms:.0f}ms")
        else:
            print(f"Within 5s threshold (small p99={sp99:.0f}ms).")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
