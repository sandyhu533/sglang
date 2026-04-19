#!/usr/bin/env bash
# Throughput validation sweep for issue #22831 (pre-registered in CONCLUSION.md).
#
# Compares baseline (A) vs proposed fix (E3, SGLANG_TTFT_HOL_SMART=1) vs
# fix+chunk (G3) on ShareGPT under sustained saturation. 3 concurrency
# levels × 3 seeds → 9 bench cells per config.
#
# Per config: boot server once, run all 9 bench_serving cells against it,
# kill server. ~2 hr total on a single 4090.
#
# Prereq: patches/apply_patches.py applied (env-gated; A run uses no env).
#
# Usage:  ./run_throughput.sh [config]
#   config ∈ {A, E3, G3, all}   default: all (= A + E3; G3 is explicit-only)
#
# G3 was excluded from the default sweep because the TTFT ablation already
# showed G3 ≈ E3 orthogonally — running G3 here would not change the
# "does E3 regress throughput?" answer. Pass `G3` explicitly to run it.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
PORT="${PORT:-30001}"
NUM_PROMPTS="${NUM_PROMPTS:-1000}"
CONCURRENCIES="${CONCURRENCIES:-32 64 128}"
SEEDS="${SEEDS:-0 1 2}"
OUTDIR="${OUTDIR:-./results/throughput/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUTDIR"
# Resolve to absolute now — the bench client `cd`s to $BENCH_CWD before
# running, so any relative path here would break under the subshell.
OUTDIR="$(cd "$OUTDIR" && pwd)"

# Same KV baseline as run_ablation.sh for an apples-to-apples comparison.
COMMON_ARGS="${COMMON_ARGS:---mem-fraction-static 0.82}"

# bench_serving's import chain (sglang.benchmark.datasets) is shadowed by
# namespace-package resolution when cwd == /workspace/sglang. Run the
# client from /tmp so the editable install resolves cleanly.
BENCH_CWD="${BENCH_CWD:-/tmp}"

launch_server() {
  local label="$1"; shift
  local extra_env="$1"; shift
  local extra_args="$*"

  echo ">>> [$label] launching server: $COMMON_ARGS $extra_args  (env: $extra_env)"
  # shellcheck disable=SC2086
  env $extra_env \
    python -m sglang.launch_server \
    --model "$MODEL" --port "$PORT" \
    --log-level info \
    $COMMON_ARGS \
    $extra_args \
    >"$OUTDIR/$label.server.log" 2>&1 &
  SERVER_PID=$!

  for i in $(seq 1 1200); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "!!! [$label] server died during startup — tail of log:"
      tail -20 "$OUTDIR/$label.server.log" >&2
      return 1
    fi
    if curl -fsS --max-time 5 "http://localhost:$PORT/health" >/dev/null 2>&1; then
      echo ">>> [$label] server ready after ${i}s"
      return 0
    fi
    sleep 1
  done
  echo "!!! [$label] server failed to become ready within 1200s"
  kill -9 "$SERVER_PID" 2>/dev/null || true
  return 1
}

kill_server() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
  lsof -ti:"$PORT" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
  sleep 2
}

trap kill_server EXIT INT TERM

run_bench() {
  local label="$1"; local conc="$2"; local seed="$3"
  local outfile="$OUTDIR/${label}_c${conc}_s${seed}.json"
  local logfile="$OUTDIR/${label}_c${conc}_s${seed}.log"

  if [[ -s "$outfile" ]]; then
    echo ">>> [$label c=$conc s=$seed] SKIP (existing $outfile)"
    return 0
  fi

  echo ">>> [$label c=$conc s=$seed] bench_serving num_prompts=$NUM_PROMPTS"
  if ! ( cd "$BENCH_CWD" && python -m sglang.bench_serving \
           --backend sglang \
           --base-url "http://localhost:$PORT" \
           --model "$MODEL" \
           --dataset-name sharegpt \
           --num-prompts "$NUM_PROMPTS" \
           --max-concurrency "$conc" \
           --request-rate inf \
           --seed "$seed" \
           --output-file "$outfile" \
           --disable-tqdm \
           >"$logfile" 2>&1 ); then
    echo "!!! [$label c=$conc s=$seed] bench FAILED — tail:" >&2
    tail -40 "$logfile" >&2
    return 1
  fi
}

run_config() {
  local label="$1"; shift
  local env_vars="$1"; shift
  local server_args="$*"

  echo "======================================================================"
  echo "CONFIG $label  env=[$env_vars]  args=[$server_args]"
  echo "======================================================================"

  launch_server "$label" "$env_vars" $server_args

  local rc=0
  for conc in $CONCURRENCIES; do
    for seed in $SEEDS; do
      run_bench "$label" "$conc" "$seed" || rc=$?
      if [[ $rc -ne 0 ]]; then break; fi
    done
    if [[ $rc -ne 0 ]]; then break; fi
  done

  kill_server

  if [[ $rc -ne 0 ]]; then
    echo "!!! [$label] aborting (rc=$rc)" >&2
    exit $rc
  fi
  echo ">>> [$label] DONE — outputs in $OUTDIR/${label}_*.json"
}

TARGET="${1:-all}"

if [[ "$TARGET" == "A" || "$TARGET" == "all" ]]; then
  run_config "A" "" ""
fi

if [[ "$TARGET" == "E3" || "$TARGET" == "all" ]]; then
  run_config "E3" "SGLANG_TTFT_HOL_SMART=1" ""
fi

if [[ "$TARGET" == "G3" ]]; then
  run_config "G3" "SGLANG_TTFT_HOL_SMART=1" "--chunked-prefill-size 32768"
fi

echo ">>> All throughput cells complete. Outputs: $OUTDIR"
echo ">>> Next: python3 parse_throughput.py $OUTDIR"
