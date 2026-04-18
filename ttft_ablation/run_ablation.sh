#!/usr/bin/env bash
# Phase 1B ablation runner for SGLang issue #22831
# Runs 5 configs × N runs, captures server logs + TTFT numbers.
#
# Prereq:
#   - patches/01_debug_log.patch applied
#   - patches/02_smoke_fix_batch_is_full.patch applied (gated by env, safe to apply)
#   - repro_latency.py downloaded in this dir
#
# Usage:  ./run_ablation.sh [config]
#   config ∈ {A, B, C, D, E, F1, all}
#   default: all

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
PORT="${PORT:-30001}"
RUNS="${RUNS:-3}"
OUTDIR="${OUTDIR:-./results/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUTDIR"

# Guard: make sure repro script present
if [[ ! -f ./repro_latency.py ]]; then
  echo ">>> Fetching repro_latency.py from gist..."
  curl -fsSL -o ./repro_latency.py \
    https://gist.githubusercontent.com/Yunzez/75bccce8f066932c42f5252c192d6b0b/raw/repro_latency.py
fi

launch_server() {
  local label="$1"; shift
  local extra_env="$1"; shift
  local extra_args="$*"

  echo ">>> [$label] launching server: $extra_args  (env: $extra_env)"
  # shellcheck disable=SC2086
  env $extra_env SGLANG_TTFT_DEBUG=1 \
    python -m sglang.launch_server \
    --model "$MODEL" --port "$PORT" \
    --log-level info \
    $extra_args \
    >"$OUTDIR/$label.server.log" 2>&1 &
  SERVER_PID=$!

  # Wait for ready (up to 90s)
  for i in $(seq 1 90); do
    if curl -fsS "http://localhost:$PORT/health_generate" -X POST \
         -H 'content-type: application/json' \
         -d '{"text":"hi","sampling_params":{"max_new_tokens":1}}' \
         >/dev/null 2>&1; then
      echo ">>> [$label] server ready after ${i}s"
      return 0
    fi
    sleep 1
  done
  echo "!!! [$label] server failed to become ready"
  kill -9 "$SERVER_PID" 2>/dev/null || true
  return 1
}

kill_server() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
  # extra safety: kill anything on port
  lsof -ti:"$PORT" | xargs -r kill -9 2>/dev/null || true
  sleep 2
}

run_repro() {
  local label="$1"
  echo ">>> [$label] running repro_latency.py (runs=$RUNS)"
  python3 ./repro_latency.py \
    --base-url "http://localhost:$PORT" \
    --runs "$RUNS" \
    2>&1 | tee "$OUTDIR/$label.repro.log"
}

run_config() {
  local label="$1"; shift
  local env_vars="$1"; shift
  local server_args="$*"

  echo "======================================================================"
  echo "CONFIG $label"
  echo "env: $env_vars"
  echo "args: $server_args"
  echo "======================================================================"

  launch_server "$label" "$env_vars" $server_args
  run_repro "$label"
  kill_server
  echo ">>> [$label] DONE — logs in $OUTDIR/$label.*"
  echo ""
}

TARGET="${1:-all}"

# A — baseline (reporter's original config)
if [[ "$TARGET" == "A" || "$TARGET" == "all" ]]; then
  run_config "A_baseline" "" ""
fi

# B — disable mixed chunk (isolate H2: mix_with_running asymmetry)
if [[ "$TARGET" == "B" || "$TARGET" == "all" ]]; then
  run_config "B_no_mixed_chunk" "" "--disable-mixed-chunk"
fi

# C — force small max-running-requests (test H3: does tighter slot = faster shrink?)
if [[ "$TARGET" == "C" || "$TARGET" == "all" ]]; then
  run_config "C_mrr8" "" "--max-running-requests 8"
fi

# D — larger chunked-prefill-size (test if bigger chunks clear flag faster)
if [[ "$TARGET" == "D" || "$TARGET" == "all" ]]; then
  run_config "D_chunk32k" "" "--chunked-prefill-size 32768"
fi

# E — priority scheduling (existing knob)
if [[ "$TARGET" == "E" || "$TARGET" == "all" ]]; then
  run_config "E_priority" "" "--enable-priority-scheduling"
fi

# F1 — SMOKE FIX: reset batch_is_full each iteration (the key ablation)
if [[ "$TARGET" == "F1" || "$TARGET" == "all" ]]; then
  run_config "F1_smoke_fix" "SGLANG_TTFT_SMOKE_FIX=1" ""
fi

echo ">>> All configs complete. Results in: $OUTDIR"
echo ">>> Next step: python3 parse_logs.py $OUTDIR"
