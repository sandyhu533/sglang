#!/usr/bin/env bash
# Issue-repro (#22831) paired sweep for A vs E3.
#
# Uses repro_bench.py's bimodal workload (40 large + 120 small requests,
# co-firing at t=0) — the same workload where the TTFT ablation showed
# HOL blocking deterministically. This is the "upper bound on E3's impact"
# measurement referenced in the PR body.
#
# Per config: boot server once, run repro_bench.py with --seed in SEEDS,
# kill. Output: <OUTDIR>/<label>_s<seed>.repro.log per cell.
#
# Usage:  ./run_issue_repro.sh [config]
#   config ∈ {A, E3, all}   default: all

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
PORT="${PORT:-30001}"
SEEDS="${SEEDS:-0 1 2 3 4}"
RUNS="${RUNS:-3}"  # inner runs per seed (averaged inside repro_bench.py)
OUTDIR="${OUTDIR:-./results/issue_repro/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUTDIR"
OUTDIR="$(cd "$OUTDIR" && pwd)"

COMMON_ARGS="${COMMON_ARGS:---mem-fraction-static 0.82}"

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

run_repro() {
  local label="$1"; local seed="$2"
  local outlog="$OUTDIR/${label}_s${seed}.repro.log"

  if [[ -s "$outlog" ]]; then
    echo ">>> [$label s=$seed] SKIP (existing $outlog)"
    return 0
  fi

  echo ">>> [$label s=$seed] repro_bench.py (seed=$seed, runs=$RUNS)"
  if ! python3 "$(dirname "$0")/repro_bench.py" \
         --base-url "http://localhost:$PORT" \
         --runs "$RUNS" \
         --seed "$seed" \
         >"$outlog" 2>&1; then
    echo "!!! [$label s=$seed] repro FAILED — tail:" >&2
    tail -30 "$outlog" >&2
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
  for seed in $SEEDS; do
    run_repro "$label" "$seed" || rc=$?
    if [[ $rc -ne 0 ]]; then break; fi
  done

  kill_server

  if [[ $rc -ne 0 ]]; then
    echo "!!! [$label] aborting (rc=$rc)" >&2
    exit $rc
  fi
  echo ">>> [$label] DONE — outputs in $OUTDIR/${label}_s*.repro.log"
}

TARGET="${1:-all}"

if [[ "$TARGET" == "A" || "$TARGET" == "all" ]]; then
  run_config "A" "" ""
fi

if [[ "$TARGET" == "E3" || "$TARGET" == "all" ]]; then
  run_config "E3" "SGLANG_TTFT_HOL_SMART=1" ""
fi

echo ">>> All repro cells complete. Outputs: $OUTDIR"
