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
# Default to repro_bench.py (self-contained, no fuzzer deps). Fall back to
# repro_latency.py if the user has the fuzzer + finding JSON checked out.
REPRO_SCRIPT="${REPRO_SCRIPT:-./repro_bench.py}"
mkdir -p "$OUTDIR"

# Applied to every config so they share a common KV-memory baseline. The
# sglang default (mem_fraction_static=0.88) leaves only ~2.4 GB on a 24GB
# RTX 4090, which starves the parent (tokenizer_manager + detokenizer +
# metrics) and causes a BrokenPipe on scheduler init. 0.82 leaves ~4.4 GB
# headroom; ablation signal (A vs F1 TTFT delta) is preserved because the
# sticky-flag bug is independent of KV slot count.
COMMON_ARGS="${COMMON_ARGS:---mem-fraction-static 0.82}"

# Guard: the chosen repro script must exist.
if [[ ! -f "$REPRO_SCRIPT" ]]; then
  echo "!!! $REPRO_SCRIPT not found (set REPRO_SCRIPT=... to override)" >&2
  exit 1
fi

# Preflight: fail fast (before launching a server for 2 min) if the repro
# script can't even import its deps or resources.
echo ">>> preflight: dry-run importing $REPRO_SCRIPT..."
if ! REPRO_SCRIPT="$REPRO_SCRIPT" python3 - <<'PY'
import importlib.util, os, sys, traceback
path = os.environ["REPRO_SCRIPT"]
spec = importlib.util.spec_from_file_location("repro_mod", path)
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except SystemExit:
    pass
except Exception:
    traceback.print_exc()
    sys.exit(1)
for attr in ("FINDING_PATH",):
    p = getattr(mod, attr, None)
    if p and not os.path.exists(p):
        print(f"preflight FAILED: missing {attr}={p}", file=sys.stderr)
        sys.exit(1)
print("preflight OK")
PY
then
  cat >&2 <<EOF
!!! preflight FAILED — $REPRO_SCRIPT cannot run.
    Either fix its imports / assets, or pick a different script:
      REPRO_SCRIPT=./repro_bench.py ./run_ablation.sh ...
EOF
  exit 1
fi

launch_server() {
  local label="$1"; shift
  local extra_env="$1"; shift
  local extra_args="$*"

  # If PERSIST_SERVER=1 AND the port already has a live sglang, skip the
  # 2-minute model-load + CUDA-graph-capture dance and reuse it. Caller is
  # responsible for making sure the running server actually matches the
  # config we want to test (same CLI flags + env). Intended for dev loops
  # where only the *client* (repro_bench.py) is being iterated on.
  if [[ "${PERSIST_SERVER:-0}" = "1" ]] \
     && curl -fsS --max-time 3 "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo ">>> [$label] PERSIST_SERVER=1 and port $PORT already healthy — reusing"
    SERVER_PID=""  # empty → kill_server will not touch it
    return 0
  fi

  echo ">>> [$label] launching server: $COMMON_ARGS $extra_args  (env: $extra_env)"
  # shellcheck disable=SC2086
  env $extra_env SGLANG_TTFT_DEBUG=1 \
    python -m sglang.launch_server \
    --model "$MODEL" --port "$PORT" \
    --log-level info \
    $COMMON_ARGS \
    $extra_args \
    >"$OUTDIR/$label.server.log" 2>&1 &
  SERVER_PID=$!

  # Wait for ready (up to 1200s). --max-time on curl guards against a
  # health probe that itself hangs (scheduler wedged mid-init).
  for i in $(seq 1 1200); do
    # Detect scheduler crash early so we don't wait the full 1200s on a dead server.
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "!!! [$label] server process died during startup — tail of log:"
      tail -20 "$OUTDIR/$label.server.log" >&2
      return 1
    fi
    # GET /health returns 200 once the server is fully up. (The old POST
    # /health_generate returns 405 in current sglang — it's a GET-only route —
    # which made the wait loop spin for the full 1200s.)
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
  # PERSIST_SERVER=1 means the server was externally owned — never kill it.
  if [[ "${PERSIST_SERVER:-0}" = "1" ]]; then
    return 0
  fi
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
  # extra safety: kill anything on port
  lsof -ti:"$PORT" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
  sleep 2
}

# Always kill the server on script exit (success, failure, Ctrl-C) so a
# failed config doesn't leak a running model onto the GPU. Honored unless
# PERSIST_SERVER=1 (in which case kill_server is a no-op).
trap kill_server EXIT INT TERM

run_repro() {
  local label="$1"
  local repro_log="$OUTDIR/$label.repro.log"
  echo ">>> [$label] running $REPRO_SCRIPT (runs=$RUNS)"
  # Capture exit code instead of masking it with `tee`. Capture directly and
  # re-print on failure so tracebacks are visible on stderr for CI logs.
  if ! python3 "$REPRO_SCRIPT" \
         --base-url "http://localhost:$PORT" \
         --runs "$RUNS" \
         >"$repro_log" 2>&1; then
    echo "!!! [$label] $REPRO_SCRIPT FAILED — tail:" >&2
    tail -40 "$repro_log" >&2
    return 1
  fi
  cat "$repro_log"
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
  # Always run kill_server after repro, even if repro fails, so the next
  # config gets a clean GPU. Capture repro's exit code and propagate it.
  local rc=0
  run_repro "$label" || rc=$?
  kill_server
  if [[ $rc -ne 0 ]]; then
    echo "!!! [$label] aborting ablation at this config (rc=$rc)" >&2
    exit $rc
  fi
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
