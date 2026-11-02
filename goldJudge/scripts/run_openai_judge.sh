#!/usr/bin/env bash
# G0/G1/G2 x {fixed, on-demand} with an OpenAI API judge (default gpt-6-luna), in stages:
#   smoke5 -> h50 (same subset as the Haiku and Qwen runs) -> all 204 labelled failures.
# A stage with any failed trajectory stops the script. Everything is resumable (same versions):
# rerun to retry. Harness settings match scripts/bridges/run_qwen_judge.sbatch.
# API calls only (no GPU), so it runs on a login node:
#   nohup scripts/run_openai_judge.sh > /dev/null 2>&1 &
# Overrides via env: TAG, EFFORT, WORKERS, MAX_TOKENS, STAGES, ENV_FILE
set -uo pipefail
cd "$(dirname "$0")/.."

PY=/ocean/projects/cis260247p/rgupta19/envs/pixel/bin/python
TAG="${TAG:-gpt6luna}"
EFFORT="${EFFORT:-none}"             # gpt-6-luna: tool calls on Chat Completions need "none"
WORKERS="${WORKERS:-3}"              # concurrent requests per run (6 runs in parallel)
MAX_TOKENS="${MAX_TOKENS:-4096}"     # same for both harnesses
STAGES="${STAGES:-smoke5 h50 all}"
ENV_FILE="${ENV_FILE:-config/openai.env}"
V_FIXED="${TAG}-fixed-v0.3"
V_OD="${TAG}-od-v0.3"
LOGDIR="/ocean/projects/cis260247p/rgupta19/logs/${TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
exec > >(tee -a "$LOGDIR/driver.log") 2>&1
echo "tag=$TAG effort=$EFFORT workers=$WORKERS max_tokens=$MAX_TOKENS stages=$STAGES logs=$LOGDIR"

COMMON=(--env-file "$ENV_FILE" --reasoning-effort "$EFFORT" --max-tokens "$MAX_TOKENS" --workers "$WORKERS")

run_all() {  # $1 = label, rest = extra args (e.g. --pilot FILE)
  local label="$1"; shift
  local pids=()
  for C in G0 G1 G2; do
    "$PY" scripts/run_judge.py --cond "$C" --judge-version "$V_FIXED" "${COMMON[@]}" "$@" \
      > "$LOGDIR/${label}_fixed_$C.log" 2>&1 & pids+=($!)
    "$PY" scripts/run_judge.py --cond "$C" --judge-version "$V_OD" "${COMMON[@]}" "$@" \
      --harness on-demand --agent-images 8 --gold-images 4 --view-budget 24 --per-call 4 --max-turns 10 \
      > "$LOGDIR/${label}_od_$C.log" 2>&1 & pids+=($!)
  done
  local rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  if grep -q "^FAILED\|Traceback\|trajectories failed" "$LOGDIR/${label}"_*.log; then
    grep -h "^FAILED\|Traceback\|trajectories failed" "$LOGDIR/${label}"_*.log | head -40; rc=1
  fi
  return $rc
}
score() {  # $1 = label, rest = optional --pilot FILE
  local label="$1"; shift
  for V in "$V_FIXED" "$V_OD"; do
    "$PY" scripts/score_judge.py --judge-version "$V" --out "data/metrics/$V/$label" "$@" > "$LOGDIR/score_${label}_$V.txt" 2>&1
    echo "== $V ($label)"; sed -n '/== pooled ==/,/== confusion/p' "$LOGDIR/score_${label}_$V.txt"
  done
}

t0=$(date +%s)
for S in $STAGES; do
  echo "=== stage $S start $(date +%T)"
  if [ "$S" = "all" ]; then args=(); else args=(--pilot "data/index/$S.json"); fi
  run_all "$S" "${args[@]}"; rc=$?
  score "$S" "${args[@]}"
  "$PY" scripts/cost_report.py --price-in 0.10 --price-out 0.50 "$V_FIXED" "$V_OD" | tail -1
  if [ $rc -ne 0 ]; then echo "=== stage $S had failures — stopping (rerun to resume)"; exit 1; fi
  echo "=== stage $S done $(date +%T)"
done
"$PY" scripts/cost_report.py --price-in 0.10 --price-out 0.50 "$V_FIXED" "$V_OD" > "$LOGDIR/cost.txt" 2>&1
echo "done in $(( $(date +%s) - t0 )) s; logs in $LOGDIR"
