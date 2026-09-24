#!/bin/bash
#SBATCH -J gj-qwen-judge
#SBATCH -A cis260247p
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:h100-80:2
#SBATCH -t 10:00:00
#SBATCH --cpus-per-task=16
#SBATCH -o /ocean/projects/cis260247p/rgupta19/logs/qwen_judge_%j.out
#SBATCH -e /ocean/projects/cis260247p/rgupta19/logs/qwen_judge_%j.err
#
# Self-contained judge job: serve Qwen3-VL with vLLM ON THIS NODE, run every judge request
# from inside the job, score, then stop the server. The job ends as soon as the work is done,
# so the GPUs never sit idle waiting for an external client (the failure mode of jobs
# 46807831 / 46870610, which served for 8 h with no requests).
#
#   cd goldJudge && sbatch scripts/bridges/run_qwen_judge.sbatch
#
# Runs G0/G1/G2 x {fixed, on-demand} on SUBSET (default: the Haiku h50 set, for a direct
# comparison), then — if FULL=1 — the rest of the 204 labelled failures under the same
# versions (resume skips what is done). Everything is resumable: resubmit to continue.
# Overrides via env: MODEL, TP, MAX_LEN, MAX_IMAGES, SUBSET, FULL, WORKERS, TAG
set -uo pipefail

MODEL="${MODEL:-Qwen/Qwen3-VL-32B-Instruct}"
TP="${TP:-2}"
MAX_LEN="${MAX_LEN:-98304}"
MAX_IMAGES="${MAX_IMAGES:-40}"       # on-demand: 8 agent + 4 gold seed + 24 budget = 36
PORT="${PORT:-8000}"
SUBSET="${SUBSET:-data/index/h50.json}"
FULL="${FULL:-1}"
WORKERS="${WORKERS:-4}"               # concurrent requests per run (6 runs in parallel)
TAG="${TAG:-qwen3vl32b}"
V_FIXED="${TAG}-fixed-v0.3"
V_OD="${TAG}-od-v0.3"
MAX_TOKENS="${MAX_TOKENS:-4096}"      # same for both harnesses

ROOT=/ocean/projects/cis260247p/rgupta19
GJ="$ROOT/pixel_agent/goldJudge"
PY="$ROOT/envs/pixel/bin/python"
VLLM="$ROOT/envs/vllm/bin/vllm"
LOGDIR="$ROOT/logs/qwen_judge_${SLURM_JOB_ID:-local}"
mkdir -p "$LOGDIR"
cd "$GJ" || exit 1

export HF_HOME="$ROOT/hf_cache" HF_HUB_OFFLINE=1 VLLM_LOGGING_LEVEL=INFO
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda/12.6.1

echo "node=$(hostname) job=${SLURM_JOB_ID:-} model=$MODEL tp=$TP subset=$SUBSET full=$FULL"
nvidia-smi --query-gpu=name,memory.total --format=csv

# ---- 1. vLLM server (background; always killed on exit) -------------------------------------
"$VLLM" serve "$MODEL" \
  --served-model-name judge "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size "$TP" --dtype bfloat16 \
  --max-model-len "$MAX_LEN" \
  --limit-mm-per-prompt "{\"image\": ${MAX_IMAGES}}" \
  --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --trust-remote-code > "$LOGDIR/vllm.log" 2>&1 &
VLLM_PID=$!
cleanup() { echo "stopping vLLM ($VLLM_PID)"; kill "$VLLM_PID" 2>/dev/null; sleep 10; kill -9 "$VLLM_PID" 2>/dev/null; }
trap cleanup EXIT

t0=$(date +%s)
until curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" | grep -q '"id"'; do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then echo "vLLM died during startup:"; tail -50 "$LOGDIR/vllm.log"; exit 1; fi
  if [ $(( $(date +%s) - t0 )) -gt 2700 ]; then echo "vLLM not ready after 45 min — giving up"; exit 1; fi
  sleep 15
done
echo "vLLM ready after $(( $(date +%s) - t0 )) s"

export JUDGE_PROVIDER=openai JUDGE_BASE_URL="http://127.0.0.1:$PORT/v1" JUDGE_MODEL=judge JUDGE_API_KEY=EMPTY
COMMON=(--provider openai --model judge --env-file /dev/null --max-tokens "$MAX_TOKENS" --workers "$WORKERS")

# ---- 2. smoke check: one image call must succeed before any real run ------------------------
"$PY" scripts/smoke_judge.py || { echo "smoke call failed"; exit 1; }

# ---- 3. judge runs: 3 conditions x 2 harnesses, all in parallel -----------------------------
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
  grep -h "^FAILED\|Traceback\|Error" "$LOGDIR/${label}"_*.log | head -40
  return $rc
}

score() {  # $1 = label, $2 = optional --pilot FILE
  local label="$1"; shift
  for V in "$V_FIXED" "$V_OD"; do
    "$PY" scripts/score_judge.py --judge-version "$V" --out "data/metrics/$V/$label" "$@" > "$LOGDIR/score_${label}_$V.txt" 2>&1
    sed -n '/== pooled ==/,/== confusion/p' "$LOGDIR/score_${label}_$V.txt"
  done
}

run_all h50 --pilot "$SUBSET" || echo "some h50 trajectories failed (see logs; resubmit to retry)"
score h50 --pilot "$SUBSET"

if [ "$FULL" = "1" ]; then
  run_all all || echo "some trajectories failed (see logs; resubmit to retry)"
  score all
fi

"$PY" scripts/cost_report.py --price-in 0 --price-out 0 "$V_FIXED" "$V_OD" > "$LOGDIR/usage_tokens.txt" 2>&1 || true
echo "done in $(( $(date +%s) - t0 )) s; logs in $LOGDIR"
