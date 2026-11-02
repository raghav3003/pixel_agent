#!/bin/bash
#SBATCH -J gj-qwen-judge
#SBATCH -A cis260247p
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:h100-80:2
#SBATCH -t 20:00:00
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
# Overrides via env: MODEL, TP, MAX_LEN, MAX_IMAGES, SUBSET, FULL, WORKERS, TAG, OTHER_JOB,
#   MAX_SEQS, RESTARTS, VLLM_EXTRA (extra `vllm serve` flags)
# FP8 on one GPU:
#   MODEL=Qwen/Qwen3-VL-32B-Instruct-FP8 TP=1 MAX_LEN=65536 WORKERS=2 TAG=qwen3vl32b-fp8 OTHER_JOB=<jobid> \
#     sbatch --gres=gpu:h100-80:1 --cpus-per-task=8 scripts/bridges/run_qwen_judge.sbatch
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

# OTHER_JOB: an alternative submission of the same work (e.g. bf16 vs FP8). Whichever starts
# first wins: cancel the other if it is still queued; exit at once if it is already running.
if [ -n "${OTHER_JOB:-}" ]; then
  OTHER_STATE=$(sacct -X -n -j "$OTHER_JOB" -o State%20 2>/dev/null | awk 'NR==1{print $1}')
  if [ "$OTHER_STATE" = "PENDING" ]; then echo "cancelling queued alternative job $OTHER_JOB"; scancel "$OTHER_JOB"
  elif [ "$OTHER_STATE" != "CANCELLED" ] && [ "$OTHER_STATE" != "CANCELLED+" ]; then
    echo "alternative job $OTHER_JOB already ran or is running (state: ${OTHER_STATE:-unknown}) — exiting"; exit 0
  fi
fi
nvidia-smi --query-gpu=name,memory.total --format=csv

# ---- 1. vLLM server (background; always killed on exit) -------------------------------------
# A watchdog in run_all restarts it if it dies (job 46968149 hit a CUDA illegal memory access
# mid-run and every later request was refused). After the first crash it restarts with
# --enforce-eager (no CUDA graphs), the usual workaround for that class of crash.
MAX_SEQS="${MAX_SEQS:-16}"            # cap concurrent sequences in the engine
RESTARTS="${RESTARTS:-3}"
N_STARTS=0
VLLM_PID=""
start_vllm() {
  local extra=()
  [ "$N_STARTS" -ge 1 ] && extra+=(--enforce-eager)
  N_STARTS=$((N_STARTS + 1))
  echo "starting vLLM (start #$N_STARTS ${extra[*]:-})"
  "$VLLM" serve "$MODEL" \
    --served-model-name judge "$MODEL" \
    --host 127.0.0.1 --port "$PORT" \
    --tensor-parallel-size "$TP" --dtype bfloat16 \
    --max-model-len "$MAX_LEN" --max-num-seqs "$MAX_SEQS" \
    --limit-mm-per-prompt "{\"image\": ${MAX_IMAGES}}" \
    --gpu-memory-utilization 0.92 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --trust-remote-code ${VLLM_EXTRA:-} "${extra[@]}" >> "$LOGDIR/vllm.log" 2>&1 &
  VLLM_PID=$!
  local t=$(date +%s)
  until curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" | grep -q '"id"'; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then echo "vLLM died during startup:"; tail -50 "$LOGDIR/vllm.log"; return 1; fi
    if [ $(( $(date +%s) - t )) -gt 2700 ]; then echo "vLLM not ready after 45 min"; return 1; fi
    sleep 15
  done
  echo "vLLM ready after $(( $(date +%s) - t )) s"
}
vllm_ok() { kill -0 "$VLLM_PID" 2>/dev/null && curl -s -m 10 "http://127.0.0.1:$PORT/health" -o /dev/null -w '%{http_code}' | grep -q 200; }
stop_vllm() { [ -n "$VLLM_PID" ] || return 0; kill "$VLLM_PID" 2>/dev/null; sleep 10; kill -9 "$VLLM_PID" 2>/dev/null; pkill -9 -f "vllm serve" 2>/dev/null; sleep 5; }
cleanup() { echo "stopping vLLM ($VLLM_PID)"; stop_vllm; }
trap cleanup EXIT

t0=$(date +%s)
start_vllm || exit 1

export JUDGE_PROVIDER=openai JUDGE_BASE_URL="http://127.0.0.1:$PORT/v1" JUDGE_MODEL=judge JUDGE_API_KEY=EMPTY
COMMON=(--provider openai --model judge --env-file /dev/null --max-tokens "$MAX_TOKENS" --workers "$WORKERS")

# ---- 2. smoke check: one image call must succeed before any real run ------------------------
"$PY" scripts/smoke_judge.py || { echo "smoke call failed"; exit 1; }

# ---- 3. judge runs: 3 conditions x 2 harnesses, all in parallel -----------------------------
launch_all() {  # $1 = label, rest = extra args; fills PIDS
  local label="$1"; shift
  PIDS=()
  for C in G0 G1 G2; do
    "$PY" scripts/run_judge.py --cond "$C" --judge-version "$V_FIXED" "${COMMON[@]}" "$@" \
      >> "$LOGDIR/${label}_fixed_$C.log" 2>&1 & PIDS+=($!)
    "$PY" scripts/run_judge.py --cond "$C" --judge-version "$V_OD" "${COMMON[@]}" "$@" \
      --harness on-demand --agent-images 8 --gold-images 4 --view-budget 24 --per-call 4 --max-turns 10 \
      >> "$LOGDIR/${label}_od_$C.log" 2>&1 & PIDS+=($!)
  done
}

run_all() {  # 3 conditions x 2 harnesses in parallel; restart vLLM + resume if it dies
  local label="$1"; shift
  while :; do
    launch_all "$label" "$@"
    local crashed=0
    while :; do
      local alive=0
      for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
      [ "$alive" = 0 ] && break
      if ! vllm_ok; then sleep 20; vllm_ok || { crashed=1; break; }; fi
      sleep 30
    done
    if [ "$crashed" = 0 ]; then
      for p in "${PIDS[@]}"; do wait "$p"; done
      break
    fi
    echo "$(date) vLLM is down during $label — stopping runs and restarting"
    grep -m3 -i "illegal memory\|CUDA error" "$LOGDIR/vllm.log" | tail -3
    for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; sleep 5
    stop_vllm
    if [ "$N_STARTS" -gt "$RESTARTS" ]; then echo "giving up after $N_STARTS vLLM starts"; return 1; fi
    start_vllm || return 1
  done
  grep -h "^FAILED\|Traceback" "$LOGDIR/${label}"_*.log | tail -20
  return 0
}

score() {  # $1 = label, $2 = optional --pilot FILE
  local label="$1"; shift
  for V in "$V_FIXED" "$V_OD"; do
    "$PY" scripts/score_judge.py --judge-version "$V" --out "data/metrics/$V/$label" "$@" > "$LOGDIR/score_${label}_$V.txt" 2>&1
    sed -n '/== pooled ==/,/== confusion/p' "$LOGDIR/score_${label}_$V.txt"
  done
}

# each stage runs twice: the 2nd pass only retries rows that FAILED in the 1st (resume skips the rest)
{ run_all h50 --pilot "$SUBSET" && run_all h50 --pilot "$SUBSET"; } || { echo "h50 stage aborted (see logs; resubmit to resume)"; score h50 --pilot "$SUBSET"; exit 1; }
score h50 --pilot "$SUBSET"

if [ "$FULL" = "1" ]; then
  { run_all all && run_all all; } || echo "full stage aborted (see logs; resubmit to resume)"
  score all
fi

"$PY" scripts/cost_report.py --price-in 0 --price-out 0 "$V_FIXED" "$V_OD" > "$LOGDIR/usage_tokens.txt" 2>&1 || true
echo "done in $(( $(date +%s) - t0 )) s; logs in $LOGDIR"
