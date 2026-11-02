#!/bin/bash
#SBATCH -J gj-qwen-subgoals
#SBATCH -A cis260247p
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:h100-80:2
#SBATCH -t 20:00:00
#SBATCH --cpus-per-task=16
#SBATCH -o /ocean/projects/cis260247p/rgupta19/logs/qwen_subgoals_%j.out
#SBATCH -e /ocean/projects/cis260247p/rgupta19/logs/qwen_subgoals_%j.err
#
# Subgoal pipeline (scripts/subgoals.py: caption -> extract -> consolidate -> verify -> track -> localize
# -> score) with Qwen3-VL served by vLLM ON THIS NODE; same self-contained pattern as run_qwen_judge.sbatch
# (server killed on exit, restarted with --enforce-eager if it crashes, every stage resumable).
#
#   cd goldJudge && sbatch scripts/bridges/run_qwen_subgoals.sbatch
#
# Stages: smoke5 -> h50 -> all 204 labelled failures (STAGES), each scored. Resubmit to resume.
# Overrides via env: MODEL, TP, MAX_LEN, MAX_IMAGES, STAGES, WORKERS, VERSION, MAX_SEQS, RESTARTS, VLLM_EXTRA
set -uo pipefail

MODEL="${MODEL:-Qwen/Qwen3-VL-32B-Instruct}"
TP="${TP:-2}"
MAX_LEN="${MAX_LEN:-65536}"
MAX_IMAGES="${MAX_IMAGES:-16}"        # caption 7, track 8, localize 12
PORT="${PORT:-8000}"
STAGES="${STAGES:-smoke5 h50 all}"
WORKERS="${WORKERS:-16}"
VERSION="${VERSION:-qwen3vl32b-sg-v0.2}"
MAX_SEQS="${MAX_SEQS:-16}"
RESTARTS="${RESTARTS:-3}"

ROOT=/ocean/projects/cis260247p/rgupta19
GJ="$ROOT/pixel_agent/goldJudge"
PY="$ROOT/envs/pixel/bin/python"
VLLM="$ROOT/envs/vllm/bin/vllm"
LOGDIR="$ROOT/logs/qwen_subgoals_${SLURM_JOB_ID:-local}"
mkdir -p "$LOGDIR"
cd "$GJ" || exit 1

export HF_HOME="$ROOT/hf_cache" HF_HUB_OFFLINE=1 VLLM_LOGGING_LEVEL=INFO
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda/12.6.1
echo "node=$(hostname) job=${SLURM_JOB_ID:-} model=$MODEL tp=$TP version=$VERSION stages=$STAGES"
nvidia-smi --query-gpu=name,memory.total --format=csv

N_STARTS=0
VLLM_PID=""
start_vllm() {
  local extra=()
  [ "$N_STARTS" -ge 1 ] && extra+=(--enforce-eager)
  N_STARTS=$((N_STARTS + 1))
  echo "$(date) starting vLLM (start #$N_STARTS ${extra[*]:-})"
  "$VLLM" serve "$MODEL" \
    --served-model-name judge "$MODEL" \
    --host 127.0.0.1 --port "$PORT" \
    --tensor-parallel-size "$TP" --dtype bfloat16 \
    --max-model-len "$MAX_LEN" --max-num-seqs "$MAX_SEQS" \
    --limit-mm-per-prompt "{\"image\": ${MAX_IMAGES}}" \
    --gpu-memory-utilization 0.92 \
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
trap 'echo "stopping vLLM ($VLLM_PID)"; stop_vllm' EXIT

t0=$(date +%s)
start_vllm || exit 1
export JUDGE_PROVIDER=openai JUDGE_BASE_URL="http://127.0.0.1:$PORT/v1" JUDGE_MODEL=judge JUDGE_API_KEY=EMPTY
"$PY" scripts/smoke_judge.py || { echo "smoke call failed"; exit 1; }

COMMON=(--version "$VERSION" --provider openai --model judge --env-file /dev/null --workers "$WORKERS" --max-tokens 3000)

run_stage() {  # $1 = label, rest = extra args. Up to 3 passes (a pass retries what failed); watchdog on vLLM.
  local label="$1"; shift
  for pass in 1 2 3; do
    "$PY" scripts/subgoals.py all "${COMMON[@]}" "$@" >> "$LOGDIR/${label}.log" 2>&1 &
    local pid=$! crashed=0
    while kill -0 "$pid" 2>/dev/null; do
      if ! vllm_ok; then sleep 20; vllm_ok || { crashed=1; break; }; fi
      sleep 30
    done
    if [ "$crashed" = 1 ]; then
      echo "$(date) vLLM down during $label pass $pass — restarting"
      kill "$pid" 2>/dev/null; sleep 5; stop_vllm
      [ "$N_STARTS" -gt "$RESTARTS" ] && { echo "giving up after $N_STARTS vLLM starts"; return 1; }
      start_vllm || return 1
      continue
    fi
    wait "$pid" && { echo "$(date) $label complete (pass $pass)"; return 0; }
    echo "$(date) $label pass $pass had failures"; grep "^FAILED" "$LOGDIR/${label}.log" | tail -5
  done
  return 1
}

for S in $STAGES; do
  case "$S" in
    all) run_stage all; rc=$? ;;
    *)   run_stage "$S" --pilot "data/index/$S.json"; rc=$? ;;
  esac
  # a smoke stage that still fails after 3 passes means a prompt/parse problem: do not burn GPU hours
  [ "$rc" != 0 ] && [ "$S" = smoke5 ] && { echo "smoke5 failed — stopping; see $LOGDIR/smoke5.log"; exit 1; }
  sed -n '/== localization/,/== by agent/p' "$LOGDIR/$S.log" | tail -30
done
# the agent's own subgoals from its reasoning (needs the fail tracks above); 2nd pass retries failures only
for pass in 1 2; do "$PY" scripts/agent_goals.py run "${COMMON[@]}" >> "$LOGDIR/agent_goals.log" 2>&1 && break; done
"$PY" scripts/agent_goals.py score --version "$VERSION" | tee -a "$LOGDIR/agent_goals.log"
echo "done in $(( $(date +%s) - t0 )) s; outputs data/subgoals/$VERSION; logs $LOGDIR"
