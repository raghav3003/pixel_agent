#!/usr/bin/env bash
#SBATCH -J gj-luna-subgoals
#SBATCH -A cis260247p
#SBATCH -p RM-shared
#SBATCH --ntasks-per-node=4
#SBATCH -t 12:00:00
#SBATCH -o /ocean/projects/cis260247p/rgupta19/logs/luna_subgoals_%j.out
# Subgoal pipeline with an OpenAI API model (default gpt-6-luna), stages smoke5 -> h50 -> all 204.
# Caption/track calls run without reasoning; extract/consolidate/localize with THINK effort (Responses API).
# API calls only (no GPU), but long drivers get killed on login nodes, so submit it:
#   cd goldJudge && sbatch scripts/run_openai_subgoals.sh
# Overrides via env: VERSION, EFFORT, THINK, WORKERS, STAGES, ENV_FILE. Resumable: rerun to continue.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

PY=/ocean/projects/cis260247p/rgupta19/envs/pixel/bin/python
VERSION="${VERSION:-gpt6luna-sg-v0.2}"
EFFORT="${EFFORT:-none}"
THINK="${THINK:-medium}"
WORKERS="${WORKERS:-8}"
STAGES="${STAGES:-smoke5 h50 all}"
ENV_FILE="${ENV_FILE:-config/openai.env}"
LOGDIR="/ocean/projects/cis260247p/rgupta19/logs/luna_subgoals_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOGDIR"
echo "version=$VERSION effort=$EFFORT think=$THINK workers=$WORKERS stages=$STAGES logs=$LOGDIR"

COMMON=(--version "$VERSION" --env-file "$ENV_FILE" --reasoning-effort "$EFFORT" --think-effort "$THINK" --workers "$WORKERS")
for S in $STAGES; do
  extra=(); [ "$S" != all ] && extra=(--pilot "data/index/$S.json")
  ok=0
  for pass in 1 2; do  # 2nd pass retries failed units only
    "$PY" scripts/subgoals.py all "${COMMON[@]}" "${extra[@]}" >> "$LOGDIR/$S.log" 2>&1 && { ok=1; break; }
    grep -q "authentication error" "$LOGDIR/$S.log" && { echo "API key rejected — fix $ENV_FILE"; exit 1; }
  done
  sed -n '/== localization/,/== by agent/p' "$LOGDIR/$S.log" | tail -30
  [ "$ok" = 0 ] && [ "$S" = smoke5 ] && { echo "smoke5 failed — stopping; see $LOGDIR/smoke5.log"; exit 1; }
done
for pass in 1 2; do "$PY" scripts/agent_goals.py run "${COMMON[@]}" >> "$LOGDIR/agent_goals.log" 2>&1 && break; done
"$PY" scripts/agent_goals.py score --version "$VERSION" | tee -a "$LOGDIR/agent_goals.log"
"$PY" scripts/subgoal_cost.py "$VERSION" --price-in 0.10 --price-out 0.50 || true
