#!/usr/bin/env bash
#SBATCH -J gj-ctx-judges
#SBATCH -A cis260247p
#SBATCH -p RM-shared
#SBATCH --ntasks-per-node=8
#SBATCH -t 6:00:00
#SBATCH -o /ocean/projects/cis260247p/rgupta19/logs/ctx_judges_%j.out
# Idea 6: the original judge prompt (screens labelled "after", no reasoning) with analysis aids added; two variants at a
# time (more image-heavy runs at once get OOM-killed). Resumable: resubmit to continue. Usage:
#   cd goldJudge && sbatch scripts/run_context_judges.sh            # VARIANTS="F:stepflags SF:subgoals,stepflags ..."
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PY=/ocean/projects/cis260247p/rgupta19/envs/pixel/bin/python
VARIANTS="${VARIANTS:-F:stepflags SF:subgoals,stepflags C:constraints SFC:subgoals,stepflags,constraints}"
run() {  # $1 = tag, $2 = comma-separated aids
  for p in 1 2; do
    "$PY" scripts/run_judge.py --cond G0 --judge-version "gpt6luna-fixed-after-ctx$1-v0.3" --screen-labels after \
      --context ${2//,/ } --env-file config/openai.env --reasoning-effort none --max-tokens 4096 --workers 6 && break
  done
}
set -- $VARIANTS
while [ $# -gt 0 ]; do
  run "${1%%:*}" "${1#*:}" & [ $# -gt 1 ] && { run "${2%%:*}" "${2#*:}" & }
  wait; shift; [ $# -gt 0 ] && shift
done
"$PY" scripts/compare_localizers.py --ref judge:gpt6luna-fixed-after-v0.3/G0 judge:gpt6luna-fixed-v0.3/G0 judge:gpt6luna-fixed-v0.3/G1 \
  judge:gpt6luna-fixed-after-v0.3/G1 $(for v in S $VARIANTS; do echo "judge:gpt6luna-fixed-after-ctx${v%%:*}-v0.3/G0"; done) \
  stepcheck:gpt6luna-step-v0.1 stepcheck:gpt6luna-step-v0.1:harm constraints:gpt6luna-cons-v0.1 const:3
