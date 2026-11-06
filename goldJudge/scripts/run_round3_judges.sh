#!/usr/bin/env bash
#SBATCH -J gj-r3
#SBATCH -A cis260247p
#SBATCH -p RM-shared
#SBATCH --ntasks-per-node=8
#SBATCH -t 8:00:00
#SBATCH -o /ocean/projects/cis260247p/rgupta19/logs/round3_%j.out
# Round-3 judge variants: the repaired original judge (same settings as scripts/run_repaired_judges.sh, s1) plus
# per-trajectory extras from scripts/round3.py. RUNS="tag:cond:extras-file ..." (two at a time). Resumable.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PY=/ocean/projects/cis260247p/rgupta19/envs/pixel/bin/python
RUNS="${RUNS:-n1:G1:data/round3/extras_n1.json n2:G1:data/round3/extras_n2.json n12:G1:data/round3/extras_n12.json ckpt:G0:data/round3/extras_ckpt.json}"
COMMON=(--screen-labels after --task-version run-date --thought-chars 1500 --show-reflection --env-file config/openai.env
        --reasoning-effort none --max-tokens 4096 --workers 6)
run() { IFS=: read -r tag cond file <<< "$1"
  for p in 1 2; do "$PY" scripts/run_judge.py --cond "$cond" --judge-version "gpt6luna-r3-$tag-v0.3" --extra-file "$file" "${COMMON[@]}" && break; done; }
set -- $RUNS
while [ $# -gt 0 ]; do run "$1" & [ $# -gt 1 ] && { run "$2" & }; wait; shift; [ $# -gt 0 ] && shift; done
