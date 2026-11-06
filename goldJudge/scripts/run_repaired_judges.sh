#!/usr/bin/env bash
#SBATCH -J gj-repaired
#SBATCH -A cis260247p
#SBATCH -p RM-shared
#SBATCH --ntasks-per-node=8
#SBATCH -t 8:00:00
#SBATCH -o /ocean/projects/cis260247p/rgupta19/logs/repaired_%j.out
# Repaired-input original judge (docs/research/2026-11-07-round3-research-and-audit.md): task version pinned to the
# run date, screens labelled "after", Qwen's initial screen, thoughts up to 1500 chars, Gemini reflections; no reasoning.
# SAMPLES independent runs per condition (s1 = the baseline; s2.. feed the distribution readout). Resumable.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PY=/ocean/projects/cis260247p/rgupta19/envs/pixel/bin/python
SAMPLES="${SAMPLES:-5}"
COMMON=(--screen-labels after --task-version run-date --thought-chars 1500 --show-reflection --env-file config/openai.env
        --reasoning-effort none --max-tokens 4096 --workers 6)
for k in $(seq 1 "$SAMPLES"); do
  for C in G1 G0; do
    for p in 1 2; do "$PY" scripts/run_judge.py --cond "$C" --judge-version "gpt6luna-repaired-v0.3-s$k" "${COMMON[@]}" && break; done &
  done
  wait
done
