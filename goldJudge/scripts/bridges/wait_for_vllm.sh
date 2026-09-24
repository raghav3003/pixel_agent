#!/bin/bash
# Poll a vLLM Slurm job until its OpenAI endpoint answers, then print JUDGE_BASE_URL.
#   scripts/bridges/wait_for_vllm.sh <jobid> [port] [timeout_s]
set -uo pipefail
JOB="$1"; PORT="${2:-8000}"; TIMEOUT="${3:-3600}"
t0=$(date +%s)
while true; do
  STATE=$(squeue -h -j "$JOB" -o %T 2>/dev/null || true)
  NODE=$(squeue -h -j "$JOB" -o %N 2>/dev/null || true)
  if [ -z "$STATE" ]; then echo "job $JOB not in queue (finished/failed)"; exit 1; fi
  if [ "$STATE" = "RUNNING" ] && [ -n "$NODE" ]; then
    if curl -s -m 5 "http://$NODE:$PORT/v1/models" | grep -q '"id"'; then
      echo "READY node=$NODE"
      echo "export JUDGE_BASE_URL=http://$NODE:$PORT/v1 JUDGE_MODEL=judge"
      exit 0
    fi
  fi
  if [ $(( $(date +%s) - t0 )) -gt "$TIMEOUT" ]; then echo "timeout"; exit 2; fi
  sleep 20
done
