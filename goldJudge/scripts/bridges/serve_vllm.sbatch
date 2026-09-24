#!/bin/bash
#SBATCH -J gj-vllm
#SBATCH -A cis260247p
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:h100-80:2
#SBATCH --export=ALL
#SBATCH -t 08:00:00
#SBATCH --cpus-per-task=16
#SBATCH -o /ocean/projects/cis260247p/rgupta19/logs/vllm_%j.out
#SBATCH -e /ocean/projects/cis260247p/rgupta19/logs/vllm_%j.err
#
# Serve the judge VLM with vLLM on PSC Bridges-2 (OpenAI-compatible API).
#   sbatch scripts/bridges/serve_vllm.sbatch
#   # then on the login node: export JUDGE_BASE_URL=http://$(squeue -h -j <jobid> -o %N):8000/v1
#
# Override with env: MODEL, TP, MAX_LEN, MAX_IMAGES, PORT, DTYPE
set -euo pipefail
mkdir -p /ocean/projects/cis260247p/rgupta19/logs

MODEL="${MODEL:-Qwen/Qwen3-VL-32B-Instruct}"
TP="${TP:-2}"
MAX_LEN="${MAX_LEN:-65536}"
MAX_IMAGES="${MAX_IMAGES:-32}"
PORT="${PORT:-8000}"
DTYPE="${DTYPE:-bfloat16}"

export HF_HOME=/ocean/projects/cis260247p/rgupta19/hf_cache
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export VLLM_LOGGING_LEVEL=INFO

source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda/12.6.1
ENV=/ocean/projects/cis260247p/rgupta19/envs/vllm

echo "node=$(hostname) job=$SLURM_JOB_ID model=$MODEL tp=$TP"
echo "JUDGE_BASE_URL=http://$(hostname):${PORT}/v1"
nvidia-smi --query-gpu=name,memory.total --format=csv

exec "$ENV/bin/vllm" serve "$MODEL" \
  --served-model-name judge "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --dtype "$DTYPE" \
  --max-model-len "$MAX_LEN" \
  --limit-mm-per-prompt "{\"image\": ${MAX_IMAGES}}" \
  --gpu-memory-utilization 0.92 \
  --trust-remote-code
