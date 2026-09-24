# goldJudge run book (PSC Bridges-2)

All commands from `goldJudge/` with `PY=/ocean/projects/cis260247p/rgupta19/envs/pixel/bin/python`.

## 0. One-time

```bash
$PY -m pip install -e ".[dev,judge]"
$PY -m pytest -q
```

Data already on disk (see `AGENTS.md`). To add another OSWorld-Verified package:

```bash
$PY - <<'EOF'
from huggingface_hub import hf_hub_download
hf_hub_download("xlangai/ubuntu_osworld_verified_trajs", "<file>.zip", repo_type="dataset",
                local_dir="/ocean/projects/cis260247p/rgupta19/data/osworld_verified_trajs")
EOF
unzip -q <file>.zip -d /ocean/projects/cis260247p/rgupta19/data/osworld_verified_trajs/extracted/<file-stem>
# add the package -> agent id to loaders/osworld_verified.py:PACKAGE_AGENTS, then rebuild the index
```

## 1. Indices

```bash
$PY scripts/build_index.py --out data/index          # episodes.parquet
$PY scripts/build_gold_index.py --index data/index   # gold.parquet (G1/G2 per task)
```

## 2. Serve the judge (vLLM, 2×H100, charged to cis260247p)

```bash
sbatch scripts/bridges/serve_vllm.sbatch                       # Qwen/Qwen3-VL-32B-Instruct, TP=2
scripts/bridges/wait_for_vllm.sh <jobid>                       # prints export line when ready
export JUDGE_BASE_URL=http://<node>:8000/v1 JUDGE_MODEL=judge
$PY scripts/smoke_judge.py                                     # one-screenshot sanity call
```

Overrides: `MODEL=Qwen/Qwen3-VL-32B-Instruct-FP8 TP=1 sbatch --gres=gpu:h100-80:1 ...`.
Never run vLLM on the login node.

## 2b. Pipeline validation with an API model (no GPU)

`config/judge.env` (git-ignored; template `config/judge.env.example`) selects the
backend. For validation use the cheapest current model:

```
JUDGE_PROVIDER=anthropic
JUDGE_MODEL=claude-haiku-4-5
ANTHROPIC_API_KEY=...
```

Measured 2026-09-23 on the 15-trajectory pilot (`data/index/pilot15.json`):
~13k input tokens per trajectory for G0/G1, ~22k for G2 (8 gold screenshots),
6–7 s per call, **$0.80 for all 45 calls**. Run conditions **sequentially** — three
parallel runs tripped the key's concurrent-connection limit (429).

## 3. Judge runs (versioned, resumable, never overwrite)

```bash
V=qwen3vl32b-v0.1
$PY scripts/run_judge.py --cond G0 --judge-version $V --agents gemini-2.5-pro --limit 5   # pilot
$PY scripts/run_judge.py --cond G0 --judge-version $V
$PY scripts/run_judge.py --cond G1 --judge-version $V
$PY scripts/run_judge.py --cond G2 --judge-version $V      # only tasks with G2 gold
```

Outputs: `data/judge/<V>/<cond>/<agent>.jsonl` (one row per trajectory, includes the
human annotation for convenience, the raw model text, token usage, and latency).

## 3b. On-demand screenshots (Anthropic backend)

The fixed harness sends at most `--agent-images` (16) evenly spaced screenshots; on the
204 labelled failures that misses the human root step's screenshot in 67 cases (33%).
`--harness on-demand` sends a *seed* (`--agent-images`, `--gold-images`) plus every
step's text, and gives the judge a `view_screenshots(trajectory, steps)` tool — in
**every** condition (agent trajectory always, gold run only in G2/G3), so conditions
differ only in the reference. Budget: `--view-budget` extra screenshots per trajectory,
`--per-call` per call, `--max-turns` model calls (the last one has tools switched off).
Rows log `viewed_agent_steps`, `screenshot_requests`, `n_turns`; `score_judge.py`
reports `root_seen` (did the judge see the human root step's screenshot).

```bash
$PY scripts/select_subset.py --out data/index/long15.json --exclude data/index/pilot15.json
COMMON="--pilot data/index/long15.json --thinking adaptive --max-tokens 16000"
$PY scripts/run_judge.py --cond G0 $COMMON --judge-version sonnet5-long15-fixed-v0.3
$PY scripts/run_judge.py --cond G0 $COMMON --harness on-demand --agent-images 8 --gold-images 4 \
    --judge-version sonnet5-long15-od-v0.3
```

Thinking tokens count toward `--max-tokens`; with `--thinking adaptive` (or Sonnet 5's
default) keep it at 16000 or the verdict gets truncated. `--dry-run` prints a per-
trajectory prompt-token estimate (within ~5% of Haiku 4.5's measured usage).

## 4. Score

```bash
$PY scripts/score_judge.py --judge-version $V --out data/metrics/$V
```

Reports per agent and pooled: parse rate, root-step exact / ±2, v2 family and leaf
accuracy (via the CUADebug→v2 mapping), joint, and CUADebug-scheme family accuracy.

Leaf ids are only comparable within one taxonomy version. Taxonomy v2.1 (2026-09-23)
merged G1+G2 and renumbered the G family, so `haiku45-test-v0.1`/`-v0.2` (23-leaf
numbering) trigger a warning when re-scored; their leaf metrics are not comparable
with v2.1 runs. Rows now carry `taxonomy_version`.

`--reparse` re-parses the stored `raw` text with the current parser (judge outputs are
not modified). Use it after parser fixes, e.g. the 2026-09-23 fix for verdicts that come
after prose containing code/JSON or with invalid `\w`-style escapes (common in the
on-demand harness, where the model narrates between tool calls).

## 5. Cost / budget notes

Haiku 4.5 smoke test (2026-09-23, `data/index/smoke5.json`, 5 long trajectories):
fixed harness ~$0.03/trajectory, on-demand ~$0.06/trajectory (3–5 turns, ~10 extra
screenshots). Without thinking, Haiku narrates its reasoning in the final turn; one
reply ran past 8000 tokens — keep `--max-tokens` generous.


Bundle sizes (1280-px JPEGs, Qwen3-VL ≈ 1.2k tokens/image): 10–35k text chars +
7–24 images ≈ 10–30k tokens per trajectory. 204 trajectories × 3 conditions ≈ 12M
prompt tokens — minutes on 2×H100. Frontier-API runs need the $25 gate (Matt).
