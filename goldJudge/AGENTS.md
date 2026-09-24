# goldJudge — stage contract

**Idea:** gold-trajectory-conditioned LLM-as-judge for computer-use agent (CUA)
failure analysis. Research question (2026-08-29 meeting): *How much do successful
(gold) trajectories improve automated error analysis of a CUA, and which properties
of the gold set drive that improvement?*

Owner: Raghav. Started 2026-09-22. Survey and dataset decision:
`docs/research/2026-09-22-dataset-and-taxonomy-survey.md` (repo root `docs/`).

## What this stage validates

Given a **failed** agent trajectory (screenshots + actions + CoT), the task spec and
evaluator semantics, and optionally a **gold** trajectory, a judge model must output
the root-cause step, a primary error leaf, secondary leaves, and evidence. We compare
judge output against human labels under gold conditions:

| Cond | Gold given to the judge |
|---|---|
| G0 | none |
| G1 | OSWorld-Human text steps (human, no screenshots) |
| G2 | evaluator-verified successful AI trajectory with screenshots (from the OSWorld-Verified corpus) |
| G3 | human-note-guided frontier run with screenshots (Raghav's Human Agent; subset) |

## Data (on PSC Bridges, allocation `cis260247p`)

```text
/ocean/projects/cis260247p/rgupta19/data/
  cua_debugger_traj/<archive>/<domain>/<task_id>/     # 204 failed + human_annotation.json
  osworld_verified_trajs/extracted/<package>/.../<domain>/<task_id>/
  osworld_src/OSWorld/            # sparse checkout @7a17d3a: evaluation_examples + evaluators
  osworld_src/osworld-human/      # @deff1a7
/ocean/projects/cis260247p/rgupta19/envs/pixel        # Python 3.11 env (pip install -e goldJudge[dev])
```

Env vars honoured by loaders: `CUADEBUG_ROOT`, `OSWV_ROOT`, `OSWORLD_SRC`.

## Layout

```text
goldJudge/
  AGENTS.md                 this contract
  pyproject.toml            package `goldjudge`
  src/goldjudge/
    schema.py               unified Trajectory / Step / Annotation / TaskMeta
    parsing.py              pyautogui + response parsing helpers
    loaders/cuadebug.py     CUADebug archives (3 agent schemas)
    loaders/osworld_verified.py   xlangai OSWorld-Verified packages (any layout)
    loaders/osworld_meta.py task JSON, OSWorld-Human, evaluator metric source
  scripts/build_index.py    one row per episode → data/index/episodes.parquet
  docs/                     taxonomy, judge protocol, runbooks
  tests/                    pytest (synthetic fixtures only — never commit data)
  data/                     indices, judge outputs, metrics (git-ignored except small summaries)
```

## Boundaries

- Never commit trajectory data or screenshots; only indices and compact summaries.
- Never run vLLM / inference on Bridges login nodes; always `-A cis260247p`.
- Never edit `errorAnalysis/` frozen docs (see `errorAnalysis/docs/GROUNDING_MANIFEST.md`).
  This stage has its own taxonomy (`docs/taxonomy_v2.md`) and does not modify the
  frozen `failureTaxonomy.md`.
- CUADebug human labels are **validation data**, never training data for the judge.
- Judge outputs are versioned by `judge_version` + gold condition; never overwrite.
- Frontier-API spend > $25 needs Matt's OK (standing lab rule).

## Default flow

1. `python scripts/build_index.py --out data/index` — episode table across sources.
2. `python scripts/build_gold_index.py` — per task: candidate gold trajectories (G1/G2/G3).
3. `python scripts/run_judge.py --cond G0|G1|G2|G3 --judge <model>` — versioned outputs.
4. `python scripts/score_judge.py` — root-step exact/±2, family/leaf accuracy, joint.
