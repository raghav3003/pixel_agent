# Dataset & taxonomy survey — gold-trajectory-conditioned error analysis

**Date:** 2026-09-22 · **Author:** Raghav (with Claude) · **Status:** draft for decision

Research question (2026-08-29 meeting): *How much do successful (gold) trajectories
improve automated error analysis of a CUA, and what properties of the gold set drive
that improvement?* This memo fixes (1) the dataset and (2) the taxonomy inputs for
the first full experiment.

---

## 1. What the dataset must provide

| Need | Why |
|---|---|
| **Gold trajectories with screenshots + actions** on benchmark tasks | The judge's reference; the RQ's independent variable |
| **Model trajectories on the same tasks**, with pass/fail known | The thing being diagnosed |
| **Human error labels** on (some of) the failures | Validation target for judge-with-gold vs judge-without-gold |
| **No live environment needed to judge** | We are on Bridges; OSWorld VMs have been the blocker for months |

## 2. Candidates

| Dataset | Gold trajectories | Model trajectories | Human error labels | Env needed | Size / license | Verdict |
|---|---|---|---|---|---|---|
| **OSWorld-Verified** ecosystem: [`xlangai/ubuntu_osworld_verified_trajs`](https://huggingface.co/datasets/xlangai/ubuntu_osworld_verified_trajs) + [OSWorld-Human](https://github.com/WukLab/osworld-human) + [`CyT1ng/cua_debugger_traj`](https://huggingface.co/datasets/CyT1ng/cua_debugger_traj) | **(a)** OSWorld-Human text steps for all 361 tasks (human, no screenshots); **(b)** evaluator-verified *successful* AI trajectories with screenshots from the trajectory corpus (15+ models, incl. Sonnet 4.6 / Opus 4.7 / Kimi K2.6) | 60+ packages, 15+ models (OpenCUA 7B/32B/A3B, UI-TARS, Sonnet 3.7/4/4.5/4.6, o3, Qwen2.5-VL, Kimi, Doubao, …), each with `traj.jsonl` (action + CoT) + per-step PNG + `result.txt`; 3–13 GB per package | **204 failed trajectories** (Sonnet 4.5 ×144, Gemini 2.5 Pro ×30, Qwen3.5-VL ×30) with human `root_error_step`, taxonomy tag (31 subtypes), evidence, correction, confidence — from CUADebug (arXiv 2608.02643) | **None** for judging | MIT + Apache-2.0; corpus 500 GB total but we need ~10–30 GB | **PRIMARY** |
| [AgentRewardBench](https://huggingface.co/datasets/McGill-NLP/agent-reward-bench) + [WebArena](https://github.com/web-arena-x/webarena/blob/main/resources/README.md) / VisualWebArena human Playwright traces | **Real human executions** with screenshots: 179 WA tasks (78% success), 233 VWA tasks | 1,408 trajectories, 4 LLM agents (GPT-4o, Claude 3.7 Sonnet, Llama-3.3-70B, Qwen2.5-VL-72B), per-step screenshots | success / side-effect / looping / optimality only — **no error type or step** | None for judging | 38.4 GB, custom license; human traces on Google Drive | **SECONDARY** — the cross-benchmark generalization leg; the only option with genuinely human screenshot trajectories |
| AgentNetBench (OpenCUA) | 100 human demos w/ screenshots | Offline, teacher-forced step prediction — no real failed trajectories | none | none | — | No (not trajectory-level failure) |
| ClawBench, A3/AITK | Human reference runs | Live web / Android; bulk download unverified | none | Yes | — | Later, if a third benchmark is needed |
| Mind2Web, WebLINX, AITW, WebChain, PC Agent-E, VideoCUA, UI-Vision | Human demos (training data) | none on benchmark tasks | none | — | — | No |
| OSWorld 2.0 (arXiv 2606.29537) | none | 108 long-horizon tasks, rollouts "to be released" | none | — | — | No (1.6 h median human time; too long for a judge study) |
| `razzant/ouroboros-osworld-verified-opus5` | 327/361 successes | results + prompts only, **no screenshots** | none | — | 112 MB | No (useful only as a task-difficulty prior) |

### Why the OSWorld bundle wins

1. **The failed-trajectory set comes pre-labeled.** CUADebug's 204 annotations give us a
   human validation target on day one. 151 unique OSWorld task IDs, all inside the
   361-task verified list; 7 overlap the lab's existing 30-task pilot pin.
   Distribution: Reasoning & Control 110 · Perception 36 · Grounding 25 · Infeasible 20 ·
   System 13. Root step is early (median step 7, median 33% into the trajectory) —
   exactly the "earliest error is primary" regime we want.
2. **Gold trajectories exist without running a VM.** For most of those 151 tasks, some
   model in the corpus has an evaluator-verified success (`result.txt == 1.0`) with a
   screenshot per step. That is precisely the "AI-assisted gold" the 08-29 plan asked
   for, with 100% success guaranteed per trajectory by construction.
3. **The gold set becomes a controlled variable.** We can hand the judge, for the same
   failed trajectory: G0 nothing · G1 OSWorld-Human text steps · G2 successful AI
   trajectory (screenshots + actions) · G3 human-note-guided frontier run (Raghav's
   Human Agent, subset). The RQ's "which properties matter" is answered by the
   G1/G2/G3 contrast (modality, source, length, alignment with the failed path).
4. **It preserves everything the lab already built** — OSWorld pin, OpenCUA adapter,
   eval-bundle vendoring, review UI — while removing the oracle-screenshot blocker
   from the critical path.

### Verified after download (2026-09-22)

- All 204 CUADebug trajectories load: 6,632 steps, 0 missing screenshots, every
  `root_error_step` within range.
- **CUADebug's 144 Claude trajectories are the same run** as the corpus package
  `claude-sonnet-4-5-20250929_50steps` (identical per-step timestamps). That package
  has 361 episodes / 203 successes (56%), so G2 gold for the 144 failed tasks must come
  from *other* agents (Opus 4.7, OpenCUA, Kimi, …), not from Claude 4.5.
- Episode index (`goldJudge/data/index/episodes.parquet`): **2,008 episodes** —
  CUADebug 204; Opus 4.7 100-step 361 (313 ✓); Claude 4.5 50-step 361 (203 ✓);
  OpenCUA-7B 50-step 3 seeds × 361 (101/95/96 ✓). Gold index: G1 for 361/361, G2 for
  **319/361** tasks (313 via Opus 4.7). CUADebug failures with G2 gold: Claude 105/144,
  Gemini 26/30, Qwen 21/30. The Opus package logs no reasoning text (actions +
  screenshots only) but includes `eval.log` with the evaluator's metric trace.
- OSWorld metadata coverage is complete: all 361 `test_nogdrive` tasks have a task JSON
  and an OSWorld-Human reference at the pinned commits; 26 tasks are in `test_infeasible`.

### Known limitations

- CUADebug's Claude set ran at a 50-step horizon on a different harness (Anthropic
  computer-use tool, 1280×720 model coords ×1.5). Loaders must normalize.
- Their labels use *their* taxonomy (single tag + root step). We map, not adopt.
- Their annotators saw no gold trajectory; a reference-conditioned re-annotation of a
  subset is still needed to measure human-with-gold agreement.
- G2 gold is AI-produced; G1 is human but text-only; only G3 is human-guided *and*
  visual. That is a feature for the RQ, but it must be stated plainly in the paper.

## 3. Taxonomy survey

| Source | Scope | Top level | Leaves | Root-cause rule | Multi-label |
|---|---|---|---|---|---|
| **Lab `failureTaxonomy.md`** (frozen 2026-07-10) | small-model CUA, OSWorld | Perception/grounding · Cognitive/planning (+ meta `evaluator_mismatch`, `propagated_failure`) | 16 | first failure step `t*` | all-applicable, ranked (08-10) |
| **CUADebug** (2608.02643) | CUA, OSWorld | P Perception · G Grounding & Interaction · R Task Reasoning & Control · S External/System · IF Infeasible | 31 | earliest step introducing a *new* causal error | single tag |
| **OSWorld 2.0** (2606.29537) | frontier CUA, long tasks | Information grounding · Timing · Domain artifacts · Verification gaps · State drift · Model-specific | 6 | — | — |
| **How Benchmarks Mis-Score CUAs** (2607.28367) | evaluator audit, 150 traj across OSWorld/WA/VWA/WorkArena/AssistantBench | Evaluator false-negative (10.7%) · Broken task (4.7%) · genuine: Planning / Execution / Verification | 3 tiers | — | — |
| **AgentRx** (2602.02475) | tool agents | Hallucination · Tool misuse · Logic · Incomplete · Constraint violation · Latency · External · Insufficient context · Cascading | 9 | localization step | — |
| **AgentErrorTaxonomy** (2509.25370) | ALFWorld/GAIA/WebShop | Memory · Reflection · Planning · Action · System | 12 | root-cause step | single |
| **TRAIL** (2505.08638) | GAIA/SWE-bench traces | Reasoning · System execution · Planning/coordination | ~12 | span + impact level | multi |
| **Model or Harness?** (2607.28802) | general agents | edges model↔{user, context, memory, tool, peer, environment}, fault side | 41 | — | — |
| **Agent S2** error analysis | CUA | Planning · Grounding · Interaction · Navigation | 4 | — | — |
| **LongRCA Bench** (2608.15242) | long-horizon | no taxonomy; root = earliest decisive unrepaired error | — | earliest step | — |

**Synthesis for our taxonomy (to be drafted as step 3):**

- Every CUA-specific taxonomy converges on the same top level: **Perception → Grounding/Interaction → Reasoning/Planning/Control → Environment/System**, plus a bucket for **task-infeasible / evaluator artifact**. Our frozen taxonomy lacks the last two and folds grounding into perception; CUADebug lacks the fine perception leaves (icon recognition, text-matching bias, resolution) that matter for *small* models.
- Root-cause definition should follow CUADebug/LongRCA: *earliest step that introduces a new error that is not repaired and is causally upstream of the failure.* This matches the user's rule "earliest error is primary".
- Labels: **root step + primary leaf at the root step + secondary leaves** (co-occurring at the root, or downstream consequences flagged `propagated_from=<step>`). This reconciles the lab's all-applicable ranking with CUADebug's single tag.
- The gold trajectory should not add leaves; it adds **evidence types** (divergence step, milestone coverage, missing prerequisite action) that the judge cites. Keep "deviated from reference" out of the label set (human reference is non-binding — already decided 2026-07-10).

## 4. Proposed next actions

1. **Decide dataset** (this memo) → download CUADebug (5.4 GB) + 2–3 OSWorld-Verified packages: one *failing* small model the lab already uses (OpenCUA-7B 50-step, 12 GB) and one *strong* model as G2 gold source (Sonnet 4.6 / Opus 4.7 100-step, 3.7 GB; Kimi K2.6, 4.8 GB).
2. **Loaders**: unify CUADebug (3 schemas) and xlangai `traj.jsonl` into one trace schema; index `result.txt` per task per package; build the gold index (task → best successful trajectory).
3. **Taxonomy draft** (`docs/research/…-taxonomy-v2-draft.md`), with a CUADebug→ours mapping table so the 204 labels become a validation set.
4. **Judge**: vLLM on Bridges H100 (`-A cis260247p`), Qwen3-VL-32B-class open model; prompt = task + eval semantics + failed trajectory + {G0|G1|G2|G3} + taxonomy → `{root_step, primary, secondary[], evidence, confidence}`.
5. **Metrics**: root-step exact / ±2, family accuracy, leaf accuracy, joint; per-condition; then human-with-gold re-annotation on a 30-trajectory pilot.
