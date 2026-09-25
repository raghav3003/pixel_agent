# pixelAgent — Project Summary

**As of:** 2026-09-24 · **Written for:** readers who know the general research area but not the code.

---

## TL;DR

- **Goal:** understand *how and why* computer-use agents (CUAs) fail, and turn that into a reusable, trustworthy method for automated error analysis.
- **Phase 1** (May–Aug) built an LLM-judge pipeline for OpenCUA failures on OSWorld. It stalled on one problem: we could not produce reliable reference ("gold") trajectories. A reference-guided run succeeded on only **60 of 361** tasks, mostly because of benchmark and environment bugs, not the model.
- **Current direction** (from 2026-08-29): *how much do gold trajectories improve automated error analysis, and which properties of the gold trajectory matter?* We use a public dataset of **204 human-labelled OSWorld failures**, so we no longer have to generate our own references first.
- **Status:** the experiment setup is built and has been checked end to end. Early tests with a cheap judge (Claude Haiku 4.5) show **no benefit from gold yet**. The first real runs, with an open 32B judge on the cluster, are waiting for GPUs.

---

## 1. The objective

Benchmarks like OSWorld report one bit per task: pass or fail. That bit does not say whether the agent misread the screen, clicked the wrong spot, lost track of its plan, or was defeated by a broken task. We want an **LLM-as-judge** that reads a failed trajectory and reports:

- **where** the run first went wrong (the *root step*), and
- **why**, using a fixed failure taxonomy.

The judge only counts if it agrees with human annotators. The long-term payoff is to aim small-model CUA improvements at failure modes that are real, not at artifacts.

---

## 2. How the direction evolved

The question has narrowed several times. Each time, the previous framing turned out to rest on something unverified.

| When | Framing | What changed it |
|---|---|---|
| May | "Small VLMs for pixel-only computer use" | Too broad. Nobody knew where small models actually fail. |
| Late May | Grounding (not planning) is the small-model bottleneck | This came from benchmark aggregates, not from observed failures. |
| June | Build an error-analysis pipeline: a judge explains OpenCUA failures | A failed trace alone does not show *why* it failed. The judge needed a reference. |
| July | Give the judge the human solution from **OSWorld-Human** as a reference | Needed a screenshot for each human step, so we had to replay the human steps in the VM ourselves. |
| Aug | Ran the replay: **60/361** succeeded | Most failures were benchmark/environment bugs. Generating our own gold references became the blocker. |
| **Aug 29 → now** | **Measure how much gold trajectories help the judge, using existing labelled data** | Current direction (below). |

---

## 3. Phase 1: what was tried and learned

**Tried**
- Collected OpenCUA (3B-A3B and 7B) trajectories on OSWorld. Built a judge (Claude Sonnet 4.6), a 16-leaf failure taxonomy, and a two-annotator review workflow.
- Tried to produce gold trajectories by replaying OSWorld-Human steps in the VM, with UI-TARS-72B handling clicks.
- Checked our model-serving setup by reproducing published grounding numbers.

**Results**
- **Reference replay: 60/361 (16.6%) success.** This run should have been close to a ceiling. The failures came from (a) incomplete human instructions, e.g. never saying "press Enter"; (b) actions fired before the screen finished loading; (c) VMs that never finished starting, e.g. Chrome not open.
- **Grounding reproduction matched or beat published numbers.** Examples: ScreenSpot-V2 88.7 vs 88.8 published at 7B, and 91.5 vs 87.0 at 32B; OSWorld-G 34.8 vs 31.4 at 7B, and 48.8 vs 46.5 at 32B. So our serving setup is sound.
- **Judge labels:** only 16 of 361 episodes were labelled, all provisional. No human gold labels were ever finished.

**Conclusions**
- A meaningful share of raw OSWorld "failures" are benchmark or environment artifacts. A judge that ignores this will explain bugs as reasoning errors.
- Building our own gold references was too slow to put on the critical path.

---

## 4. Current direction: the gold-trajectory judge study

### Research question
> How much do successful (gold) trajectories improve automated CUA error analysis, and which properties of the gold trajectory drive the improvement?

### Design
We give the same judge the same failed trajectory and vary only the reference it gets:

| Condition | Reference given to the judge |
|---|---|
| **G0** | None (baseline) |
| **G1** | OSWorld-Human steps: human, text only |
| **G2** | A successful run by another AI agent on the same task, with screenshots |
| **G3** | A frontier model's run guided by the human notes, with screenshots (a subset; not built yet) |

We then compare the judge's answers with human labels. Contrasting G1, G2 and G3 separates the effects of modality (text vs. screenshots) and source (human vs. AI).

**Principle:** the gold trajectory is *context about what the task requires*, not a path the agent must match. Taking a different route is not an error.

### Data (decided 2026-09-22)
- **CUADebug** (a public dataset from a paper concurrent with ours): **204 failed OSWorld trajectories** from Claude Sonnet 4.5, Gemini 2.5 Pro and Qwen3.5-VL. Each comes with a human-labelled root step and error type. This is our validation target.
- **OSWorld-Verified trajectory corpus:** successful runs by other agents supply G2 gold for **319 of 361** tasks.
- **OSWorld-Human:** G1 gold for all tasks.
- **Why this data:** the failures are already human-labelled, and gold references exist without running an OSWorld VM (the blocker for months).
- Web benchmarks (WebArena and similar) are deferred as a later cross-benchmark check.

### Taxonomy v2 (draft)
- **22 failure types in 4 families:** Perception (what the agent *saw*), Grounding (turning a correct intent into an action), Reasoning/Planning (what it *decided*), and Environment/Benchmark (not the agent's fault).
- **Label format:** one root step with one **primary** error, plus optional **secondary** errors. A secondary error is tagged as co-occurring, as a downstream consequence of the primary, or as independent.
- It combines the lab's earlier taxonomy with CUADebug's and others, and maps cleanly onto CUADebug's labels, so the 204 human labels can be used directly.
- It still needs review by the team and a short two-person human labelling pilot before it is frozen.

### Metrics
Per condition:
- **Root step:** exact match, and within ±2 steps.
- **Error type:** family accuracy, fine-grained type accuracy, and joint (step + type) accuracy.

---

## 5. Results so far (test runs only, not paper results)

Judge: **Claude Haiku 4.5**, the cheapest model, used to test the pipeline. Taxonomy is still a draft.

**50-trajectory run, judge shown a fixed sample of screenshots:**

| | G0 (no gold) | G1 (human text) | G2 (AI run) |
|---|---|---|---|
| Root step within ±2 | **0.46** | 0.36 | 0.34 |
| Family accuracy | **0.64** | 0.56 | 0.54 |
| Fine-grained type accuracy | 0.12 | 0.12 | 0.08 |

**Observations**
- **Gold did not help this judge. If anything, it hurt root-step finding.** The pattern holds for all three source agents.
- An earlier 15-trajectory pilot, run on an older taxonomy draft, showed a small mixed gain for G1. Samples this small cannot settle it either way.
- Fine-grained error-type accuracy is low (~10%). This is consistent with the literature, where the best models reach about 11% joint accuracy on similar tasks.
- **Found a design flaw:** showing the judge a fixed sample of screenshots misses the human-labelled root-step screenshot in about **1/3 of trajectories** (67 of 204). We added a mode where the judge can *request* specific screenshots. That mode is offered in every condition, so it does not get mixed up with the effect of gold. Its 50-trajectory run stopped partway when API credits ran out.
- An earlier prompt version had the judge report where the agent first left the gold path. With a weak judge, this pulled its root-step answer toward that point, so we removed it.

---

## 6. Conclusions so far

1. **Setup:** the pipeline works end to end on real data.
2. **Early signal:** gold trajectories do not help automatically. At least for a small judge, extra reference material can distract. Whether a stronger judge benefits is the open question, and it is the interesting part of the paper.
3. **Screenshot selection matters:** which screenshots the judge sees is a confound as big as the gold condition. It has to be controlled.
4. **Error types are hard to judge:** fine-grained type accuracy is low. Family level and root step are the more realistic targets to report.

---

## 7. Current status

- **Waiting in the GPU queue:** full runs with **Qwen3-VL-32B** (open-weight judge, run on our own GPUs). The plan is G0/G1/G2 × {fixed screenshots, judge-requested screenshots}, first on the 50-trajectory subset, then on all 204.
- **Paused:** Anthropic API runs (credits exhausted). The requested-screenshot Haiku run and a planned Claude Sonnet 5 run are on hold.
- **Not started:** G3 (frontier model guided by human notes). It depends on getting reference replays in OSWorld working.

---

## 8. Next steps

1. Get the Qwen3-VL-32B results on all 204 trajectories and compare G0, G1 and G2.
2. Run a stronger judge (e.g. Sonnet 5) when budget allows, to test whether gold helps once the judge is capable enough.
3. Finalise taxonomy v2: team review, then a small two-person labelling pilot to check agreement.
4. **Human-with-gold re-annotation** (~30 trajectories). CUADebug annotators never saw a gold trajectory, so we need to measure how much gold helps *humans* too.
5. Analyse *which* gold properties matter: text vs. screenshots, human vs. AI, and length or similarity to the failed run.
6. Add G3 for a subset, and later check whether the findings hold on a web benchmark (WebArena / VisualWebArena).

---

## 9. Open questions and risks

- **Novelty:** CUADebug and other recent error-analysis papers overlap with our framing. Under the NeurIPS policy they count as concurrent work, but our contribution has to be the *gold-trajectory effect*, not the taxonomy.
- **AI-made gold:** G2 references come from AI runs and G1 is text only. Only G3 would be both human-guided and visual. The paper must say this plainly.
- **Benchmark relevance:** frontier models are now evaluated on OSWorld 2.0. Is error analysis on OSWorld mainly valuable for small models?
- **Benchmark noise:** OSWorld instruction gaps, setup bugs and evaluator errors remain. The taxonomy has an Environment family for them, but their true rate in our data is unknown.
- **Coordination:** the new study's taxonomy and scope still need sign-off from the rest of the team (Abdoul, Matt).
