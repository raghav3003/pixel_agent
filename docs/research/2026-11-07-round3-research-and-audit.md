# Round 3: pipeline audit, recent literature, and new methods for CUA root-step localization

2026-11-07. Six research passes: recent CUA/agent failure-analysis literature, adjacent-field judging methods, an
audit of the core judge pipeline, an audit of every newer harness, an error analysis of the best judge, and a study of
the label definition and its conventions. Earlier rounds: `goldJudge/docs/subgoals.md` sections 6-10,
`docs/research/2026-11-06-failure-analysis-harness-survey.md`.

## Audit findings that change earlier results
1. **Task-version mismatch (high).** Instructions/evaluators were read from an OSWorld checkout dated 2026-07-07; the
   agents ran 2025-07 (Gemini), 2025-10 (Claude), 2026-02 (Qwen). Pinning each trajectory to the commit in effect on its
   run date changes the instruction for ~31 and the evaluator for ~27 of 204 rows (e.g. Claude 3720f614: "Korean" vs
   "Atlantis/Xenothian, infeasible"). Fixed: `OSWorldMeta.load_task(commit=...)`, `run_judge.py --task-version run-date`.
2. **The section-8 harness frame was late-biased by itself** (median predicted position .50-.53 vs .24-.27 for the
   original judge and .26 for human roots, with and without reasoning). Every structured method built on it
   (checkpoint, spansel, docheck, counterfactual) was compared only against that frame; those negatives are confounded.
   The earlier explanation "medium reasoning makes judges late" holds for the original judge, not for that frame.
3. **Judge inputs lose information**: thoughts cut at 700 chars (Gemini's decision text cut in 17/30 root steps;
   Gemini's per-step reflection never shown); Qwen's initial screen never shown. Fixed as options
   (`--thought-chars`, `--show-reflection`, initial screen in "after" mode).
4. **Populations**: G2 covers 152 of 204 rows; compare on shared rows (`compare_localizers.py --shared`): G2 vs G0
   .441 vs .507 on the same 152.
5. Minor: duplicate step screenshots were picked by file-system order (now sorted, last); prefix-scan threshold grid
   lacked "never override"; the evaluator-trace prompt over-called omissions (79 vs 18 last-step roots).
   Verified correct: step indexing vs labels (all 204), label agreement across files, coordinates, verdict parsing,
   +-2 computation, G1/G2 task matching, reasoning settings, prior-fusion CV, adjudication, CUADebug reproduction.

## Evidence the new methods rest on
- CUADebug's best published setting pooled over all 204 is about +-2 .505 / exact .333 (Gemini 2.5 Pro debugger);
  our original judge with human steps: .505 / .338. Our CUADebugger reproduction with gpt-6-luna is .333.
- **Wrong part of the run, not near-misses**: G1 lands in a different agent subgoal than the root in 51.5% of cases;
  inside the right subgoal it is +-2 in 83%. The root is the first step of an agent subgoal in 43% (chance 25%).
- **Complementarity**: some of 5 main predictors is within +-2 in .75 of cases (random-size control .46); agreement
  predicts correctness (4/4 others agree: .73, n=60; 0/4: .28, n=18).
- **Conventions**: 29/34 sampled misses (85%) were convention/ambiguity cases, not wrong diagnoses (premature DONE
  labelled at DONE; strategy errors at the step the approach was chosen; first attempt of a retry loop; repaired errors
  are not roots). R4 strategy errors: +-2 .19; S7 evaluator artifacts .20. Cross-agent conventions drift.
- **Length**: +-2 .91 for <= 10 steps vs .33 for 36-50 steps (constant step 3 .29 there).
- Literature: retrieval of past labelled diagnoses is CUADebug's best variant and ErrorProbe's gain (+0.8 to +13);
  reflective prompt induction from labelled errors (GEPA); window-sum readout of step distributions and calibration
  (LLM-Rubric); coarse-to-fine temporal search (TimeSearch, ProTracer); cascade-aware rescoring (SOAP, needs white-box);
  selective prediction / conformal ranges for steps (CAEA).

## New methods (all on the repaired original judge; gpt-6-luna, no reasoning)
| id | method | grounded in |
|---|---|---|
| R0 | repaired baseline: original prompt + pinned task version + "after" labels + initial screen + 1500-char thoughts + reflections (5 runs per condition) | audit 1-3 |
| N1 | **convention induction**: per CV fold, an LLM reads the training folds' labelled cases (task, root step, human evidence, and the judge's miss) and writes a labelling-convention guide; the judge gets the guide | 85% convention misses; R4/S7 weakness; GEPA-style reflective induction |
| N2 | **retrieved labelled exemplars**: per case, 3 training-fold cases (same app first, then instruction similarity) with a condensed view around their root, the human root and its reason | CUADebug memory, ErrorProbe, KATE |
| N3 | **distribution readout**: window-sum (+-2) over independent judge runs (G1 and G0) instead of one answer | adjacent lit (window-sum, self-consistency); complementarity .75; the earlier vote was on the late-biased frame |
| N4 | **segment-then-step**: screenshots re-allocated to the agent subgoals that the first-pass runs point at, plus the subgoal table; the judge then picks the step | wrong-subgoal 51.5% vs 83% inside; TimeSearch / ProTracer |
| N5 | **candidate selection with global context**: candidates pooled from independent runs, each shown with its before/after screens; the judge picks among them (or another step), quoting the committing action and the violated requirement | union ceiling .75; FALAT/EMFA/TrajDebug; earlier adjudication used only one judge's candidates and local windows |
| N6 | **agreement-based selective prediction**: report accuracy vs coverage when answering only where runs agree | agreement .73 vs .28; CAEA |
Baselines kept alongside: original judge (G0/G1), constant step 3, CUADebug prompt-only and debugger reproductions,
CUADebug's published pooled numbers. Fairness re-run: checkpoints as context on the repaired original frame.
Deferred (GPU queue): SOAP-style white-box backward propagation on Qwen3-VL-32B.

Evaluation: all 204; one 5-fold split grouped by task (`harness2.folds_by_task`, seed 0) for anything fit or induced
from labels; +-2 primary (comparable with CUADebug), plus exact, +-1, +-5, MAE, signed mean error; paired sign tests
against R0 with Holm correction over the primary comparisons; strata: length > 10, agent, label confidence,
`chosen_debugger` seeded vs not.

## Results (2026-11-07; all 204, gpt-6-luna no reasoning; reference = R0 repaired G1; cost $9.59)

| method | +-2 | exact | notes |
|---|---|---|---|
| original judge G1 (`gpt6luna-fixed-v0.3`) | .505 | .338 | |
| R0 repaired G1, mean of 5 runs (sd) | .486 (.023) | .301 | input repairs change nothing measurable |
| R0 repaired G0, mean of 5 runs (sd) | .464 (.009) | .283 | |
| N1 induced conventions (per-fold guides) | .466 | .319 | +14/-18 vs R0 |
| **N2 retrieved labelled exemplars, mean of 4 runs (sd)** | **.501 (.021)** | **.327** | vs R0 mean: +.015 (95% CI -.02..+.05, Wilcoxon p=.65); exact +.026 (p=.24); first run alone .525 |
| N2 window vote x4 | .505 | .338 | |
| N1+N2 | .485 | .328 | |
| N3 window vote, G1+G0 x10 (densest window, most frequent step) | .471 | .299 | the first tie rule (window median) gave .515 / exact .196; neither significant |
| N4 segment focus | .456 | .265 | later (signed +1.6) |
| N5 candidate selection (pooled candidates, before/after screens) | .466 | .284 | |
| checkpoints as context on the repaired frame (G0) | .475 | .299 | = repaired G0: the section-8 checkpoint negative holds on the fair frame |
| constant step 3 | .426 | .108 | |

No comparison survives Holm correction. N6 (selective prediction): answering only when all 10 repaired runs agree
within +-2 covers 41% of failures at +-2 .723 / exact .482; >= 8 of 10: 59% coverage at .628 / .413. This is the one
robust positive: agreement across independent judge runs is a strong, free confidence signal for triage.

### Conclusions
- Nothing we built beats the single-pass judge by more than run-to-run noise (sd ~.02 on +-2). The closest is N2
  (labelled exemplars), +.015 +-2 / +.026 exact on average; worth keeping but not a headline result.
- Teaching conventions explicitly (N1) did not help even though 85% of sampled misses were convention cases; the
  guides read sensibly, so the judge does not apply written conventions reliably, while concrete examples (N2) help a
  little.
- Focusing screenshots on where the runs point (N4) or adjudicating pooled candidates (N5) does not fix wrong-subgoal
  choices; these remain the main error source (51.5%).
- With labels at 85.8% exact human agreement, cross-agent convention drift, and the best published CUADebug setting at
  the same level as our judge (~.505 +-2), the benchmark appears close to saturated for gpt-6-luna-class judges.
Not tried (needs GPU or a frontier model): SOAP-style white-box backward propagation; a stronger judge model.
