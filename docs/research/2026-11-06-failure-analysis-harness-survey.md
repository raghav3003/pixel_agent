# Failure-analysis harnesses from other domains, and what transfers to CUA root-step localization

2026-11-06. Synthesis of six literature passes (LLM-agent failure attribution, software fault localization, AIOps
root-cause analysis, robotics/video failure localization, reasoning process supervision, human incident
investigation). Numbers are from the papers' own tables unless marked (M, from memory / unverified). Context and our
results: `goldJudge/docs/subgoals.md` sections 8-10.

## Where we stand
- Best: single-pass judge, original taxonomy prompt, no reasoning, +-2 = .49 (with OSWorld-Human steps .505) on the 204
  CUADebug failures. Constant step 3: .43. With n=204 one SE at +-2 is ~.035, so a real gain needs roughly >= .55.
- CUADebug's own tool-using ReAct debugger: +-2 37-63% (best on the Claude split 47.9), i.e. level with our judge.
- Label caveat: CUADebug annotators could see a machine proposal while labelling (chosen_debugger field present for
  most Gemini/Qwen files), so labels may lean towards what that backward-tracing debugger finds.
- No published harness beats an all-at-once judge by a clear margin at +-2 in a comparable setting; large reported
  gains are exact-match on Who&When, whose all-at-once baseline is weak (19.8 at +-2).

## Convergent ideas (proposed independently by several passes)

| idea | proposed by | core mechanism | evidence |
|---|---|---|---|
| Position-prior fusion | SE, AIOps, process, agents, robotics | judge's distribution over steps x prior(step | length, agent), weight fit by grouped CV | our step-3 .43; CombineFL (SE): combining FL signals ~3x top-1; StepFinder position term |
| Candidates -> local adjudication, earliest-wins in code | agents, SE, process, human, robotics | judge proposes top-k + symptom step; per-candidate local call labels ROOT / INHERITED / SYMPTOM / BENIGN with an independence test; first ROOT wins | FALAT hand-crafted +-2 26-37 vs 19.8; EMFA exact 32.8/50.0; RAFFLES 29.4->51.6 exact; Who&When hybrid best exact |
| Outcome-blind prefix scan | process, human, agents | judge sees only steps 1..t (or windows in order), no outcome; stop at first confident new error; threshold by CV | BIG-Bench Mistake: prefix 52.9 vs whole-trace 39.8 (reasoning 43.4); hindsight/outcome bias in humans (Caplan 1991) |
| Evaluator-anchored backward trace | AIOps, robotics, human | from the evaluator source name the failed check and its object (cell/file/setting); list steps that touched it; earliest wrong, uninherited write; omission fallback | REFLECT progressive structure 62.8->96.0 (sim), 70.2->80.7 planning; CIRCA +25% top-1 |
| Rule-based decision-point candidates + multiple choice | robotics, SE | commit actions (Enter, OK/Save, menu pick, typed value, DONE), app/dialog changes, screen-hash jumps; one MCQ call | ProTracer 55.4->81.0 within 1 s; narratives worth 40 points |

Lower priority: RAFFLES-style critics with a primacy critic (8-12 calls); CUADebugger tool loop with forced "assume this
is a symptom, look earlier" turns (continual search helps only on long logs); cross-run spectrum (FAMAS/Ochiai) and
action-level divergence from a successful run (as features, not answers); RCTA segment recall (LongRCA, gains n.s.
after Holm); lessons learned on a calibration split (ProTracer +Exp, needs strict CV). Not recommended: binary search
with an LLM oracle, voting / judge panels (hurt on TRAIL, no gain for us), trained tracers (need replay or injected
errors), per-step success detectors (reproduce our per-step noise), 5 Whys, organisational tiers of HFACS/STAMP.

## Why our earlier structure failed, in the literature's terms
- The root action usually executes the agent's (wrong) intent perfectly, so intent-vs-outcome checks miss it and flag
  harmless slips instead (REFLECT separates execution from planning failures).
- No recoverability test: a flagged step later repaired is not a root (ProTracer, EMFA, CUADebug definition).
- Counterfactual "would fixing it suffice" says yes for late steps too; the discriminating test is independence
  (not inherited from an earlier error), with earliest-wins enforced outside the model.
- Reasoning lets a full-trace judge narrate the visible symptom; prefix-only checking removes the symptom instead of
  asking the model to ignore it.

## Proposed test plan
Evaluate on all 204 with 5-fold CV grouped by task (test154 has been looked at twice), report exact / +-1 / +-2 /
+-5, hit@3, per agent, and paired McNemar against the original judge. Free measurements first: recall of the root in
rule-based decision-point candidate sets and in judge top-k lists; share of roots that are omissions (caps the
evaluator backward trace).

1. Judge top-5 with confidences (one run) -> position-prior fusion (cheap; also the strong baseline).
2. Candidates -> local ROOT/INHERITED/SYMPTOM/BENIGN adjudication, earliest ROOT wins (~6 calls).
3. Outcome-blind windowed prefix scan, no reasoning, stop rule tuned by CV (~N/8 + 1 calls); optionally as the
   candidate source for 2.
4. Evaluator-anchored backward trace with omission fallback (~3 calls + per-step touch labels).
Optional baseline: reproduce CUADebugger's RCA prompt (paper Appendix A.1) to gauge label anchoring.

## Key sources
Who&When 2505.00212; Who&When Pro 2607.09996; FALAT 2606.00765; EMFA 2610.11600; CHIEF 2602.23701; RAFFLES 2509.06822;
ECHO 2510.04886; A2P 2509.10401; AgentRx 2602.02475; AgentDebug 2509.25370; Continual Search 2609.13463; CUADebug
2608.02643; LongRCA/RCTA 2608.15242; CIRCA 2206.05871; RCA comparison 2408.13729; Flow-of-Action 2502.08224; FAMAS
2509.13782; AutoFL 2308.05487; FlexFL 2411.10714; CombineFL 1803.09939; Cleve & Zeller ICSE 2005; REFLECT 2306.15724;
ProTracer 2609.21369; RoboFAC 2505.12224; ViFailback 2512.02787; TimeSearch 2504.01407; GVL 2411.04549; ProcessBench
2412.06559; BIG-Bench Mistake 2311.08516; ThinkPRM 2504.16828; Caplan et al. JAMA 1991; RCA2 (NPSF); Why-Because
Analysis (Ladkin); TICK 2410.03608.

## Results (2026-11-06; all 204, 5-fold CV grouped by task, gpt-6-luna without reasoning, screens labelled "after")

Free measurements first: rule-based decision points (commit actions) contain the root exactly in only .37 of runs at
~12 candidates (.58 with big screen changes, ~16 candidates of 33 steps); most roots are plain clicks (109/204). Roots
at a DONE/FAIL step: .08; at the last step: .09 (omissions are rare).

| predictor | +-2 | exact | vs original G1 judge |
|---|---|---|---|
| original judge + OSWorld-Human steps (G1, `gpt6luna-fixed-v0.3`) | **.505** | **.338** | ref |
| original judge G0 | .490 | .294 | p=.76 |
| top-5 judge (`gpt6luna-fixed-after-top5-v0.3/G1`), its first choice | .490 | .328 | p=.76 |
| 1 judge candidates x position prior (lambda, eps fit per fold) | .485 | .299 | p=.64 |
| prior only | .353 | .074 | p<.01 |
| 2 candidates -> local ROOT/INHERITED/SYMPTOM/BENIGN, earliest ROOT | .431 | .265 | p=.06 |
| 3 outcome-blind prefix scan, threshold fit per fold (picks ~100) | .348 | .225 | p<.01 |
| 4 evaluator-anchored backward trace | .314 | .172 | p<.01 |
| constant step 3 | .426 | .108 | p=.06 |

- The judge's top-5 contains the root (+-2) in .647 of runs, so picking among its candidates could at best reach .65;
  local adjudication reaches .43, below the judge's own first choice.
- The prefix scan stays late even without seeing the outcome: an error is only flagged once its consequences are
  visible, so "remove the symptom" does not move predictions earlier; CV sets the threshold to ~100 (rarely override).
- The position prior adds nothing on top of the judge (training-fold +-2 .50-.53 vs CV .485).
Outputs: `goldJudge/data/harness2/gpt6luna-h2-v0.1/`, code `goldJudge/src/goldjudge/subgoals/harness2.py`,
`goldJudge/scripts/harness2.py`, `run_judge.py --topk`. Cost of this round ~$3.
