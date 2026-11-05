# Subgoal-based error analysis (started 2026-10-30)

**Why.** Handing a whole gold trajectory to a judge did not help (paired tests on the 204 CUADebug
failures: G1/G2 ≈ G0 on step localization; see `data/metrics/gpt6luna-*`). This track uses the
gold material differently: (1) extract every trajectory's actions as replayable text, (2) derive a
per-task list of **required subgoals**, (3) track which subgoals each failed run reaches, and
(4) put the error at the first required subgoal the agent never reaches, or reaches and then undoes.

## 1. Replayable action logs (`goldjudge.actions`, `scripts/extract_actions.py`)

Deterministic, no model calls. All six sources (Claude, Gemini, Qwen, Opus, OpenCUA, OSWorld-Human)
map to one action grammar:

```
CLICK "target" @(x,y) +mods | DOUBLE_CLICK | TRIPLE_CLICK | RIGHT_CLICK | DRAG @(x,y) -> (x2,y2)
SCROLL n @(x,y) | TYPE "text" | PRESS key xN | HOTKEY ctrl+a | KEY_DOWN/KEY_UP | WAIT s | DONE | FAIL | CODE <src>
```

- Executed code is parsed with `ast`, **never executed**. keyDown/keyUp pairs fold into PRESS/HOTKEY or
  click modifiers; one-`press()`-per-character typing becomes TYPE; padding sleeps are dropped.
- Non-pyautogui steps (subprocess, uno, wmctrl) stay as `CODE` with the source verbatim, so they replay.
- Ops that were no-ops on the machine are flagged (`!warn`): Claude's `keyDown('left left left')`,
  OpenCUA's untranslated `computer.tripleClick`.
- `ActionLog.to_pyautogui()` regenerates code per step (coordinate replay, same layout);
  the caption stage fills `target` (the element under the click, described without coordinates) and
  `effect` (visible change), so a grounding model can replay a log on a shifted layout.
- Coverage on all 2,008 episodes + 369 human references: `data/actions/coverage.csv`.

## 2. Subgoals: what counts

A subgoal is a **state** of the machine that a screenshot can confirm ("the Steam cart lists 'The
Dota 2 Official Soundtrack'"), never an action ("click Add to cart"). Each has
`necessity` (essential | conditional | optional), `depends_on` (true precedence only),
`persistent` (must still hold at the end), `observable_forms`, and `success_check`.
Only **essential** subgoals localize errors: if any route can skip a state, missing it says nothing.

Necessity applies to the state, and a state can have several forms ("a place to change the preference is
open": settings page, config editor, or the file in a text editor). Prompt v0.1 demanded route-independent
states without this, and collapsed to ~1.2 subgoals per task on smoke5 (just the evaluator's end
condition, often not on screen), which makes localization vacuous: "never reached, step 1". v0.2 adds the
forms, a phase walk (reach, locate, change, commit, final state) and screen-only checks.

## 3. Making subjective extraction as reliable as possible

| Safeguard | How | Where |
|---|---|---|
| Anchor in the evaluator | Final subgoals come from the evaluator spec + metric source; others by backward chaining | `EXTRACT_SYSTEM` rule 2 |
| Explicit necessity test | "Could a competent user finish without this state ever being true?" | rule 3 |
| State, not route | Rules 1, 2 (`observable_forms` lists each route's form) | prompts |
| Multiple evidence views | Independent extractions from *task only*, *+OSWorld-Human*, *+captioned AI success*, merged with per-subgoal `support` and `necessity_votes` | `extract`, `consolidate` |
| Empirical counterexamples | Track the subgoals over every successful run of the task; an essential subgoal a success never reaches is **refuted**; order edges a success violates are dropped | `verify` |
| Ablation of the above | Localization is scored with and without verification (`rule_verified` vs `rule_unverified`) | `score_subgoals.py` |
| Cross-model agreement | Same pipeline with gpt-6-luna and Qwen3-VL-32B; compare sets per task | planned |
| Human audit | Raghav + Abdoul label necessity on a sample of sets; agreement vs the model | planned |

Limitation: 40 of the 151 CUADebug tasks have no successful AI run (only OSWorld-Human text, which has
no screenshots), so their subgoals stay `unverified`; 89 have exactly one success, so one checker miss
refutes a subgoal. `gold_hits` keeps the per-success evidence; `gold_seen_in_extraction` marks the
success that the `ai` view saw (not held out).

## 4. Localization rule (`core.locate_failure`)

Screenshot *t* is the observation **before** action *t*.

- **never_reached** — the first essential subgoal (topological order) that never holds; window =
  [screenshot where its prerequisites were first seen, last step]; rule point estimate = window start.
- **lost** — every essential subgoal was reached, but a persistent one stops holding for good at
  screenshot *t*; the action at *t−1* undid it; window = ±3 steps.
- **all_reached** — every essential subgoal holds yet the evaluator failed: the set is too coarse or
  the error is a wrong value; window = [last progress, end]. The share of this mode measures how
  complete the subgoal sets are.

An LLM then picks the root step **inside** the window (`LOCALIZE_SYSTEM`), seeing the subgoal status,
the window's screenshots (with click marks), actions, captions and the agent's thoughts.

## 5. The agent's own subgoals (`subgoals/intents.py`, `scripts/agent_goals.py`)

The task subgoals say what had to happen; the reasoning chain says what the agent thought it was doing.
Two calls per failed run, on top of its track:

1. **Extract, blind**: reasoning + actions only, no task subgoals and no screenshots, so goals are not rewritten to
   fit the list they are compared to, and claims stay separate from outcomes. Each goal has a step span,
   `claimed_done`, `abandoned`, and `inferred` (the agent never stated the purpose).
2. **Align**: each goal is `required` / `supporting` / `unnecessary` / `contradicts` against the task subgoals, and
   `completed` is judged from the screenshot after its last step, never from the claim.

Derived per required subgoal (`intents.classify`): `reached`, `false_completion` (not seen, but claimed), `attempted`
(not seen, worked on), `omitted` (not seen, no goal served it). This separates failure kinds: *omitted* or a
contradicting goal points to reasoning, *attempted* to execution, *false completion* to misread progress.
`divergence_step` is the earliest contradicting goal or false claim. Reported: share of steps by alignment per
agent, the fate of the first missed subgoal against the human error family, and the divergence step against the
human root step.

Limitation: reasoning detail differs by agent (median chars per step: Gemini ~730, Claude ~180, Qwen ~110, and Qwen
logs only an "Action:" line), so goals are compared within agent, and `inferred` is reported. Stated reasoning may
also not be what drove an action.

## 6. Text-only subgoals, agent and gold (`subgoals/textgoals.py`, `scripts/text_subgoals.py`)

A simpler replacement under test (2026-11-02): sections 2-5 derive required states from the evaluator and check them
on screen, which does not measure what the agent was trying to do, and scored below the plain judge.

1. **Extract**: instruction + per-step text -> ordered subgoals with purpose, `find`/`do`, steps, and how the text says
   each ended. One prompt for both sides, with no screenshots, evaluator, captions or outcome. Agent text is reasoning +
   actions; gold text is the OSWorld-Human steps, the only source on all 151 tasks with usable text (Opus 4.7 logs
   "Executing GUI action.", so it has no reasoning to read).
2. **Align**: gold list vs agent list, text only. Each gold subgoal is `finished` (claimed, or a later dependent stage
   started), `unfinished` or `missing`; each agent goal `serves`, is `extra`, or `conflicts`.
3. **Rule**: first gold subgoal not finished, or an earlier conflicting goal, is the divergence: `knowledge` (only ever
   searched), `attempted`, `planning` (never worked on), `conflict`, or `unexplained` (text says all done; no step).

Checks: the whole chain runs twice (test-retest), and human-derived gold is compared with a second gold wherever a
successful run has real reasoning.

4. **Screen check** (`text_subgoals.py screen`, repeat 1 only): (a) each claimed-done subgoal that serves gold is checked
   against the screenshot after it -> `false_claim`; (b) each long search (>= 4 steps) gets up to 6 screenshots and the
   actions: was the needed control visible, and did the agent use it? Visible and unused -> `perception`; a text
   `knowledge` candidate whose target was visible and used stays `knowledge`. Earliest event wins.
   v0.1 of (b) asked only "was it visible?": 49 of 67 searches said yes, mostly because the agent found it and used it;
   v0.2 adds the "used" question (17 visible-but-unused). Outputs `screen/` (v0.1, kept) and `screen_v0.2/`.

Results on h50 (49 labelled, gpt-6-luna, 2026-11-05): step within +-2 is .163 text-only (27 predicted), .245 with the
screen check (38 predicted), vs constant step 3 .388, judge G0 .408, judge G1 .490. The screen check halves
"unexplained" (22 -> 11). Labels do not line up with the human CUADebug families (e.g. 1 of 8 `perception` is human P).

## 8. Procedure study (2026-11-05)

Four research passes (procedure audit, data diagnosis on dev, CUADebug label semantics, literature) found:
- **Screenshot bug, pipeline-wide**: OSWorld's `lib_run_single.py` writes `step_<t>.png` from the observation returned by
  `env.step(action)`, i.e. AFTER action t. Every prompt here (judge bundles included) labelled it "before step t".
  Loaders now also set `Step.screen_before` / `screen_after`; `screenshot_path` is unchanged so older runs stay reproducible.
- **Label**: CUADebug's `root_error_step` = earliest step whose executed action introduces a new causal error (not inherited,
  not a later-repaired slip), placed on the step that acts on the wrong choice; step 1 and a final premature DONE occur.
  Exact inter-annotator agreement 85.8% before adjudication; CUADebug's own localizer: 37-63% within +-2.
- **Why subgoal divergence fails**: the root is the step where a wrong action/choice is committed, usually inside a stage the
  alignment calls "finished" (26/48 dev) or a wrong approach that still "serves" a purpose-level gold step. Segmentation is
  fine (root exactly at an agent span start 41%, ~2x chance), selection is the problem (right span in 7/48).

Design, fixed before results: dev = h50 (49 labelled), held-out = `data/index/test154.json` (154; 19 of its tasks also occur
in dev with another agent); primary metric +-2 (as CUADebug), also exact, +-1, +-5; paired sign tests on +-2 against the
new plain judge; the winner is run once on the held-out set. Methods (`localizers.py`, `scripts/localize_ideas.py`), all
with the CUADebug label definition and screens labelled "after step t":

| id | method | idea |
|---|---|---|
| A | `judge`, `judge_g1` | plain judge (+ OSWorld-Human steps): new baselines |
| B | `checkpoint` | gold subgoals as checkpoints; earliest step that breaks one for good |
| C | `spansel` | agent spans as candidates; name the failed evaluator check, pick the span, then the step |
| D | `docheck` | result of every `do` span judged on screen; first wrong one, then the step |
| E | `--samples k` + vote | densest +-2 window over k runs of the best method |
| F | counterfactual | "would fixing this step have been enough?" over the top candidates (if time allows) |

Dev results (h50, n=50, gpt-6-luna medium reasoning, 2026-11-05/06; +-2 / exact):

| method | first run | mean of 5 runs (sd) | note |
|---|---|---|---|
| A `judge` | .42 / .32 | .408 (.046) / .284 | old G0 judge: .40 / .22 |
| A `judge_g1` | .50 / .32 | .440 (.035) / .288 | old G1 judge: .48 / .34 |
| B `checkpoint` | .48 / .38 | .440 (.032) / .324 | best exact |
| C `spansel` | .34 / .28 | - | below plain judge |
| D `docheck` | .44 / .24 | - | |
| E vote over 5 runs | .40-.46 | - | no gain; lowers exact |
| E ensembles across methods | .40-.50 | - | no gain |
| F `counterfactual` | .36 / .26 | - | candidates hold the root only 64%, picks it 56% of those |
| constant step 3 | .38 / .06 | | |

Run-to-run sd (~.04, about 2 of 50) is as large as the gaps between A, B and the G1 judge, so single runs cannot rank
them; none differs significantly from A. Subgoal-structured localizers (C, D, F, and the section 6 rule) do not beat a
judge that reads the whole run; gold helps a little as context (G1, checkpoints), not as an alignment target.

**Held-out check (test154, one run each) reversed the dev picture** (+-2 / exact):

| method | reasoning | test154 | dev h50 |
|---|---|---|---|
| old judge G0 (`gpt6luna-fixed-v0.3`) | none | **.519** / .318 | .40 / .22 |
| old judge G1 | none | .513 / **.338** | .48 / .34 |
| old judge G1, screens labelled "after" (`gpt6luna-fixed-after-v0.3`) | none | **.539** / .325 | .38 / .18 |
| old judge G0, screens labelled "after" | none | .487 / .318 | .36 / .18 |
| old judge G0 (`gpt6luna-think-fixed-v0.3`) | medium | .422 / .253 | .44 / .30 |
| constant step 3 | | .442 / .123 | .38 / .06 |
| B `checkpoint` (new harness) | medium / none | .422 / .435 | .48 / .30 |
| A `judge_g1` (new harness) | medium / none | .416 / .364 | .50 / .38 |
| A `judge` (new harness) | medium / none | .403 / .338 | .42 / .36 |

- The new harness loses to the old judge on held-out (G0: p=.01 at medium, p<.01 without reasoning) and does not beat the
  constant. Medium reasoning makes judges predict late (old G0 late misses .22 -> .42), and the held-out set is 79% Claude
  runs, whose roots come early (median 38% of the run in test, 14% in dev); dev had 22 Claude runs, so it hid this.
- Correcting the screenshot labels on the old judge has no significant effect (G0 -.03, G1 +.03 on test154; p=.47/.57).
  The legacy-labelled results are not materially distorted by the bug.
- The held-out set has now been looked at twice; any further claim needs a fresh split or cross-validation, and dev
  splits should be stratified by agent.

## 9. Running

```bash
# action logs (deterministic, ~90 s)
$PY scripts/extract_actions.py
# full pipeline, API (resumable; versions never overwritten)
sbatch scripts/run_openai_subgoals.sh                     # gpt-6-luna, smoke5 -> h50 -> all
sbatch scripts/bridges/run_qwen_subgoals.sbatch           # Qwen3-VL-32B on 2xH100, same stages
$PY scripts/subgoals.py <stage|all> --version V [--pilot data/index/h50.json] ...
$PY scripts/score_subgoals.py --version V [--pilot ...]   # vs human root step + judge baselines
$PY scripts/agent_goals.py run --version V ...             # the agent's own subgoals; then `score`
$PY scripts/subgoal_cost.py V --price-in 0.10 --price-out 0.50
$PY scripts/text_subgoals.py run|score|review --version V ...   # section 6; outputs data/textgoals/<V>/
$PY scripts/localize_ideas.py run --method M --version V --pilot ... ; score  # section 8; outputs data/ideas/<V>/
```

Outputs: `data/subgoals/<V>/{captions,extract,sets,sets_verified,tracks/{gold,fail},localize,metrics}`.
Metrics: exact / ±2 vs the CUADebug root step, window hit rate and width, by mode and agent, next to
the earlier judges on the same trajectories and the constant "step 3" baseline (±2 ≈ .43).
CUADebug labels are used only for scoring, never in any prompt.
