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

## 7. Running

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
```

Outputs: `data/subgoals/<V>/{captions,extract,sets,sets_verified,tracks/{gold,fail},localize,metrics}`.
Metrics: exact / ±2 vs the CUADebug root step, window hit rate and width, by mode and agent, next to
the earlier judges on the same trajectories and the constant "step 3" baseline (±2 ≈ .43).
CUADebug labels are used only for scoring, never in any prompt.
