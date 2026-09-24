# CUA failure taxonomy v2 (goldJudge) — DRAFT 2026-09-23 (v2.1)

> **v2.1 change (2026-09-23, Raghav):** former G1 *Imprecise coordinates* and G2
> *Wrong target element* are merged into one leaf, **G1 Mis-grounded target**; the
> remaining G leaves are renumbered (old G3→G2, G4→G3, G5→G4). The v2 G family now
> lines up 1:1 with CUADebug's G1–G4. 22 leaves. Judge outputs `haiku45-test-v0.1`
> and `-v0.2` were produced under the old 23-leaf numbering.

Purpose: a label set an LLM judge (or a human) applies to a **failed** computer-use
trajectory, optionally with a **gold** trajectory for the same task, to say *where*
the run went wrong and *why*. Built from the lab's frozen 16-leaf taxonomy
(`errorAnalysis/failureTaxonomy.md`), CUADebug's 31 subtypes (arXiv 2608.02643,
204 human-labelled OSWorld failures), OSWorld 2.0's failure modes, the
"How Benchmarks Mis-Score CUAs" audit, AgentRx, TRAIL and AgentErrorTaxonomy.
Survey: `docs/research/2026-09-22-dataset-and-taxonomy-survey.md`.

Design constraints (from Raghav, 2026-09-22): comprehensive and exhaustive; not
vague, not bloated; supports **primary + secondary** errors; supports errors that
are **consequences of an earlier error**; the **earliest error is the primary cause**.

---

## 1. Label structure

```yaml
root_step: 12            # earliest step introducing a new, unrepaired, causally-upstream error
primary: G1              # the leaf that best explains what went wrong AT root_step
secondary:               # optional; each with a step and a relation to the primary
  - {leaf: P3, step: 12, relation: co-occurring}      # also true at the root step
  - {leaf: R7, step: 15, relation: downstream}        # consequence of the root error
  - {leaf: R6, step: 31, relation: independent}       # separate error, would have failed anyway
evidence: "..."          # cite the agent's own steps, screenshots, CoT quotes
correction: "..."        # what should have happened at root_step
confidence: high|mid|low
flags:                   # orthogonal, not leaves
  evaluator_doubt: false # agent appears to have met the task; evaluator may be wrong
```

**Root step definition** (aligned with CUADebug and LongRCA): the earliest step at
which the agent's executed action, given what was observable, takes the run off any
viable path to success *and the deviation is never repaired*. Earlier suboptimal but
recoverable steps are not the root. If the environment or evaluator, not the agent,
caused the failure, the root step is the first step at which that fault bit, and the
primary leaf is in family **E**.

**Primary vs secondary.** Exactly one primary leaf, at the root step. Secondary
leaves are (a) *co-occurring* — a second leaf that is also true at the root step
(e.g. a perception error that led to a wrong-target click: P3 + G1); (b)
*downstream* — later errors that only happened because of the root (looping,
premature `DONE`); (c) *independent* — later errors that would have sunk the run on
their own. Downstream leaves carry `propagated` semantics and are never the primary.

---

## 2. Families and leaves (22)

Question order when labelling a step: **E → P → G → R**. Was the environment at
fault? If not, did the agent *see* wrongly? If not, did its reasoning name the
*right* target but the executed action miss it? Otherwise the error is in what it
*decided* (including choosing the wrong target in its reasoning).

### P — Perception (what the agent saw)

| ID | Leaf | Definition | Typical evidence | Absorbs |
|---|---|---|---|---|
| **P1** | Visual hallucination | Claims to see an element, text, or state that is not in the screenshot | CoT names a button/dialog that is absent | CUADebug P1; lab Reasoning Drift (perceptual part) |
| **P2** | Misrecognition | Misreads something that *is* visible: OCR error, wrong icon→function mapping, shape/colour confusion, wrong widget type | Reads "Save" as "Send"; treats a gear as "search"; mistakes a label for a field | CUADebug P2, P3; lab Visual Confusion, Software Commonsense (Icon), Text Matching Bias (perceptual half) |
| **P3** | Observation omission | Fails to notice task-relevant visible information: error banners, dialogs, result lists, a changed value, an already-open tab | Screenshot shows a warning the CoT never mentions | CUADebug P4; OSWorld 2.0 "information grounding" |
| **P4** | Feedback misinterpretation | Sees the screen but misjudges the *outcome* of the previous action or the current UI state (thinks a click succeeded, a dialog closed, a value changed) | CoT "the file is now saved" while title bar still shows unsaved | CUADebug P5, R11 |

### G — Grounding & interaction (turning intent into an action)

**Hard rule (decided 2026-09-23):** a G leaf applies **only when the agent's reasoning
names the correct target or effect and the executed action misses it.** If the
reasoning itself names the wrong element or the wrong approach, the error is P2
(the element was misread from the screen) or an R leaf (wrong plan, goal, or target
choice) — never G. Example: in the VLC hotkey task the agent reasoned "double-click
the Play/Pause row in the Hotkey column" when the task needed the *Global* column;
that is R1/R2, not G1, because the wrong target was chosen in the reasoning.

| ID | Leaf | Definition | Typical evidence | Absorbs |
|---|---|---|---|---|
| **G1** | Mis-grounded target | Intent names the right element T; the executed action does not land on T — either a near miss just outside it (boundary, compact widget, zoom/scale drift) or a *different* element (label instead of field, neighbour in a list, wrong side of a landmark, similar-looking control) | CoT names T; click lands beside T or on another element | lab Click Region Error, Fine-Grained Manipulation, Resolution/Scale Brittleness, Location Hallucination, Spatial Reasoning Error, Text Matching Bias (action half); CUADebug G1 |
| **G2** | Inaccessible target | Acts on an element that is off-screen, occluded, disabled, or behind an overlay, instead of first scrolling / closing / enabling | Target not rendered in the pre-action screenshot | CUADebug G2 |
| **G3** | Interaction mechanics | Right target, wrong primitive or sequencing: single vs double click, typing without focus, wrong hotkey, drag/select failure, Enter never pressed, acting before the UI settled when a wait was available | Typed text appears nowhere; menu opened but key sent elsewhere | CUADebug G3; lab Fine-Grained Manipulation (mechanics half); OSWorld-Human "missing press Enter" (agent side) |
| **G4** | Distraction / misdirection | Redirected by ads, pop-ups, cookie banners, decoys, or suggestions unrelated to the goal | Clicks a promoted result / overlay | CUADebug G4 |

### R — Reasoning, planning & control (what the agent decided)

| ID | Leaf | Definition | Typical evidence | Absorbs |
|---|---|---|---|---|
| **R1** | Task misunderstanding | Misreads the instruction, drops or violates an explicit constraint, or invents a goal that was never asked for; picks an evaluator-rejected reading of an ambiguous task | Solves a neighbouring task; ignores "without changing X" | lab Goal Hallucination, Instruction Ambiguity; CUADebug R1 |
| **R2** | Plan / decomposition error | Wrong sub-goal order, missing prerequisite step, or an approach that cannot reach the goal (impossible action, wrong app/feature) | Edits the wrong document first; tries to filter before loading data | CUADebug R2, R3 |
| **R3** | Affordance / knowledge gap | Understands the goal but does not know *how* the software does it: never tries the menu, shortcut, sidebar tab, context menu, or dialog the task needs; keeps to visible/salient controls | Repeatedly opens top-level menus while the setting lives in a sidebar | lab Hidden Operation Blindness; OSWorld 2.0 "domain artifacts"; CUADebug R3 (knowledge part) |
| **R4** | Action–intent mismatch | The emitted action does not implement the stated intent: wrong parameters, wrong text typed, wrong argument, malformed or non-existent action | CoT says "type 42", action types "24"; `pyautogui.press('enter')` when intent was Ctrl+Enter | CUADebug R5, R6, R7 |
| **R5** | False belief / context loss | CoT asserts something false about task state or history that is *not* a direct misreading of the current screen: forgets progress, hallucinates a past result, loses clipboard/file/sub-goal state across steps or apps | "I already saved the file" (never did); pastes stale clipboard | lab Long-Horizon Memory, Cross-Application Context Loss, Reasoning Drift (non-perceptual); CUADebug R8, R9 |
| **R6** | Progress misjudgment | Declares success (`DONE`) or gives up (`FAIL`) when the task is not complete, or fails to verify a task-critical property it could have checked | `DONE` while the required tab is not open | CUADebug R10; OSWorld 2.0 "verification gaps"; Mis-score Tier 3 |
| **R7** | Failed recovery / looping | Detects (or should detect) that an action is not working, but repeats it or applies an ineffective fix without changing approach | Same click ≥3× with no state change; alternates between two non-working actions | lab Action Looping; CUADebug R12 |
| **R8** | Inefficient strategy | A valid but wasteful approach that exhausts the step budget or time (manual scrolling instead of Find, one-by-one edits instead of a bulk action) | Budget exhausted mid-way through a plan that would have worked | CUADebug R4 |
| **R9** | Infeasibility handling | Fails to recognise that the task is impossible and keeps acting, or wrongly declares a feasible task infeasible | Clicks around for a non-existent app; `FAIL` on a solvable task | lab Refusal/Infeasibility; CUADebug IF2, O1 |

### E — Environment, harness & benchmark (not the agent's fault)

| ID | Leaf | Definition | Typical evidence | Absorbs |
|---|---|---|---|---|
| **E1** | Environment / initialization fault | Required app or state never came up, VM misconfigured, app crashed or hung independent of the agent | First screenshot lacks the app the task config launches | Lab "VM initialization failure"; CUADebug S3, S6 |
| **E2** | Timing / race | The harness fired the next action before the UI finished updating and the agent had no reasonable way to wait; stale-screen actions | Screenshot mid-render; correct click on a not-yet-loaded page | Lab "premature next-action execution"; CUADebug S2; OSWorld 2.0 "timing". *If the agent chose not to wait when it could, use G3.* |
| **E3** | Harness / tool fault | Action translation or execution bug (coordinate scaling, dropped keystrokes), tool/API failure, step/resource limit hit for reasons outside the agent's control | Model coords correct after rescaling, executed coords wrong | CUADebug S4, S5; "Model or Harness" mistranslation |
| **E4** | External blocker | CAPTCHA, network failure, site changed since the task was written, login wall | reCAPTCHA page after a benign search | CUADebug S3 (external) |
| **E5** | Evaluator / task artifact | Agent's end state appears to satisfy the task but the evaluator rejects it; task spec or gold reference is broken or incomplete | Evaluator checks a property the instruction never asked for | lab `evaluator_mismatch`; CUADebug S7; Mis-score "evaluator false negative", "broken task" |

---

## 3. Confusable pairs — decision rules

| If unsure between | Choose | Rule |
|---|---|---|
| P1 vs P2 | P1 if the thing does not exist on screen at all; P2 if it exists and was misread |
| P3 vs P4 | P3 if the information was never mentioned; P4 if it was seen but its meaning/outcome was misjudged |
| P4 vs R5 | P4 if the false belief is about the *current screen / last action*; R5 if it is about *history or state not visible now* |
| any G vs P2 / R | G only if the CoT names the correct target and execution missed it; if the CoT names the wrong element → P2 (misread) or R (wrong choice). Never G |
| G3 vs R4 | G3 if the intended *effect* was right but the *mechanics* failed; R4 if the emitted action itself contradicts the stated intent (wrong text/args) |
| G3 vs E2 | E2 only when the agent could not have waited (no wait primitive / harness auto-fires); otherwise G3 |
| R1 vs R2 | R1 if the goal itself was wrong; R2 if the goal was right but the route cannot work |
| R2 vs R3 | R3 if the plan is wrong *because* the agent never considers a hidden affordance; R2 for other planning errors |
| R6 vs R7 | R6 for stopping too early / not verifying; R7 for continuing without adapting |
| R7 vs R8 | R7 when the repeated action is *failing*; R8 when actions *work* but are wasteful |
| R9 vs E5 | R9 when the task is designed infeasible; E5 when a feasible task is mis-scored |
| Anything vs E* | E* only when there is screenshot/log evidence the fault was outside the agent; the burden of proof is on E |

**Global order at the root step:** E → P → G → R, as above. Under this order a
perception error that causes a mis-grounded click is P (primary) + G (co-occurring);
a mis-grounded click that causes looping is G (primary) + R7 (downstream).

---

## 4. Using a gold trajectory (when given) — guidance only

Decided 2026-09-23. The gold trajectory (human text steps, or a successful run by
another agent) is **additional information, not a comparator**. The judge uses it to
understand what the task actually requires — which end state, which settings,
fields, options, or hidden affordances matter, what a working approach looks like —
and then locates the agent's error **from the agent's own trajectory**.

- The agent may take a completely different route. A different route is not an
  error, and the judge must not compare the trajectories step by step or report a
  divergence step. (An earlier draft had a `gold_divergence_step` flag; removed —
  with a weak judge it anchored the root step to wherever the routes first differed.)
- Gold is most useful as evidence that a *specific* agent action or belief was wrong
  for this task: e.g. the reference reveals that the setting lives in the Global
  column, that Enter must be pressed after typing, that a save is required before
  the evaluator runs.
- If the agent reaches the required end state by a different route and still fails,
  suspect E5 (evaluator artifact) or R6 (missing final verification/save).

## 5. Mapping tables

### CUADebug (31) → v2

| CUADebug | v2 | Note |
|---|---|---|
| P1 Visual hallucination | P1 | |
| P2 Misrecognition / OCR | P2 | |
| P3 Cross-modal misbinding | P2 | |
| P4 Observation omission | P3 | |
| P5 Semantic misunderstanding | P4 | |
| G1 Coordinate / element grounding | G1 | |
| G2 Visibility / accessibility | G2 | |
| G3 Interaction mechanics | G3 | |
| G4 Distraction | G4 | |
| R1 Constraint violation | R1 | |
| R2 Impossible plan / action | R2 | |
| R3 Decomposition failure | R2 (or R3 when a hidden affordance is the cause) | |
| R4 Inefficient / redundant | R8 | |
| R5 Action–intent misalignment | R4 | |
| R6 Invalid / malformed action | R4 | |
| R7 Parameter / argument error | R4 | |
| R8 Context loss | R5 | |
| R9 Memory hallucination | R5 | |
| R10 Progress misjudgment | R6 | |
| R11 Outcome misinterpretation | P4 | |
| R12 Failed self-correction | R7 | |
| R13 Causal misattribution | R5 | (rare; 0 in the 204) |
| S1 Rendering / layout | E1 | |
| S2 Timing / race | E2 | |
| S3 Unexpected system behavior | E1 or E4 | by cause |
| S4 Step / resource limit | E3 (or R8 if self-inflicted) | |
| S5 Tool / API failure | E3 | |
| S6 Environment instability | E1 | |
| S7 Benchmark / evaluation artifact | E5 | |
| IF1 / IF2 / O1 | R9 (IF2, O1); IF1 is not a failure | |

Family-level mapping is exact except CUADebug's R11 → P4 and R4/S4 boundary; the
family accuracy metric will be computed on both the original and the re-mapped
families so the paper can report agreement with CUADebug's own scheme.

### Lab frozen taxonomy (16) → v2

| Lab leaf | v2 |
|---|---|
| Click Region Error | G1 |
| Visual Confusion (Primitive Reliance) | P2 |
| Text Matching Bias | P2 (perceived label as target) / G1 (clicked label instead of field) |
| Resolution/Scale Brittleness | G1 |
| Fine-Grained Manipulation Failure | G1 / G3 |
| Software Commonsense (Icon Recognition) | P2 |
| Action Looping (Repetition) | R7 |
| Location Hallucination | G1 |
| Spatial Reasoning Error | G1 |
| Goal Hallucination | R1 |
| Reasoning Drift | P1 (perceptual) / R5 (non-perceptual) |
| Long-Horizon Memory Failure | R5 |
| Instruction Ambiguity Failure | R1 |
| Refusal/Infeasibility Error | R9 |
| Hidden Operation Blindness | R3 |
| Cross-Application Context Loss | R5 |
| meta `evaluator_mismatch` | E5 |
| meta `propagated_failure` | secondary relation = downstream |

---

## 6. What was deliberately left out

- **Inefficiency as a primary when the task was still completed** — a success is
  not a failure; efficiency is measured separately (OSWorld-Human WES).
- **Harness-side memory/compaction leaves** (Model-or-Harness, TRAIL) — the agents
  here run fixed-window screenshot histories; context loss shows up as R5.
- **Multi-agent / delegation leaves** — not applicable to single-agent CUA runs.
- **Separate "premature action" agent leaf** — it is G3 when the agent could have
  waited and E2 when it could not; a third leaf would only add confusion.

## 7. Open items before freezing v2

1. Pilot-label ~30 CUADebug trajectories with v2 (Raghav + Abdoul, independent) and
   measure per-leaf κ; merge any leaf pair with κ < 0.4 and few instances.
2. ~~Decide whether G1/G2 should collapse into one grounding leaf~~ — **done in v2.1**
   (merged into G1 Mis-grounded target; the near-miss vs wrong-element split needed an
   element-size estimate from screenshots, and CUADebug's G1 does not make the split).
3. Confirm the E2/G3 boundary on the OSWorld harness (`sleep_after_execution`
   settings differ across packages).
