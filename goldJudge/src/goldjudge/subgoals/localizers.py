"""Root-step localizers compared in the procedure study (docs/subgoals.md section 8).

All of them share one rendering of the failed run, with the screenshot convention fixed: OSWorld saves step_<t>.png
AFTER action t, so images are labelled "screen after step t" (``Step.screen_after``). They also share the label
definition from CUADebug (arXiv 2608.02643): the earliest step whose executed action introduces a new causal error that
leads to the failure, not one that only inherits an earlier mistake; placed on the step that acts on the wrong choice.

* ``judge``      plain judge: task + evaluator + every step's reasoning/action + evenly spaced screenshots (A)
* ``judge_g1``   the same + OSWorld-Human steps (A)
* ``checkpoint`` the same + gold subgoals as checkpoints: earliest step that breaks one for good (B)
* ``spansel``    agent subgoal spans as candidates + the screen after each span: pick the span, then the step (C)
* ``docheck``    the result of every ``do`` span checked on screen: first wrong one, then the step (D)
* ``counterfactual`` candidate steps proposed by the other methods; for each: would fixing only this step (everything
                 before unchanged) have avoided the failure? The earliest sufficient candidate wins (F)
"""

from __future__ import annotations

from typing import Any

from goldjudge.actions import ActionLog
from goldjudge.judge.images import select_steps_for_images
from goldjudge.schema import TaskMeta, Trajectory
from goldjudge.subgoals.core import marked_image, parse_json, trunc, txt
from goldjudge.subgoals.textgoals import TextGoal

LOCALIZER_PROMPT_VERSION = "localizer-v0.1"

LABEL_DEFINITION = """The ROOT-CAUSE STEP is the earliest step whose executed action introduces a new error that leads \
to the failure. Not a step that merely inherits an earlier mistake, and not a slip the agent later repaired. Put it on \
the step where the agent ACTS on the wrong choice (clicks the wrong element, types the wrong value, picks the wrong \
tool or approach, declares DONE too early), not the earlier step where it merely looked, and not later retries or \
loops. It can be step 1, or the final step (e.g. a premature DONE). Steps are 1-based."""

_COMMON_OUT = """Reply with JSON only: {"root_step": <int>, "evidence": "<what went wrong at that step>"}"""

JUDGE_SYSTEM = f"""You find where a FAILED computer-use agent run on an Ubuntu desktop went wrong.

{LABEL_DEFINITION}

You get the task, what the evaluator checks (spec and source), every step's reasoning and executed action, and some \
screenshots; each screenshot is labelled with the step whose result it shows ("screen after step t"). Work backward from \
the evaluator's failed check to the step that introduced it.

{_COMMON_OUT}"""

JUDGE_G1_SYSTEM = JUDGE_SYSTEM.replace("Work backward", "You also get the steps a human wrote down for this task (one \
valid route; the agent may legitimately take another). Use them to see what doing it right involves. Work backward")

CHECKPOINT_SYSTEM = f"""You find where a FAILED computer-use agent run on an Ubuntu desktop went wrong.

{LABEL_DEFINITION}

You get the task, what the evaluator checks, every step's reasoning and executed action, screenshots labelled with the \
step whose result they show, and CHECKPOINTS: the stages a successful run goes through, derived from a human's run \
(a different route to the same stage counts). For each checkpoint decide where the agent reached it correctly, if ever. \
The root-cause step is the earliest step that breaks a checkpoint for good: the action that gets that stage wrong or \
skips it, where nothing later repairs it. A checkpoint the agent gets wrong and later fixes does not count.

Reply with JSON only: {{"checkpoints": [{{"id": "1", "reached_correctly_at": <int or null>}}], "root_step": <int>, \
"broken_checkpoint": "<id>", "evidence": "..."}}"""

SPANSEL_SYSTEM = f"""You find where a FAILED computer-use agent run on an Ubuntu desktop went wrong.

{LABEL_DEFINITION}

The run is already split into the agent's own subgoals (step spans, from its reasoning). You get the task, what the \
evaluator checks, every step's reasoning and executed action, and for each span the screen after its last step.
1. Name the evaluator check that failed, i.e. which part of the end state is wrong or missing.
2. Pick the span whose actions produced that wrong end state or committed the wrong choice behind it. If the needed \
stage was never done at all, pick the span where the agent finished or moved on without it.
3. Inside that span, give the step where the wrong choice is acted on.

Reply with JSON only: {{"failed_check": "...", "span": "<span id>", "root_step": <int>, "evidence": "..."}}"""

DOCHECK_SYSTEM = f"""You check, stage by stage, whether a FAILED computer-use agent run produced the right result.

{LABEL_DEFINITION}

You get the task, what the evaluator checks, every step's reasoning and executed action, and the agent's "do" stages \
(spans that change or produce something), each with the screen after its last step. For every stage judge its RESULT \
against the task and the evaluator, at the level of detail the evaluator checks (the right target, value, range, \
format, location): "right", "wrong" or "unclear". Do not trust the agent's claims. Then give the root-cause step: inside \
the first stage whose result is wrong and stays wrong, the step where the wrong choice is acted on. If every stage is \
right, find the stage that is missing and give the step where the agent finished or moved on without it.

Reply with JSON only: {{"stages": [{{"id": "<span id>", "result": "right|wrong|unclear", "why": "..."}}], \
"root_step": <int>, "evidence": "..."}}"""


def steps_text(traj: Trajectory, log: ActionLog, *, thought_chars: int = 500) -> str:
  lsteps = {s.step: s for s in log.steps}
  lines = []
  for s in traj.steps:
    ls = lsteps.get(s.step)
    lines.append(f"[step {s.step}] reasoning: {trunc(s.thought, thought_chars) or trunc(s.intent, 200) or '-'}\n"
                 f"  action: {trunc(ls.render(), 200) if ls else '?'}")
  return "\n".join(lines)


def after_image(traj: Trajectory, t: int, max_width: int) -> list[dict[str, Any]]:
  s = next((x for x in traj.steps if x.step == t), None)
  if s is None or not s.screen_after:
    return []
  return [txt(f"(screen after step {t})"), marked_image(s.screen_after, [], traj.screen, max_width)]


def _head(task: str, traj: Trajectory) -> str:
  return f"{task}\n\n# FAILED RUN: agent {traj.agent}, {traj.n_steps} steps, evaluator score {traj.result}"


def judge_content(task: str, traj: Trajectory, log: ActionLog, *, image_budget: int, max_width: int,
                  human: str | None = None, checkpoints: list[TextGoal] | None = None) -> list[dict[str, Any]]:
  parts = [txt(_head(task, traj))]
  if human:
    parts.append(txt("# STEPS A HUMAN WROTE DOWN FOR THIS TASK\n" + human))
  if checkpoints:
    parts.append(txt("# CHECKPOINTS\n" + "\n".join(f"{g.id}. {g.purpose}" for g in checkpoints)))
  parts.append(txt("# STEPS\n" + steps_text(traj, log)))
  for t in select_steps_for_images(traj.n_steps, image_budget):
    parts += after_image(traj, t, max_width)
  parts.append(txt("JSON only."))
  return parts


def spans_content(task: str, traj: Trajectory, log: ActionLog, spans: list[TextGoal], *, image_budget: int,
                  max_width: int, kinds: set[str] | None = None) -> list[dict[str, Any]]:
  use = [g for g in spans if kinds is None or g.kind in kinds]
  keep = {use[i - 1].id for i in select_steps_for_images(len(use), image_budget)}
  parts = [txt(_head(task, traj)), txt("# STEPS\n" + steps_text(traj, log)), txt("# THE AGENT'S SUBGOALS")]
  for g in use:
    parts.append(txt(f"span {g.id} [{g.kind}, steps {g.first_step}-{g.last_step}, ended: {g.ended}]: {g.purpose}"))
    if g.id in keep:
      parts += after_image(traj, g.last_step, max_width)
  parts.append(txt("JSON only."))
  return parts


def parse_root(text: str, n_steps: int) -> dict[str, Any]:
  obj = parse_json(text, "root_step") or {}
  try:
    step = int(obj.get("root_step"))
  except (TypeError, ValueError):
    return {"root_step": None, "parse_ok": False}
  return {**{k: v for k, v in obj.items() if k != "root_step"}, "root_step": max(1, min(step, n_steps)), "parse_ok": True}


COUNTERFACTUAL_SYSTEM = f"""You decide which of several candidate steps is the root cause of a FAILED computer-use agent run.

{LABEL_DEFINITION}

You get the task, what the evaluator checks, every step's reasoning and executed action, screenshots labelled with the \
step whose result they show, and CANDIDATE steps proposed by other analysts. For each candidate imagine that only that \
step's action is replaced by the right one, with everything before it unchanged and the agent continuing sensibly:
- "sufficient": would that single fix have avoided the failure? (false if an earlier mistake still dooms the run, or if \
a later independent mistake would still fail it)
- "inherited": is the mistake at this step only a consequence of an earlier one?
The root-cause step is the EARLIEST candidate that is sufficient and not inherited. If none is, give the step you think \
is the root cause.

Reply with JSON only: {{"candidates": [{{"step": <int>, "sufficient": true|false, "inherited": true|false, "why": "..."}}], \
"root_step": <int>, "evidence": "..."}}"""


def candidate_steps(preds: list[int], *, max_n: int = 4) -> list[int]:
  """Distinct candidate steps by support (predictions within 1 step pooled), most supported first, then sorted."""
  ps = sorted(int(p) for p in preds if p == p)
  groups: list[list[int]] = []
  for p in ps:
    if groups and p - groups[-1][-1] <= 1:
      groups[-1].append(p)
    else:
      groups.append([p])
  groups.sort(key=lambda g: (-len(g), g[0]))
  reps = [max(set(g), key=g.count) for g in groups[:max_n]]
  return sorted(reps)


def counterfactual_content(task: str, traj: Trajectory, log: ActionLog, candidates: list[int], *, image_budget: int,
                           max_width: int) -> list[dict[str, Any]]:
  parts = [txt(_head(task, traj)), txt("# STEPS\n" + steps_text(traj, log))]
  shown = set(select_steps_for_images(traj.n_steps, max(0, image_budget - 2 * len(candidates))))
  for c in candidates:  # each candidate with the screens around it
    shown |= {c - 1, c} & set(range(1, traj.n_steps + 1))
  for t in sorted(shown):
    parts += after_image(traj, t, max_width)
  parts.append(txt(f"# CANDIDATE STEPS\n{', '.join(map(str, candidates))}\nAssess each candidate. JSON only."))
  return parts
