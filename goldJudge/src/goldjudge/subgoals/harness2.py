"""Harnesses from the 2026-11-06 cross-domain survey (docs/research/2026-11-06-failure-analysis-harness-survey.md).

* ``adjudicate``  idea 2: the judge's candidate steps are each checked locally (steps c-2..c+2, screens before/after c)
                  and labelled ROOT / INHERITED / SYMPTOM / BENIGN; the earliest ROOT wins (decided in code).
* ``prefix``      idea 3: outcome-blind scan. Windows are judged in order, each seeing only the run up to its end and
                  never the outcome: the first step in the window that introduces a new error, with a confidence.
                  All windows are scored; the stop threshold is chosen by cross-validation when decoding.
* ``evtrace``     idea 4: from the evaluator, the end-state facts it checks (per task); the steps that touched them
                  (text); then, with those steps' screens, the earliest wrong, uninherited write, or an omission.
Position-prior fusion (idea 1) needs no calls and lives in ``scripts/harness2.py``.
All screenshots use the corrected timing: step_<t>.png is the screen AFTER action t.
"""

from __future__ import annotations

from typing import Any

from goldjudge.actions import ActionLog
from goldjudge.judge.images import select_steps_for_images
from goldjudge.schema import Trajectory
from goldjudge.subgoals.core import marked_image, op_points, parse_json, trunc, txt
from goldjudge.subgoals.localizers import LABEL_DEFINITION

HARNESS2_PROMPT_VERSION = "harness2-v0.1"


def _step(traj: Trajectory, t: int):
  return next((s for s in traj.steps if s.step == t), None)


def brief_log(traj: Trajectory, log: ActionLog, upto: int | None = None, *, chars: int = 110) -> str:
  """One line per step: a short slice of the reasoning and the executed action."""
  lsteps = {s.step: s for s in log.steps}
  lines = []
  for s in traj.steps:
    if upto is not None and s.step > upto:
      break
    ls = lsteps.get(s.step)
    lines.append(f"{s.step}: {trunc(s.thought or s.intent, chars) or '-'} || {trunc(ls.render(), 90) if ls else '?'}")
  return "\n".join(lines)


def full_steps(traj: Trajectory, log: ActionLog, steps: list[int], *, max_width: int, after_images: bool = True,
               thought_chars: int = 500) -> list[dict[str, Any]]:
  lsteps = {s.step: s for s in log.steps}
  parts = []
  for t in steps:
    s, ls = _step(traj, t), lsteps.get(t)
    if s is None:
      continue
    parts.append(txt(f"[step {t}] reasoning: {trunc(s.thought, thought_chars) or trunc(s.intent, 200) or '-'}\n"
                     f"  action: {trunc(ls.render(), 200) if ls else '?'}"))
    if after_images and s.screen_after:
      parts.append(txt(f"(screen after step {t})"))
      parts.append(marked_image(s.screen_after, [], traj.screen, max_width))
  return parts


# --- idea 2: local adjudication of candidates ----------------------------------------------------

ADJUDICATE_SYSTEM = f"""You examine ONE candidate step of a FAILED computer-use agent run on an Ubuntu desktop.

{LABEL_DEFINITION}

You get the task, what the evaluator checks, a one-line log of the whole run, and in full the candidate step and its \
neighbours with the screen before the candidate's action (red marks: where its pointer landed) and the screens after. \
Label the candidate:
- "ROOT": its action itself introduces a NEW error (wrong element, value, approach, skipped requirement, premature DONE) \
that leads to the failure and is never repaired later.
- "INHERITED": it is wrong, but only because of an earlier error it builds on (give that earlier step).
- "SYMPTOM": the failure becomes visible or gets worse here, but it was caused earlier (give the earlier step if you can).
- "BENIGN": not an error, or an error that is repaired later.

Reply with JSON only: {{"label": "ROOT|INHERITED|SYMPTOM|BENIGN", "earlier_cause_step": <int or null>, \
"repaired_later": true|false, "evidence": "..."}}"""


def adjudicate_content(task: str, traj: Trajectory, log: ActionLog, c: int, *, max_width: int) -> list[dict[str, Any]]:
  parts = [txt(f"{task}\n\n# FAILED RUN: agent {traj.agent}, {traj.n_steps} steps\n# ONE-LINE LOG OF THE WHOLE RUN\n"
               + brief_log(traj, log)), txt(f"# CANDIDATE: step {c}")]
  s = _step(traj, c)
  ls = next((x for x in log.steps if x.step == c), None)
  if s is not None and s.screen_before:
    parts.append(txt(f"(screen before step {c}; red marks = where step {c}'s pointer landed)"))
    parts.append(marked_image(s.screen_before, op_points(ls.ops) if ls else [], traj.screen, max_width))
  window = [t for t in range(c - 2, c + 3) if 1 <= t <= traj.n_steps]
  lsteps = {x.step: x for x in log.steps}
  for t in window:
    st = _step(traj, t)
    if st is None:
      continue
    parts.append(txt(f"[step {t}{' = CANDIDATE' if t == c else ''}] reasoning: {trunc(st.thought, 500) or '-'}\n"
                     f"  action: {trunc(lsteps[t].render(), 200) if t in lsteps else '?'}"))
    if t in (c, c + 2) and st.screen_after:
      parts.append(txt(f"(screen after step {t})"))
      parts.append(marked_image(st.screen_after, [], traj.screen, max_width))
  parts.append(txt(f"Label step {c}. JSON only."))
  return parts


def parse_adjudication(text: str) -> dict[str, Any]:
  obj = parse_json(text, "label") or {}
  lab = str(obj.get("label") or "").upper()
  try:
    earlier = int(obj["earlier_cause_step"]) if obj.get("earlier_cause_step") is not None else None
  except (TypeError, ValueError):
    earlier = None
  return {"label": lab if lab in {"ROOT", "INHERITED", "SYMPTOM", "BENIGN"} else None, "earlier_cause_step": earlier,
          "repaired_later": bool(obj.get("repaired_later")), "evidence": str(obj.get("evidence") or "")[:500]}


def decide_adjudication(candidates: list[int], labels: dict[int, dict[str, Any]], fallback: int | None) -> tuple[int | None, str]:
  """Earliest ROOT; else the earliest earlier-cause named by an INHERITED/SYMPTOM candidate; else the fallback."""
  roots = sorted(c for c in candidates if (labels.get(c) or {}).get("label") == "ROOT")
  if roots:
    return roots[0], "root"
  earlier = sorted(v["earlier_cause_step"] for c, v in labels.items()
                   if v.get("label") in {"INHERITED", "SYMPTOM"} and v.get("earlier_cause_step"))
  if earlier:
    return earlier[0], "earlier_cause"
  return fallback, "fallback"


# --- idea 3: outcome-blind prefix scan ----------------------------------------------------------

PREFIX_SYSTEM = """You review a computer-use agent's run on an Ubuntu desktop AS IT HAPPENS, window by window. You see \
the task, what will be checked at the end, a short log of the earlier steps, and the steps of the current window in \
full with the screen after each. You do NOT know how the run ends.

Decide whether any step in THIS window introduces a NEW error: an action that, if nothing later repaired it, would make \
the end check fail. Examples: acting on the wrong element or item, entering a wrong value, choosing an approach that \
cannot produce what is checked, changing something the task did not ask for, moving on or declaring DONE while a \
required part is missing. Exploring, scrolling, opening a menu to look, or a slip that changes nothing are NOT errors. \
Judge each action by what the agent could know at that moment.

Reply with JSON only: {"error_step": <int or null>, "confidence": <0-100>, "evidence": "..."}"""


def prefix_windows(n_steps: int, size: int) -> list[list[int]]:
  return [list(range(i, min(i + size, n_steps + 1))) for i in range(1, n_steps + 1, size)]


def prefix_content(task_spec: str, traj: Trajectory, log: ActionLog, window: list[int], *, max_width: int) -> list[dict[str, Any]]:
  a = window[0]
  parts = [txt(task_spec)]
  if a > 1:
    parts.append(txt("# EARLIER STEPS (short)\n" + brief_log(traj, log, upto=a - 1)))
    prev = _step(traj, a - 1)
    if prev is not None and prev.screen_after:
      parts.append(txt(f"(screen after step {a - 1}, i.e. before this window)"))
      parts.append(marked_image(prev.screen_after, [], traj.screen, max_width))
  elif _step(traj, 1) is not None and _step(traj, 1).screen_before:
    parts.append(txt("(screen before step 1)"))
    parts.append(marked_image(_step(traj, 1).screen_before, [], traj.screen, max_width))
  parts.append(txt(f"# CURRENT WINDOW: steps {window[0]}-{window[-1]}"))
  parts += full_steps(traj, log, window, max_width=max_width)
  parts.append(txt("First step in this window that introduces a new error, or null. JSON only."))
  return parts


def parse_prefix(text: str, window: list[int]) -> dict[str, Any]:
  obj = parse_json(text, "confidence") or parse_json(text, "error_step") or {}
  try:
    step = int(obj["error_step"]) if obj.get("error_step") is not None else None
  except (TypeError, ValueError):
    step = None
  try:
    conf = float(obj.get("confidence") or 0)
  except (TypeError, ValueError):
    conf = 0.0
  if step is not None and step not in window:
    step = None
  return {"error_step": step, "confidence": conf if step is not None else 0.0, "evidence": str(obj.get("evidence") or "")[:400]}


def decode_prefix(windows: list[dict[str, Any]], theta: float, fallback: int | None) -> int | None:
  """First window step whose confidence reaches theta (windows in order), else the fallback."""
  for w in windows:
    if w.get("error_step") is not None and w.get("confidence", 0) >= theta:
      return w["error_step"]
  return fallback


# --- idea 4: evaluator-anchored backward trace -----------------------------------------------

CHECKS_SYSTEM = """From an OSWorld task's evaluator (spec and metric source) list the end-state facts it checks. For each: \
the object it reads (a file and what in it, a cell range, a setting key, a URL or tab, a UI element) and the expected \
value or condition, in plain words. Typically 1-4 facts.

Reply with JSON only: {"checks": [{"id": "E1", "object": "...", "expected": "..."}]}"""

TOUCH_SYSTEM = """You read the action log of a computer-use agent run and list the steps whose action could have changed, \
created, opened for editing, or navigated to one of the CHECKED OBJECTS (e.g. typed into the target cell, applied a \
format to it, saved the checked file, changed the checked setting, opened the checked URL), plus the step where the \
agent declared DONE or stopped.

Reply with JSON only: {"touches": [{"step": <int>, "checks": ["E1"], "what": "..."}], "done_step": <int or null>}"""

TRACE_SYSTEM = f"""You trace a FAILED computer-use agent run backward from the evaluator's failed check.

{LABEL_DEFINITION}

You get the task, the evaluator's CHECKED FACTS, a one-line log of the whole run, and in full the steps that touched \
the checked objects, with the screen after each and the final screen.
1. Decide which checked fact is wrong at the end.
2. Find the earliest step that wrote that object wrongly or chose the wrong object/approach for it, where the mistake is \
not just inherited from an earlier wrong step ("wrong_write").
3. If no step ever produced the needed state (it was skipped), give the step where the agent moved on without it or \
declared DONE ("omission").

Reply with JSON only: {{"failed_check": "E?", "kind": "wrong_write|omission|unclear", "root_step": <int>, "evidence": "..."}}"""


def checks_content(task: str) -> list[dict[str, Any]]:
  return [txt(task), txt("List the checked facts. JSON only.")]


def render_checks(checks: list[dict[str, Any]]) -> str:
  return "\n".join(f"{c.get('id')}: {c.get('object')} -> expected {c.get('expected')}" for c in checks) or "(none found)"


def touch_content(task: str, checks: list[dict[str, Any]], traj: Trajectory, log: ActionLog) -> list[dict[str, Any]]:
  return [txt(f"{task}\n\n# CHECKED OBJECTS\n{render_checks(checks)}\n\n# ACTION LOG ({traj.n_steps} steps)\n"
              + brief_log(traj, log, chars=200)), txt("List the touching steps. JSON only.")]


def parse_touches(text: str, n_steps: int) -> tuple[list[int], int | None]:
  obj = parse_json(text, "touches") or {}
  steps = set()
  for r in obj.get("touches") or []:
    try:
      t = int(r.get("step"))
    except (TypeError, ValueError, AttributeError):
      continue
    if 1 <= t <= n_steps:
      steps.add(t)
  try:
    done = int(obj["done_step"]) if obj.get("done_step") is not None else None
  except (TypeError, ValueError):
    done = None
  return sorted(steps), done


def trace_content(task: str, checks: list[dict[str, Any]], traj: Trajectory, log: ActionLog, touches: list[int],
                  done: int | None, *, image_budget: int, max_width: int) -> list[dict[str, Any]]:
  show = touches
  if len(show) > image_budget:
    show = [touches[i - 1] for i in select_steps_for_images(len(touches), image_budget)]
  parts = [txt(f"{task}\n\n# CHECKED FACTS\n{render_checks(checks)}\n\n# FAILED RUN: agent {traj.agent}, {traj.n_steps} "
               f"steps; declared DONE/stopped at step {done}\n# ONE-LINE LOG\n" + brief_log(traj, log)),
           txt(f"# STEPS THAT TOUCHED THE CHECKED OBJECTS: {touches or 'none'}")]
  parts += full_steps(traj, log, show, max_width=max_width)
  last = _step(traj, traj.n_steps)
  if last is not None and last.screen_after and traj.n_steps not in show:
    parts.append(txt(f"(final screen, after step {traj.n_steps})"))
    parts.append(marked_image(last.screen_after, [], traj.screen, max_width))
  parts.append(txt("Trace back from the failed check. JSON only."))
  return parts
