"""Step-level intent vs outcome (idea 1 of the procedure study, docs/subgoals.md section 10).

For every step: the intent is read from THAT step's reasoning only (prospective, so no hindsight from later steps), and
the outcome from the screens around it, with the screenshot timing fixed: the screen before action t is step_<t-1>.png
(``Step.screen_before``; red marks show where action t's pointer landed), the screen after it is step_<t>.png. A step
whose outcome does not match its intent is a candidate root (a wrong click, wrong value, action with no effect).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from goldjudge.actions import ActionLog
from goldjudge.schema import Trajectory
from goldjudge.subgoals.core import marked_image, op_points, parse_json, trunc, txt

STEPCHECK_PROMPT_VERSION = "stepcheck-v0.1"

STEPCHECK_SYSTEM = """You check, step by step, whether a computer-use agent's actions did what the agent meant them to do.

For each step you get the agent's reasoning AT THAT STEP (what it intended), the executed action, the screen before the \
action (red circles: where the pointer landed) and the screen after it.

For every step return:
- "intent": what the agent meant this single action to achieve, in a few words, from its reasoning at that step only.
- "outcome": what visibly happened, in a few words.
- "match": "yes" if the outcome is what the agent intended; "no" if it is not (the click landed on a different element, \
a different value was entered, the wrong item was selected, nothing happened when something should have, an error \
appeared); "unclear" if the screens cannot show it (e.g. a keyboard shortcut with no visible effect yet).
- "harm": true if the outcome moves the task backwards or into a wrong state (wrong item chosen, data lost, wrong \
setting applied), not merely a wasted step.
Judge from the screens, not from the agent's later claims.

Reply with JSON only: {"steps": [{"step": <int>, "intent": "...", "outcome": "...", "match": "yes|no|unclear", "harm": true|false}]}"""


class StepVerdict(BaseModel):
  step: int
  intent: str = ""
  outcome: str = ""
  match: Literal["yes", "no", "unclear"] = "unclear"
  harm: bool = False


def windows(n_steps: int, size: int) -> list[list[int]]:
  return [list(range(i, min(i + size, n_steps + 1))) for i in range(1, n_steps + 1, size)]


def window_content(instruction: str, traj: Trajectory, log: ActionLog, steps: list[int], *,
                   max_width: int, thought_chars: int = 500) -> list[dict[str, Any]]:
  by, lsteps = {s.step: s for s in traj.steps}, {s.step: s for s in log.steps}
  parts = [txt(f"Task the agent was given: {instruction}\nSteps {steps[0]}-{steps[-1]} of {traj.n_steps}.")]
  for t in steps:
    s, ls = by.get(t), lsteps.get(t)
    if s is None:
      continue
    if s.screen_before:
      parts.append(txt(f"(screen before step {t}; red marks = where step {t}'s pointer landed)"))
      parts.append(marked_image(s.screen_before, op_points(ls.ops) if ls else [], traj.screen, max_width))
    else:
      parts.append(txt(f"(no screen before step {t} was logged)"))
    parts.append(txt(f"step {t} reasoning: {trunc(s.thought, thought_chars) or trunc(s.intent, 200) or '-'}\n"
                     f"step {t} action: {trunc(ls.render(), 200) if ls else '?'}"))
  last = by.get(steps[-1])
  if last is not None and last.screen_after:
    parts.append(txt(f"(screen after step {steps[-1]})"))
    parts.append(marked_image(last.screen_after, [], traj.screen, max_width))
  parts.append(txt(f"Return one entry per step {steps[0]}-{steps[-1]}. The screen after step t is the screen before step t+1. JSON only."))
  return parts


def parse_window(text: str, steps: list[int]) -> list[StepVerdict]:
  obj = parse_json(text, "steps") or {}
  out = {}
  for r in obj.get("steps") or []:
    if not isinstance(r, dict):
      continue
    try:
      t = int(r.get("step"))
    except (TypeError, ValueError):
      continue
    if t in steps:
      m = str(r.get("match") or "").lower()
      out[t] = StepVerdict(step=t, intent=str(r.get("intent") or "")[:200], outcome=str(r.get("outcome") or "")[:200],
                           match=m if m in {"yes", "no", "unclear"} else "unclear", harm=bool(r.get("harm")))
  return [out[t] for t in steps if t in out]


def render_flags(verdicts: list[StepVerdict], *, max_lines: int = 25) -> str:
  """Context for the judge: the steps whose outcome did not match the intent (harmful ones marked)."""
  bad = [v for v in verdicts if v.match == "no"]
  if not bad:
    return "No step was flagged: every checked action did what the agent intended (or could not be judged)."
  lines = [f"step {v.step}{' [HARMFUL]' if v.harm else ''}: meant to {v.intent}; actually {v.outcome}" for v in bad[:max_lines]]
  more = f"\n(+{len(bad) - max_lines} more flagged steps)" if len(bad) > max_lines else ""
  return "\n".join(lines) + more
