"""The agent's own subgoals, read from its reasoning, against the task's required subgoals.

Two calls per failed trajectory (driven by ``scripts/agent_goals.py``):

1. extract  reasoning + actions only -> the goals the agent was pursuing, with step spans and what it claimed.
            Blind on purpose: it sees neither the task subgoals nor the screenshots, so the agent's goals are
            not rewritten to fit the list they are later compared to, and claims stay separate from outcomes.
2. align    task subgoals + tracker status + screenshots after each goal -> does each agent goal serve the
            task, and did it really complete?

``classify`` then derives (deterministically) what happened to each required subgoal, and the earliest step
where the agent's intent or its belief about progress departed from the task.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from goldjudge.actions import ActionLog
from goldjudge.judge.images import image_path_ok, select_steps_for_images
from goldjudge.schema import TaskMeta, Trajectory
from goldjudge.subgoals.core import marked_image, parse_json, render_subgoals, trunc, txt
from goldjudge.subgoals.schema import SubgoalSet, Track

INTENT_PROMPT_VERSION = "intent-v0.1"

Alignment = Literal["required", "supporting", "unnecessary", "contradicts"]
ALIGNMENTS = {"required", "supporting", "unnecessary", "contradicts"}

EXTRACT_INTENTS_SYSTEM = """You read the reasoning of a computer-use agent working on a task, and write down the \
subgoals the AGENT ITSELF was pursuing, in its own terms and in the order it pursued them.

Rules:
- Report what the agent was trying to do, even when that was a bad idea or unrelated to the task. Do not correct it \
and do not judge it. You do not see the screen: never say whether a goal really succeeded.
- A goal spans the steps spent on one purpose ("get to the Composition settings", "replace the value in cell B2"). \
Merge consecutive steps with the same purpose; start a new goal when the purpose changes. Typically 2-10 goals.
- Record a goal again if the agent returns to it after doing something else (retries are informative).
- "claimed_done": true if the agent's own text says this goal was achieved; "claimed_step" is the step where it says \
so. "abandoned": true if the agent's text gives it up or moves on without claiming it.
- "evidence": a short quote from the reasoning that shows the goal.
- Some agents log only a one-line intent per step: infer the goal from those lines and the actions, and set \
"inferred": true when the agent never stated the purpose itself.

Reply with JSON only:
{"goals": [{"id": "A1", "description": "...", "first_step": 1, "last_step": 3, "claimed_done": false, \
"claimed_step": null, "abandoned": false, "inferred": false, "evidence": "..."}]}"""

ALIGN_SYSTEM = """You compare what a computer-use agent was trying to do with what its task actually required.

You get the task, its required subgoals (with a tracker's result: where each was first seen on screen), the agent's \
own goals with the steps they span and what the agent claimed, and for each goal the screenshot taken just after its \
last step (the observation before the next step). The run failed the task's evaluator.

For each agent goal return:
- "alignment": "required" (directly works towards a listed required subgoal), "supporting" (a reasonable means to \
one: navigation, checking, recovering from a mistake), "unnecessary" (harmless but not needed for the task), or \
"contradicts" (works against the task or the instruction: wrong target, extra change not asked for, undoing progress).
- "aligned_to": the task subgoal ids it serves, [] if none.
- "completed": "yes", "no" or "unclear", judged ONLY from the screenshots, never from the agent's claim.
- "evidence": what on screen supports that, in one short sentence.

Reply with JSON only: {"goals": [{"id": "A1", "alignment": "...", "aligned_to": ["S1"], "completed": "yes|no|unclear", \
"evidence": "..."}]}"""


class AgentGoal(BaseModel):
  id: str
  description: str
  first_step: int
  last_step: int
  claimed_done: bool = False
  claimed_step: int | None = None
  abandoned: bool = False
  inferred: bool = False
  evidence: str = ""
  # from the align call
  alignment: Alignment | None = None
  aligned_to: list[str] = Field(default_factory=list)
  completed: Literal["yes", "no", "unclear"] | None = None
  completion_evidence: str = ""

  @property
  def false_claim(self) -> bool:
    return self.claimed_done and self.completed == "no"


def _int(v: Any) -> int | None:
  try:
    return int(v)
  except (TypeError, ValueError):
    return None


# --- 1. extract --------------------------------------------------------------------------------


def extract_content(meta: TaskMeta, traj: Trajectory, log: ActionLog, *, thought_chars: int = 900) -> list[dict[str, Any]]:
  lsteps = {s.step: s for s in log.steps}
  lines = []
  for s in traj.steps:
    ls = lsteps.get(s.step)
    text = trunc(s.thought, thought_chars) or trunc(s.intent, 200) or "(no reasoning logged)"
    lines.append(f"[step {s.step}] {text}\n  executed: {trunc(ls.render(), 200) if ls else '?'}")
  return [txt(f"Task the agent was given: {meta.instruction}\nAgent: {traj.agent}; {traj.n_steps} steps.\n\n"
              "# THE AGENT'S REASONING AND ACTIONS\n" + "\n".join(lines)), txt("List the agent's goals. JSON only.")]


def parse_goals(text: str, n_steps: int) -> list[AgentGoal]:
  obj = parse_json(text, "goals") or {}
  out = []
  for i, r in enumerate(obj.get("goals") or [], 1):
    if not isinstance(r, dict):
      continue
    first, last = _int(r.get("first_step")), _int(r.get("last_step"))
    if first is None:
      continue
    last = first if last is None else last
    first, last = max(1, min(first, n_steps)), max(1, min(last, n_steps))
    claimed = _int(r.get("claimed_step"))
    out.append(AgentGoal(id=str(r.get("id") or f"A{i}"), description=str(r.get("description") or ""),
                         first_step=first, last_step=max(first, last), claimed_done=bool(r.get("claimed_done")),
                         claimed_step=claimed if claimed is not None and 1 <= claimed <= n_steps else None,
                         abandoned=bool(r.get("abandoned")), inferred=bool(r.get("inferred")),
                         evidence=str(r.get("evidence") or "")[:400]))
  return out


# --- 2. align ----------------------------------------------------------------------------------


def goal_image_steps(goals: list[AgentGoal], traj: Trajectory, budget: int) -> dict[str, int]:
  """Goal id -> step whose screenshot shows the result of that goal (the one after its last step).

  The final step has no later screenshot, so its own is the closest. Over budget, evenly spaced goals keep theirs.
  """
  have = {s.step for s in traj.steps if image_path_ok(s.screenshot_path)}
  picks = {}
  for g in goals:
    cands = [t for t in (g.last_step + 1, g.last_step) if t in have]
    if cands:
      picks[g.id] = cands[0]
  ids = list(picks)
  keep = {ids[i - 1] for i in select_steps_for_images(len(ids), budget)}
  return {k: v for k, v in picks.items() if k in keep}


def align_content(meta: TaskMeta, sset: SubgoalSet, track: Track, goals: list[AgentGoal], traj: Trajectory,
                  *, image_budget: int, max_width: int) -> list[dict[str, Any]]:
  first = track.first_hold()
  status = [f"{s.id}: " + (f"first seen at screenshot {first[s.id]}" if s.id in first else "never seen")
            for s in sset.ordered(essential_only=True)]
  parts = [txt(f"Task: {meta.instruction}\n\n# REQUIRED SUBGOALS OF THE TASK\n"
               f"{render_subgoals(sset.ordered(essential_only=True), full=False)}\n\n# TRACKER\n" + "\n".join(status)
               + f"\n\n# THE AGENT'S GOALS ({traj.agent}, {traj.n_steps} steps; result {traj.result})")]
  by = {s.step: s for s in traj.steps}
  shots = goal_image_steps(goals, traj, image_budget)
  for g in goals:
    claim = f"claimed done at step {g.claimed_step}" if g.claimed_done else ("abandoned" if g.abandoned else "no claim")
    parts.append(txt(f"{g.id} (steps {g.first_step}-{g.last_step}; {claim}): {g.description}"))
    t = shots.get(g.id)
    if t is not None:
      parts.append(txt(f"(screenshot before step {t})"))
      parts.append(marked_image(by[t].screenshot_path, [], traj.screen, max_width))
  parts.append(txt("Assess every agent goal. JSON only."))
  return parts


def apply_alignment(goals: list[AgentGoal], text: str, valid_subgoals: set[str]) -> list[AgentGoal]:
  obj = parse_json(text, "goals") or {}
  by = {str(r.get("id")): r for r in obj.get("goals") or [] if isinstance(r, dict)}
  out = []
  for g in goals:
    g = g.model_copy(deep=True)
    r = by.get(g.id)
    if r:
      a = str(r.get("alignment") or "").lower()
      c = str(r.get("completed") or "").lower()
      g.alignment = a if a in ALIGNMENTS else None
      g.aligned_to = [s for s in map(str, r.get("aligned_to") or []) if s in valid_subgoals]
      g.completed = c if c in {"yes", "no", "unclear"} else None
      g.completion_evidence = str(r.get("evidence") or "")[:400]
    out.append(g)
  return out


# --- 3. deterministic summary ------------------------------------------------------------------


def classify(goals: list[AgentGoal], sset: SubgoalSet, track: Track) -> dict[str, Any]:
  """What happened to each required subgoal, and where intent or belief first departed from the task.

  Per required subgoal: ``reached`` (the tracker saw it), ``false_completion`` (not seen, but the agent claimed a
  goal serving it was done), ``attempted`` (not seen, the agent worked on it), ``omitted`` (not seen, no goal served it).
  ``divergence_step`` is the earliest of: the start of a goal that contradicts the task; a false claim of completion.
  It is None when intent and beliefs never departed (the failure, if any, is then in execution).
  """
  first = track.first_hold()
  required = sset.ordered(essential_only=True)
  outcome: dict[str, str] = {}
  for s in required:
    serving = [g for g in goals if s.id in g.aligned_to]
    if s.id in first:
      outcome[s.id] = "reached"
    elif any(g.claimed_done for g in serving):
      outcome[s.id] = "false_completion"
    elif serving:
      outcome[s.id] = "attempted"
    else:
      outcome[s.id] = "omitted"
  events: list[tuple[int, str, str]] = []
  for g in goals:
    if g.alignment == "contradicts":
      events.append((g.first_step, "contradicting_goal", g.id))
    if g.false_claim:
      events.append((g.claimed_step or g.last_step, "false_claim", g.id))
  events.sort()
  counts = {a: sum(g.alignment == a for g in goals) for a in sorted(ALIGNMENTS)}
  steps = {a: sum(g.last_step - g.first_step + 1 for g in goals if g.alignment == a) for a in sorted(ALIGNMENTS)}
  unreached = [s.id for s in required if outcome[s.id] != "reached"]
  primary = outcome[unreached[0]] if unreached else "all_reached"
  return {"outcomes": outcome, "first_unreached": unreached[0] if unreached else None, "primary": primary,
          "divergence_step": events[0][0] if events else None, "divergence_kind": events[0][1] if events else None,
          "divergence_goal": events[0][2] if events else None, "alignment_counts": counts, "alignment_steps": steps,
          "n_goals": len(goals), "n_false_claims": sum(g.false_claim for g in goals),
          "n_inferred": sum(g.inferred for g in goals), "n_retried": _retried(goals)}


def _retried(goals: list[AgentGoal]) -> int:
  """Goals pursued again after doing something else: same aligned subgoal in non-adjacent goals."""
  seen: dict[str, int] = {}
  n = 0
  for i, g in enumerate(goals):
    for s in g.aligned_to:
      if s in seen and seen[s] < i - 1:
        n += 1
      seen[s] = i
  return n


