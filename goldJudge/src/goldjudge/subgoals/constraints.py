"""Required states + method constraints per task, checked against a failed run (idea 4, docs/subgoals.md section 10).

1. extract  task + evaluator + OSWorld-Human steps + captioned successful AI runs + candidate states (essential states
            from the evaluator-anchored pipeline, sections 2-5) -> each state marked required or not, plus method
            constraints: approaches the evaluator forces or rules out ("the change must apply to every slide").
            Every constraint must name the evaluator check that fails when it is broken.
2. verify   each successful run is checked against the constraints; a constraint a success breaks is dropped.
3. check    failed run's reasoning + actions -> the earliest step that commits to an approach breaking a constraint, or
            skips / gets wrong a required state, and is not repaired later.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from goldjudge.actions import ActionLog
from goldjudge.schema import Trajectory
from goldjudge.subgoals.core import parse_json, trunc, txt
from goldjudge.subgoals.localizers import LABEL_DEFINITION, steps_text

CONSTRAINT_PROMPT_VERSION = "constraints-v0.1"

EXTRACT_SYSTEM = """You write the reference for checking computer-use agent runs on one task (Ubuntu desktop).

You get the task, what the evaluator checks (spec and source code), steps a human wrote down, successful AI runs (as \
action logs with what each click hit and what changed), and CANDIDATE STATES proposed earlier.

1. For each candidate state decide "required": true only if every successful route must reach it (the evaluator, or a \
precondition of what it checks, needs it). Successful runs that skip it prove it is not required.
2. Write METHOD CONSTRAINTS: choices of approach that decide success, that a run can get wrong even while every state \
looks reached. Examples of the kind: "the colour must be applied to the text, not the box fill"; "the change must apply \
to all slides (master slide, or all slides selected), not only the current one"; "the value must be typed into the \
cell, not as a comment"; "do not change other cells". Each constraint needs "kind": "must" or "must_not", and \
"evaluator_reason": the evaluator check that fails when it is broken. Write only constraints the evaluator really \
enforces; do not require the human's particular menu path. Typically 1-5.

Reply with JSON only: {"states": [{"id": "<candidate id>", "required": true|false, "why": "..."}], \
"constraints": [{"id": "C1", "kind": "must|must_not", "text": "...", "evaluator_reason": "..."}]}"""

VERIFY_SYSTEM = """You check a SUCCESSFUL computer-use run (it passed the evaluator) against method constraints that were \
written for its task. For each constraint answer "broken": true only if the run clearly breaks it. A successful run that \
breaks a constraint shows the constraint is wrong.

Reply with JSON only: {"constraints": [{"id": "C1", "broken": true|false, "why": "..."}]}"""

CHECK_SYSTEM = f"""You find where a FAILED computer-use agent run on an Ubuntu desktop went wrong, using a reference of \
what the task requires.

{LABEL_DEFINITION}

You get the task, what the evaluator checks, the REQUIRED STATES and METHOD CONSTRAINTS for the task, and every step's \
reasoning and executed action. Go through the run from step 1 and find the earliest step that (a) commits to an approach \
that breaks a method constraint, or (b) gets a required state wrong or moves on without it, where nothing later repairs \
it. Put it on the step whose action carries out the wrong choice. If the reference does not explain the failure, give \
the step you think is the root cause anyway and say so.

Reply with JSON only: {{"violation": "<constraint id, state id, or none>", "root_step": <int>, "evidence": "..."}}"""


class Constraint(BaseModel):
  id: str
  kind: str = "must"
  text: str
  evaluator_reason: str = ""
  broken_by: list[str] = Field(default_factory=list, description="successful runs that break it")


class StateMark(BaseModel):
  id: str
  description: str
  required: bool = True
  why: str = ""


def extract_content(task: str, human: str | None, golds: list[ActionLog], states: list[tuple[str, str]]) -> list[dict[str, Any]]:
  parts = [txt(task)]
  if human:
    parts.append(txt("# STEPS A HUMAN WROTE DOWN\n" + human))
  for g in golds:
    parts.append(txt(f"# SUCCESSFUL RUN ({g.agent}, {len(g.steps)} steps)\n" + trunc(g.render(prefix="g"), 6000)))
  parts.append(txt("# CANDIDATE STATES\n" + ("\n".join(f"{i}: {d}" for i, d in states) or "(none)")))
  parts.append(txt("Mark the states and write the constraints. JSON only."))
  return parts


def parse_extract(text: str, states: list[tuple[str, str]]) -> tuple[list[StateMark], list[Constraint]] | None:
  obj = parse_json(text, "constraints")
  if obj is None:
    return None
  marks = {str(r.get("id")): r for r in obj.get("states") or [] if isinstance(r, dict)}
  sm = [StateMark(id=i, description=d, required=bool(marks.get(i, {}).get("required", True)),
                  why=str(marks.get(i, {}).get("why") or "")[:300]) for i, d in states]
  cs = []
  for k, r in enumerate(obj.get("constraints") or [], 1):
    if isinstance(r, dict) and r.get("text"):
      kind = str(r.get("kind") or "must").lower()
      cs.append(Constraint(id=str(r.get("id") or f"C{k}"), kind=kind if kind in {"must", "must_not"} else "must",
                           text=str(r["text"])[:400], evaluator_reason=str(r.get("evaluator_reason") or "")[:400]))
  return sm, cs


def render_reference(states: list[StateMark], cons: list[Constraint]) -> str:
  req = [s for s in states if s.required]
  lines = ["# REQUIRED STATES"] + ([f"{s.id}: {s.description}" for s in req] or ["(none)"])
  lines += ["# METHOD CONSTRAINTS"] + ([f"{c.id} [{c.kind}]: {c.text} (evaluator: {c.evaluator_reason})" for c in cons] or ["(none)"])
  return "\n".join(lines)


def verify_content(task: str, gold: ActionLog, cons: list[Constraint]) -> list[dict[str, Any]]:
  return [txt(task), txt("# CONSTRAINTS\n" + "\n".join(f"{c.id} [{c.kind}]: {c.text}" for c in cons)),
          txt(f"# SUCCESSFUL RUN ({gold.agent})\n" + trunc(gold.render(prefix="g"), 8000)), txt("Check every constraint. JSON only.")]


def parse_verify(text: str) -> dict[str, bool]:
  obj = parse_json(text, "constraints") or {}
  return {str(r.get("id")): bool(r.get("broken")) for r in obj.get("constraints") or [] if isinstance(r, dict)}


def check_content(task: str, reference: str, traj: Trajectory, log: ActionLog) -> list[dict[str, Any]]:
  return [txt(f"{task}\n\n{reference}\n\n# FAILED RUN: agent {traj.agent}, {traj.n_steps} steps"),
          txt("# STEPS\n" + steps_text(traj, log)), txt("JSON only.")]
