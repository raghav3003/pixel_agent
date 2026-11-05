"""Optional analysis aids added to the judge's input (idea 6, docs/subgoals.md section 10).

Each aid is a text section built from an earlier pipeline's saved output; a missing output yields a short
"(not available)" note so every trajectory gets the same layout.

* ``subgoals``     the agent's own subgoals from its reasoning (text-subgoal pipeline), as a table of contents
* ``stepflags``    steps whose outcome did not match the agent's intent (step check)
* ``constraints``  required states and method constraints for the task, constraints broken by a success removed
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCES = {
  "subgoals": ROOT / "data/textgoals/gpt6luna-text-v0.3/agent",
  "stepflags": ROOT / "data/stepcheck/gpt6luna-step-v0.1",
  "constraints": ROOT / "data/constraints/gpt6luna-cons-v0.1",
}
CONTEXT_NOTE = ("# ANALYSIS AIDS (automatic pre-analysis of this run; useful but may contain errors - verify against "
                "the steps and screenshots before relying on them)")


def _subgoals(agent: str, task_id: str) -> str | None:
  p = SOURCES["subgoals"] / agent / f"{task_id}.r1.json"
  if not p.exists():
    return None
  sgs = json.loads(p.read_text(encoding="utf-8"))["subgoals"]
  lines = [f"- steps {g['first_step']}-{g['last_step']} [{g['kind']}, {g['ended']}]: {g['purpose']}" for g in sgs]
  return "## The agent's own subgoals, read from its reasoning\n" + "\n".join(lines)


def _stepflags(agent: str, task_id: str) -> str | None:
  from goldjudge.subgoals.stepcheck import StepVerdict, render_flags

  p = SOURCES["stepflags"] / agent / f"{task_id}.json"
  if not p.exists():
    return None
  v = [StepVerdict(**x) for x in json.loads(p.read_text(encoding="utf-8"))["verdicts"]]
  return "## Steps whose visible outcome did not match the agent's intent\n" + render_flags(v)


def _constraints(agent: str, task_id: str) -> str | None:
  from goldjudge.subgoals.constraints import Constraint, StateMark, render_reference

  base = SOURCES["constraints"]
  p = base / "tasks" / f"{task_id}.json"
  if not p.exists():
    return None
  d = json.loads(p.read_text(encoding="utf-8"))
  broken = set()
  vdir = base / "verify" / task_id
  for v in vdir.glob("*.json") if vdir.exists() else []:
    broken |= {k for k, b in json.loads(v.read_text(encoding="utf-8"))["broken"].items() if b}
  cons = [Constraint(**c) for c in d["constraints"] if c["id"] not in broken]
  return "## What the task requires\n" + render_reference([StateMark(**s) for s in d["states"]], cons)


BUILDERS = {"subgoals": _subgoals, "stepflags": _stepflags, "constraints": _constraints}


def context_sections(names: list[str], agent: str, task_id: str) -> tuple[str, dict[str, bool]]:
  """The analysis-aid text for one trajectory and which aids were available."""
  have, parts = {}, [CONTEXT_NOTE]
  for n in names:
    s = BUILDERS[n](agent, task_id)
    have[n] = s is not None
    parts.append(s if s is not None else f"## {n}: (not available for this run)")
  return "\n\n".join(parts), have
