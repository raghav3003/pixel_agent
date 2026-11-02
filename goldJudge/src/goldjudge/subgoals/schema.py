"""Subgoal records: per-task subgoal sets, per-step captions, per-step subgoal status, localizations."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Necessity = Literal["essential", "conditional", "optional"]


class Subgoal(BaseModel):
  """A milestone stated as an observable *state* of the machine, not as an action.

  ``essential``: every successful route must pass through this state (the evaluator, or a
  precondition of what it checks, requires it). ``conditional``: required only on some routes or
  for some initial states. ``optional``: helpful but skippable. Only essential subgoals are used to
  localize errors; the rest are kept for analysis. Necessity applies to the state: every route must
  pass through at least one of ``observable_forms``, even if no single form is required.
  """

  id: str
  description: str
  observable_forms: list[str] = Field(default_factory=list, description="how the state looks on each route; any one counts")
  success_check: str = Field(default="", description="what must be visible on screen for this to hold")
  necessity: Necessity = "essential"
  necessity_reason: str = ""
  depends_on: list[str] = Field(default_factory=list, description="subgoals that must hold before this one")
  persistent: bool = Field(default=False, description="must still hold at the end (undoing it breaks the task)")
  alternatives: str = Field(default="", description="other ways to reach the same state")
  # consolidation / verification (filled by the pipeline)
  support: list[str] = Field(default_factory=list, description="extraction views that proposed it")
  support_frac: float = 0.0
  necessity_votes: dict[str, int] = Field(default_factory=dict)
  verification: Literal["verified", "refuted", "unverified", "unclear"] = "unverified"
  gold_hits: dict[str, bool | None] = Field(default_factory=dict, description="gold key -> held at some step")


class SubgoalSet(BaseModel):
  task_id: str
  domain: str
  instruction: str
  version: str
  views: list[str] = Field(default_factory=list)
  subgoals: list[Subgoal]
  dropped_edges: list[list[str]] = Field(default_factory=list, description="order edges contradicted by a gold")
  notes: str = ""
  extra: dict[str, Any] = Field(default_factory=dict)

  def ordered(self, *, essential_only: bool = True, use_verification: bool = True) -> list[Subgoal]:
    """Topological order (ties by id order) over the kept subgoals."""
    keep = [s for s in self.subgoals
            if (not essential_only or s.necessity == "essential")
            and not (use_verification and s.verification == "refuted")]
    ids = {s.id for s in keep}
    pos = {s.id: i for i, s in enumerate(self.subgoals)}
    out, done = [], set()
    pending = sorted(keep, key=lambda s: pos[s.id])
    while pending:
      for s in pending:
        if all(d in done or d not in ids for d in s.depends_on):
          out.append(s)
          done.add(s.id)
          pending.remove(s)
          break
      else:  # cycle: fall back to listed order
        out.extend(pending)
        break
    return out


class StepCaption(BaseModel):
  step: int
  target: str = ""
  effect: str = ""
  hit_intended: bool | None = None


class StepStatus(BaseModel):
  """Subgoals judged to hold in the observation *before* ``step`` (i.e. after step-1's action)."""

  step: int
  holds: list[str] = Field(default_factory=list)
  unsure: list[str] = Field(default_factory=list)
  note: str = ""


class Track(BaseModel):
  key: str
  task_id: str
  agent: str
  result: float | None
  n_steps: int
  version: str
  statuses: list[StepStatus]
  usage: dict[str, Any] = Field(default_factory=dict)

  def first_hold(self) -> dict[str, int]:
    out: dict[str, int] = {}
    for st in self.statuses:
      for sid in st.holds:
        out.setdefault(sid, st.step)
    return out
