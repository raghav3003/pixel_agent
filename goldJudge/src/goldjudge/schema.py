"""Unified trajectory schema.

Every source (CUADebug archives, OSWorld-Verified packages, future web benchmarks)
is normalized into these models so the judge, the review UI, and the metrics code
never see source-specific formats.

Coordinate convention: ``coords`` on a step are always in **executed screen pixels**
(1920x1080 for OSWorld). Model-space coordinates, when the agent used a different
space (Claude computer-use 1280x720, Qwen 1000x1000), are kept in
``model_action.coordinate`` with ``Trajectory.model_coord_space`` describing them.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ActionType = Literal[
  "click", "double_click", "right_click", "middle_click", "drag", "scroll", "type",
  "key", "hotkey", "move", "wait", "done", "fail", "call_user", "other",
]


class ModelAction(BaseModel):
  """What the model *proposed* (before harness translation), if recoverable."""

  type: str = ""
  coordinate: list[float] | None = None
  text: str | None = None
  raw: dict[str, Any] | str | None = None


class Step(BaseModel):
  step: int = Field(description="1-based step index; matches the source's step_num")
  screenshot_path: str = Field(description="Observation *before* this step's action")
  action_raw: str = Field(default="", description="Executed code, usually pyautogui")
  action_type: ActionType = "other"
  coords: list[float] | None = Field(default=None, description="Executed pixel coords")
  model_action: ModelAction | None = None
  thought: str = Field(default="", description="Reasoning / CoT text at this step")
  intent: str = Field(default="", description="Short stated intent (Action: line, plan_code)")
  reflection: str = ""
  timestamp: str = ""
  reward: float | None = None
  done: bool = False
  extra: dict[str, Any] = Field(default_factory=dict)


class Annotation(BaseModel):
  """Human root-cause annotation (CUADebug schema, kept verbatim)."""

  root_error_step: int
  taxonomy_tag: str
  taxonomy_family: str = ""
  evidence: str = ""
  correction: str = ""
  confidence: str = ""
  source: str = "cuadebug"


class Trajectory(BaseModel):
  source: str = Field(description="cuadebug | osworld_verified | ...")
  package: str = Field(description="Archive / run package the trajectory came from")
  agent: str = Field(description="Canonical agent id, e.g. claude-sonnet-4-5, opencua-7b")
  task_id: str
  domain: str
  dir: str
  steps: list[Step]
  result: float | None = Field(default=None, description="OSWorld evaluator score")
  horizon: int | None = None
  screen: tuple[int, int] = (1920, 1080)
  model_coord_space: tuple[int, int] | None = None
  annotation: Annotation | None = None
  extra: dict[str, Any] = Field(default_factory=dict)

  @property
  def success(self) -> bool:
    return self.result is not None and self.result >= 1.0

  @property
  def n_steps(self) -> int:
    return len(self.steps)

  @property
  def key(self) -> str:
    return f"{self.source}/{self.package}/{self.domain}/{self.task_id}"


class HumanReference(BaseModel):
  """OSWorld-Human text steps for a task."""

  task_id: str
  domain: str
  single_action: list[str] = Field(default_factory=list)
  grouped_action: list[list[str]] = Field(default_factory=list)


class TaskMeta(BaseModel):
  task_id: str
  domain: str
  instruction: str
  evaluator: dict[str, Any] = Field(default_factory=dict)
  related_apps: list[str] = Field(default_factory=list)
  config: list[dict[str, Any]] = Field(default_factory=list)
  infeasible: bool = False
  human: HumanReference | None = None
  raw: dict[str, Any] = Field(default_factory=dict)
