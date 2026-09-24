"""On-demand screenshots: a tool the judge calls to see more steps than the seed bundle holds.

The seed bundle (``BundleConfig(on_demand=True)``) carries every step's text plus a few
screenshots; the judge requests others with ``view_screenshots``. The same tool is offered
in every gold condition (agent trajectory always; the gold run only when one is given) so
G0/G1/G2 differ only in the reference, not in the harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from goldjudge.judge.images import encode_screenshot, image_path_ok
from goldjudge.schema import Trajectory

TOOL_NAME = "view_screenshots"


@dataclass
class ScreenshotTool:
  agent: Trajectory
  gold: Trajectory | None = None
  budget: int = 24  # extra screenshots per trajectory judged
  per_call: int = 4
  max_width: int = 1280
  shown: dict[str, set[int]] = field(default_factory=dict)  # seed steps already in the bundle
  requests: list[dict[str, Any]] = field(default_factory=list)
  used: int = 0

  def __post_init__(self) -> None:
    self.shown = {k: set(v) for k, v in self.shown.items()}
    self.shown.setdefault("agent", set())
    if self.gold is not None:
      self.shown.setdefault("gold", set())

  @property
  def trajectories(self) -> list[str]:
    return ["agent", "gold"] if self.gold is not None else ["agent"]

  def _shown_text(self) -> str:
    return "; ".join(f"{w} steps {self.viewed(w)}" for w in self.trajectories)

  def definition(self) -> dict[str, Any]:
    which = "the failed agent trajectory ('agent')" + (" or the reference run ('gold')" if self.gold is not None else "")
    return {
      "name": TOOL_NAME,
      "description": (
        f"Return the screenshots taken BEFORE the given steps of {which}. Step numbers are the 1-based "
        f"indices shown in the trajectory text. At most {self.per_call} steps per call and {self.budget} "
        "screenshots in total; steps whose screenshot is already in the conversation are not re-sent "
        f"and do not count against the budget. Already in the conversation (do not request): {self._shown_text()}."
      ),
      "strict": True,
      "input_schema": {
        "type": "object",
        "properties": {
          "trajectory": {"type": "string", "enum": self.trajectories},
          "steps": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["trajectory", "steps"],
        "additionalProperties": False,
      },
    }

  def viewed(self, which: str = "agent") -> list[int]:
    return sorted(self.shown.get(which, set()))

  def __call__(self, name: str, inp: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    if name != TOOL_NAME:
      return [{"type": "text", "text": f"unknown tool {name!r}"}], True
    which = inp.get("trajectory")
    traj = self.agent if which == "agent" else self.gold if which == "gold" else None
    if traj is None:
      return [{"type": "text", "text": f"trajectory must be one of {self.trajectories}"}], True
    try:
      steps = [int(s) for s in inp.get("steps") or []]
    except (TypeError, ValueError):
      return [{"type": "text", "text": "steps must be a list of integers"}], True
    steps = list(dict.fromkeys(steps))  # dedupe, keep order
    rec: dict[str, Any] = {"trajectory": which, "requested": steps, "granted": [], "already_shown": [], "invalid": [], "over_budget": []}
    self.requests.append(rec)
    by_step = {s.step: s for s in traj.steps}
    notes: list[str] = []
    blocks: list[dict[str, Any]] = []
    if len(steps) > self.per_call:
      notes.append(f"Only the first {self.per_call} steps of a call are served.")
      rec["over_budget"] += steps[self.per_call :]
      steps = steps[: self.per_call]
    for n in steps:
      s = by_step.get(n)
      if s is None or not image_path_ok(s.screenshot_path):
        rec["invalid"].append(n)
        continue
      if n in self.shown[which]:
        rec["already_shown"].append(n)
        continue
      if self.used >= self.budget:
        rec["over_budget"].append(n)
        continue
      prefix = "g" if which == "gold" else "step "
      blocks.append({"type": "text", "text": f"(screenshot before {prefix}{n}, {which} trajectory)"})
      data = encode_screenshot(s.screenshot_path, max_width=self.max_width).split(",", 1)[1]
      blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}})
      self.shown[which].add(n)
      self.used += 1
      rec["granted"].append(n)
    if rec["invalid"]:
      notes.append(f"No screenshot exists for {which} steps {rec['invalid']} (valid range 1..{traj.n_steps}).")
    if rec["already_shown"]:
      notes.append(f"Steps {rec['already_shown']} are already shown earlier in the conversation "
                   f"(scroll back to them; do not request them again). Already shown: {self._shown_text()}.")
    if rec["over_budget"] and self.used >= self.budget:
      notes.append(f"Screenshot budget exhausted ({self.budget}); steps {rec['over_budget']} not served. Produce the verdict now.")
    notes.append(f"Budget left: {self.budget - self.used} screenshots.")
    blocks.append({"type": "text", "text": " ".join(notes)})
    return blocks, not rec["granted"] and bool(rec["invalid"])
