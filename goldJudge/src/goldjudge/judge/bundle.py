"""Build the evidence bundle the judge sees for one failed trajectory.

The bundle is condition-aware:

* ``G0`` — no reference
* ``G1`` — OSWorld-Human text steps (grouped-action form)
* ``G2`` — a successful AI trajectory for the same task (actions + screenshots)
* ``G3`` — a human-note-guided run with screenshots (same format as G2, different source)

Everything is rendered into OpenAI-style multimodal chat ``content`` parts so any
OpenAI-compatible endpoint (vLLM, OpenAI, OpenRouter) can consume it.

With ``on_demand=True`` the image budgets are only the *seed*: the bundle says which
steps carry a screenshot and the judge fetches others via ``judge.interactive``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from goldjudge.judge.images import encode_screenshot, image_path_ok, select_steps_for_images
from goldjudge.loaders.osworld_meta import OSWorldMeta
from goldjudge.schema import HumanReference, TaskMeta, Trajectory

Condition = Literal["G0", "G1", "G2", "G3"]


@dataclass
class BundleConfig:
  condition: Condition = "G0"
  agent_image_budget: int = 16
  gold_image_budget: int = 8
  image_max_width: int = 1280
  thought_chars: int = 700
  action_chars: int = 200
  eval_source_chars: int = 1500
  eval_log_chars: int = 800
  include_eval_source: bool = True
  include_eval_log: bool = True
  on_demand: bool = False  # seed-only screenshots; judge requests more via view_screenshots
  extra_must_steps: list[int] = field(default_factory=list)  # e.g. steps a first pass flagged
  # "legacy": step_<t>.png labelled "before step t" (how every run up to 2026-11-05 was built). "after": the same image
  # labelled as the screen after step t and placed after that step's text, which is what OSWorld actually saved.
  screen_labels: str = "legacy"
  show_reflection: bool = False  # Gemini logs a per-step reflection that older runs never showed


@dataclass
class Bundle:
  content: list[dict[str, Any]]  # user-message content parts
  n_images: int
  meta: dict[str, Any]
  agent_image_steps: list[int] = field(default_factory=list)
  gold_image_steps: list[int] = field(default_factory=list)


def _txt(s: str) -> dict[str, Any]:
  return {"type": "text", "text": s}


def _img(path: str, max_width: int) -> dict[str, Any]:
  return {"type": "image_url", "image_url": {"url": encode_screenshot(path, max_width=max_width)}}


def _trunc(s: str, n: int) -> str:
  s = (s or "").strip()
  return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def render_task(meta: TaskMeta, osw: OSWorldMeta, cfg: BundleConfig, *, eval_log: str | None) -> str:
  lines = [
    "# TASK",
    f"Domain: {meta.domain}",
    f"Instruction: {meta.instruction}",
    "",
    "## What the OSWorld evaluator checks",
    f"evaluator spec: {meta.evaluator}",
  ]
  if cfg.include_eval_source:
    for fn in osw.evaluator_funcs(meta):
      src = osw.metric_source(fn, (meta.raw or {}).get("_commit"))
      if src:
        lines += [f"", f"### source of metric `{fn}` (truncated)", "```python", _trunc(src, cfg.eval_source_chars), "```"]
  if cfg.include_eval_log and eval_log:
    lines += ["", "## Evaluator log for this run", "```", _trunc(eval_log, cfg.eval_log_chars), "```"]
  return "\n".join(lines)


def render_human_text(human: HumanReference) -> str:
  steps = human.grouped_action or [[a] for a in human.single_action]
  out = ["# ADDITIONAL INFORMATION — human-written steps for this task (guidance only; the agent may use a different valid route; do not compare step by step)"]
  for i, group in enumerate(steps, 1):
    out.append(f"h{i}: " + " ; ".join(group))
  return "\n".join(out)


def _on_demand_note(included: list[int], n_steps: int, which: str) -> str:
  return (
    f"(Screenshots included below for {which} steps {included}; for any other of the {n_steps} steps, "
    f"call view_screenshots with trajectory='{which}'.)"
  )


def gold_trajectory_parts(gold: Trajectory, cfg: BundleConfig) -> tuple[list[dict[str, Any]], list[int]]:
  parts: list[dict[str, Any]] = [
    _txt(
      "# ADDITIONAL INFORMATION — one successful run of this task by another agent (guidance only; "
      "the agent under review may use a different valid route; do not compare step by step). "
      f"{gold.n_steps} steps, evaluator score {gold.result}. "
      + ("Each screenshot shows the screen AFTER the listed action." if cfg.screen_labels == "after"
         else "Screenshots are the observation BEFORE the listed action.")
    )
  ]
  img_steps = set(select_steps_for_images(gold.n_steps, cfg.gold_image_budget))
  included = [s.step for s in gold.steps if s.step in img_steps and image_path_ok(s.screenshot_path)]
  if cfg.on_demand:
    parts.append(_txt(_on_demand_note(included, gold.n_steps, "gold")))
  for s in gold.steps:
    line = f"g{s.step}: [{s.action_type}] {_trunc(s.action_raw, cfg.action_chars)}"
    if s.intent and s.intent not in {"DONE", "WAIT", "FAIL"}:
      line += f"  // intent: {_trunc(s.intent, 160)}"
    after = cfg.screen_labels == "after"
    if s.step in included and not after:
      parts.append(_txt(f"(screenshot before g{s.step})"))
      parts.append(_img(s.screenshot_path, cfg.image_max_width))
    parts.append(_txt(line))
    if s.step in included and after:
      parts.append(_txt(f"(screen after g{s.step})"))
      parts.append(_img(s.screenshot_path, cfg.image_max_width))
  return parts, included


def agent_trajectory_parts(traj: Trajectory, cfg: BundleConfig) -> tuple[list[dict[str, Any]], list[int]]:
  parts: list[dict[str, Any]] = [
    _txt(
      f"# AGENT TRAJECTORY (FAILED; evaluator score {traj.result}). Agent: {traj.agent}; "
      f"{traj.n_steps} steps; step horizon {traj.horizon}.\n"
      + ("For each step: the screenshot shows the screen AFTER that step's action; 'thought' is the agent's "
         if cfg.screen_labels == "after" else "For each step: the screenshot is the observation BEFORE the action; 'thought' is the agent's ")
      + "reasoning; 'executed' is what actually ran on the machine (pixels, 1920x1080)."
    )
  ]
  img_steps = set(select_steps_for_images(traj.n_steps, cfg.agent_image_budget, must=cfg.extra_must_steps))
  included = [s.step for s in traj.steps if s.step in img_steps and image_path_ok(s.screenshot_path)]
  if cfg.on_demand:
    parts.append(_txt(_on_demand_note(included, traj.n_steps, "agent")))
  after = cfg.screen_labels == "after"
  if after and traj.steps and traj.steps[0].screen_before and image_path_ok(traj.steps[0].screen_before):
    parts.append(_txt("(initial screen, before step 1)"))  # Qwen runs log step_0
    parts.append(_img(traj.steps[0].screen_before, cfg.image_max_width))
  for s in traj.steps:
    if s.step in included and not after:
      parts.append(_txt(f"(screenshot before step {s.step})"))
      parts.append(_img(s.screenshot_path, cfg.image_max_width))
    block = [f"## step {s.step}"]
    if s.thought:
      block.append(f"thought: {_trunc(s.thought, cfg.thought_chars)}")
    if s.intent and s.intent != s.thought:
      block.append(f"intent: {_trunc(s.intent, 200)}")
    if cfg.show_reflection and s.reflection:
      block.append(f"reflection: {_trunc(s.reflection, cfg.thought_chars)}")
    ma = s.model_action
    if ma and ma.coordinate:
      block.append(f"model action: {ma.type} at {ma.coordinate} (model coordinate space {traj.model_coord_space})")
    block.append(f"executed: [{s.action_type}] {_trunc(s.action_raw, cfg.action_chars)}")
    parts.append(_txt("\n".join(block)))
    if s.step in included and after:
      parts.append(_txt(f"(screen after step {s.step})"))
      parts.append(_img(s.screenshot_path, cfg.image_max_width))
  return parts, included


def build_bundle(
  traj: Trajectory,
  meta: TaskMeta,
  osw: OSWorldMeta,
  cfg: BundleConfig,
  *,
  gold: Trajectory | None = None,
  eval_log: str | None = None,
) -> Bundle:
  parts: list[dict[str, Any]] = [_txt(render_task(meta, osw, cfg, eval_log=eval_log))]
  gold_imgs: list[int] = []
  if cfg.condition == "G1":
    if meta.human is None:
      raise ValueError(f"G1 requested but no human reference for {meta.task_id}")
    parts.append(_txt(render_human_text(meta.human)))
  elif cfg.condition in {"G2", "G3"}:
    if gold is None:
      raise ValueError(f"{cfg.condition} requested but no gold trajectory for {meta.task_id}")
    gp, gold_imgs = gold_trajectory_parts(gold, cfg)
    parts += gp
  ap, agent_imgs = agent_trajectory_parts(traj, cfg)
  parts += ap
  parts.append(_txt(
    "Request any screenshots you need with view_screenshots, then produce the JSON verdict."
    if cfg.on_demand else "Now produce the JSON verdict."
  ))
  return Bundle(
    content=parts,
    n_images=len(agent_imgs) + len(gold_imgs),
    agent_image_steps=agent_imgs,
    gold_image_steps=gold_imgs,
    meta={
      "condition": cfg.condition,
      "task_id": traj.task_id,
      "domain": traj.domain,
      "agent": traj.agent,
      "n_steps": traj.n_steps,
      "gold_key": gold.key if gold else None,
      "gold_steps": gold.n_steps if gold else None,
    },
  )
