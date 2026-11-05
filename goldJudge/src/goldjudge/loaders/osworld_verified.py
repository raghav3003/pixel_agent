"""Loader for OSWorld-Verified trajectory packages (``xlangai/ubuntu_osworld_verified_trajs``).

Packages differ in top-level layout but every episode is a directory holding
``traj.jsonl`` + ``result.txt`` + ``step_N_<ts>.png`` (+ ``recording.mp4``,
``runtime.log``). We walk for ``traj.jsonl`` and derive ``<domain>/<task_id>`` from
the two trailing path components; the remaining prefix identifies the run
(e.g. ``turn_1`` for multi-seed OpenCUA packages, or the served-model id for
Claude packages).

``traj.jsonl`` row schemas seen so far:

* **OpenCUA** — ``response`` = ``# Step N:\\n## Thought: ...\\n## Action: ...\\n## Code:
  ```python ...````; ``action`` = executed pyautogui in pixels; model code uses
  normalized 0–1 coordinates.
* **Claude computer-use** (``claude_computer_use`` harness) — same as the CUADebug
  Claude schema (``action`` dict with ``input``/``command``/``raw_response``).
* Others are parsed generically (``response`` as thought, ``action`` as command).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterator

from goldjudge.loaders.cuadebug import _parse_claude
from goldjudge.parsing import attach_before_after, normalize_result, parse_action_type, parse_coords, step_index_from_png
from goldjudge.schema import ModelAction, Step, Trajectory

DEFAULT_ROOT = Path(
  os.environ.get(
    "OSWV_ROOT", "/ocean/projects/cis260247p/rgupta19/data/osworld_verified_trajs/extracted"
  )
)

# package dir name -> canonical agent id (extend as packages are added)
PACKAGE_AGENTS: dict[str, str] = {
  "claude-sonnet-4-5-20250929_50steps": "claude-sonnet-4-5",
  "intelligence-indeed-claude-sonnet-4-6-opus-4-7-nogdrive-100steps": "claude-opus-4-7",
  "opencua_agent-opencua_7b-cot_l2-action_history-3image-Ubuntu-50steps": "opencua-7b",
  "opencua_agent-opencua_a3b-cot_l2-action_history-3image-Ubuntu-50step": "opencua-a3b",
  "opencua_agent-opencua_32b-cot_l2-action_history-3image-Ubuntu-50steps": "opencua-32b",
}

_HORIZON_RE = re.compile(r"(\d+)\s*steps?", re.I)

_THOUGHT_RE = re.compile(r"##\s*Thought:\s*(.*?)(?=\n##\s*(?:Action|Code):|\Z)", re.S | re.I)
_ACTION_RE = re.compile(r"##\s*Action:\s*(.*?)(?=\n##\s*Code:|\Z)", re.S | re.I)
_CODE_RE = re.compile(r"```python\s*(.*?)```", re.S | re.I)
_NORM_XY_RE = re.compile(r"pyautogui\.\w+\(\s*(?:x\s*=\s*)?([0-9.]+)\s*,\s*(?:y\s*=\s*)?([0-9.]+)")


def _parse_opencua(row: dict[str, Any]) -> tuple[str, str, str, ModelAction | None, bool]:
  resp = str(row.get("response") or "")
  thought = (_THOUGHT_RE.search(resp).group(1).strip() if _THOUGHT_RE.search(resp) else resp.strip())
  intent = _ACTION_RE.search(resp).group(1).strip() if _ACTION_RE.search(resp) else ""
  code = _CODE_RE.search(resp).group(1).strip() if _CODE_RE.search(resp) else ""
  command = str(row.get("action") or "")
  coord = None
  m = _NORM_XY_RE.search(code)
  if m:
    coord = [float(m.group(1)), float(m.group(2))]
  ma = ModelAction(type=parse_action_type(code) if code else "", coordinate=coord, raw=code or None)
  done = command.strip().upper() in {"DONE", "FAIL"} or "computer.terminate" in code
  return thought, intent, command, ma, done


def _parse_generic(row: dict[str, Any]) -> tuple[str, str, str, ModelAction | None, bool]:
  a = row.get("action")
  if isinstance(a, dict) and "raw_response" in a:
    return _parse_claude(row)
  resp = str(row.get("response") or "")
  if "## Thought:" in resp or "## Code:" in resp:
    return _parse_opencua(row)
  if isinstance(a, dict) and set(a) <= {"action_type"}:
    # Control-flow rows, e.g. {"action_type": "WAIT"|"DONE"|"FAIL"} (Opus 4.7 package)
    kind = str(a.get("action_type") or "").upper()
    return resp, kind, kind, ModelAction(type=kind.lower(), raw=a), kind in {"DONE", "FAIL"}
  command = a if isinstance(a, str) else json.dumps(a)
  done = str(command).strip().upper() in {"DONE", "FAIL"}
  return resp, "", str(command), None, done


def _screenshot_map(task_dir: Path) -> dict[int, str]:
  out: dict[int, str] = {}
  for p in task_dir.glob("step_*.png"):
    idx = step_index_from_png(p.name)
    if idx is not None and idx not in out:
      out[idx] = str(p)
  return out


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
  rows = []
  with path.open(encoding="utf-8") as fh:
    for line in fh:
      line = line.strip()
      if line:
        try:
          rows.append(json.loads(line))
        except json.JSONDecodeError:
          continue
  return rows


def load_trajectory(task_dir: Path, *, package: str, run: str, agent: str, horizon: int | None) -> Trajectory:
  task_dir = Path(task_dir)
  rows = _read_jsonl(task_dir / "traj.jsonl")
  shots = _screenshot_map(task_dir)
  steps: list[Step] = []
  seen: dict[int, Step] = {}
  model_space: tuple[int, int] | None = None
  for row in rows:
    n = int(row.get("step_num") or (len(steps) + 1))
    thought, intent, command, ma, done = _parse_generic(row)
    if ma and ma.coordinate and isinstance(row.get("action"), dict):
      model_space = (1280, 720)  # Claude computer-use tool space
    elif ma and ma.coordinate and max(ma.coordinate) <= 1.0:
      model_space = (1, 1)  # normalized (OpenCUA)
    reward = row.get("reward")
    try:
      reward_f = float(reward) if reward not in (None, "") else None
    except (TypeError, ValueError):
      reward_f = None
    step = Step(
      step=n,
      screenshot_path=shots.get(n) or str(task_dir / str(row.get("screenshot_file") or "")),
      action_raw=command,
      action_type="done" if done else parse_action_type(command),
      coords=parse_coords(command),
      model_action=ma,
      thought=thought,
      intent=intent,
      timestamp=str(row.get("action_timestamp") or ""),
      reward=reward_f,
      done=bool(done or str(row.get("done")).lower() == "true"),
    )
    if n in seen:
      prev = seen[n]
      prev.action_raw = (prev.action_raw.rstrip("\n") + "\n" + command).strip()
      prev.extra.setdefault("sub_actions", []).append({"action_raw": command, "intent": intent})
      if prev.coords is None:
        prev.coords = step.coords
      if prev.action_type in {"other", "wait"} and step.action_type != "other":
        prev.action_type = step.action_type
      if step.thought and step.thought not in prev.thought:
        prev.thought = (prev.thought + "\n" + step.thought).strip()
      prev.done = prev.done or step.done
      continue
    seen[n] = step
    steps.append(step)
  attach_before_after(steps, shots)

  result_txt = task_dir / "result.txt"
  result = normalize_result(result_txt.read_text(encoding="utf-8", errors="replace")) if result_txt.exists() else None
  return Trajectory(
    source="osworld_verified",
    package=package,
    agent=agent,
    task_id=task_dir.name,
    domain=task_dir.parent.name,
    dir=str(task_dir),
    steps=steps,
    result=result,
    horizon=horizon,
    model_coord_space=model_space,
    extra={"run": run},
  )


def iter_episode_dirs(root: Path | str = DEFAULT_ROOT, packages: list[str] | None = None) -> Iterator[dict[str, Any]]:
  """Yield ``{package, run, agent, horizon, task_dir}`` for every episode directory."""
  root = Path(root)
  for pkg_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    package = pkg_dir.name
    if packages and package not in packages:
      continue
    agent = PACKAGE_AGENTS.get(package, package)
    hm = _HORIZON_RE.search(package)
    horizon = int(hm.group(1)) if hm else None
    for traj in sorted(pkg_dir.rglob("traj.jsonl")):
      task_dir = traj.parent
      rel = task_dir.relative_to(pkg_dir)
      run = "/".join(rel.parts[:-2]) or "run"
      yield {"package": package, "run": run, "agent": agent, "horizon": horizon, "task_dir": task_dir}


def load_all(root: Path | str = DEFAULT_ROOT, packages: list[str] | None = None) -> list[Trajectory]:
  out = []
  for ep in iter_episode_dirs(root, packages):
    out.append(load_trajectory(ep["task_dir"], package=ep["package"], run=ep["run"], agent=ep["agent"], horizon=ep["horizon"]))
  return out
