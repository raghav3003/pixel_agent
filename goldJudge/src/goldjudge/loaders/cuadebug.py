"""Loader for the CUADebug archives (``CyT1ng/cua_debugger_traj``).

Three agents, three ``traj.jsonl`` schemas:

* ``claude_4.5_traj_144`` — Anthropic computer-use tool; ``action`` is a dict with
  ``input`` (tool args, coords in 1280x720), ``command`` (executed pyautogui) and
  ``raw_response`` (``[THINKING]`` / ``[TEXT]`` / ``[TOOL_USE]`` blocks). Several rows
  may share a ``step_num`` (one per tool call in a turn).
* ``gemini_2.5_pro_traj_30`` — planner/executor; ``executor_plan`` is the CoT,
  ``plan_code`` the grounded intent, ``action`` the executed pyautogui.
* ``qwen_3.5_traj_30`` — ``response`` holds ``Action: ...`` plus a ``<tool_call>``
  block with coords in a 1000x1000 space; ``action`` is executed pyautogui.

All three write ``human_annotation.json`` and ``result.txt`` per task directory.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterator

from goldjudge.parsing import attach_before_after, normalize_result, parse_action_type, parse_coords, step_index_from_png
from goldjudge.schema import Annotation, ModelAction, Step, Trajectory

DEFAULT_ROOT = Path(
  os.environ.get("CUADEBUG_ROOT", "/ocean/projects/cis260247p/rgupta19/data/cua_debugger_traj")
)

ARCHIVES: dict[str, dict[str, Any]] = {
  "claude_4.5_traj_144": {
    "agent": "claude-sonnet-4-5",
    "horizon": 50,
    "model_coord_space": (1280, 720),
  },
  "gemini_2.5_pro_traj_30": {"agent": "gemini-2.5-pro", "horizon": 50, "model_coord_space": None},
  "qwen_3.5_traj_30": {"agent": "qwen3.5-vl", "horizon": 100, "model_coord_space": (1000, 1000)},
}

TAG_FAMILY = {
  "P": "Perception",
  "G": "Grounding & Interaction",
  "R": "Task Reasoning & Control",
  "S": "External / System",
  "IF": "Infeasible Task",
}


def _family_of(tag: str) -> str:
  if tag.startswith("IF") or tag.lower().startswith("infeasible"):
    return TAG_FAMILY["IF"]
  return TAG_FAMILY.get(tag[:1], "")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  with path.open(encoding="utf-8") as fh:
    for line in fh:
      line = line.strip()
      if line:
        rows.append(json.loads(line))
  return rows


def _screenshot_map(task_dir: Path) -> dict[int, str]:
  out: dict[int, str] = {}
  for p in task_dir.glob("step_*.png"):
    idx = step_index_from_png(p.name)
    if idx is not None and idx not in out:
      out[idx] = str(p)
  return out


# --- per-agent step parsers -------------------------------------------------

_THINK_RE = re.compile(r"\[THINKING\]\s*(.*?)(?=\n\[(?:TEXT|TOOL_USE)\]|\Z)", re.S)
_TEXT_RE = re.compile(r"\[TEXT\]\s*(.*?)(?=\n\[(?:THINKING|TOOL_USE)\]|\Z)", re.S)


def _parse_claude(row: dict[str, Any]) -> tuple[str, str, str, ModelAction | None, bool]:
  a = row.get("action") or {}
  raw = str(a.get("raw_response") or "")
  thinking = "\n".join(m.strip() for m in _THINK_RE.findall(raw))
  text = "\n".join(m.strip() for m in _TEXT_RE.findall(raw)) or str(row.get("response") or "")
  thought = (thinking + ("\n\n" if thinking and text else "") + text).strip()
  if a.get("action_type") == "DONE":
    return thought, "DONE", "DONE", ModelAction(type="done", raw=a), True
  inp = a.get("input") or {}
  command = str(a.get("command") or "")
  ma = ModelAction(
    type=str(inp.get("action") or ""),
    coordinate=[float(x) for x in inp["coordinate"]] if inp.get("coordinate") else None,
    text=inp.get("text"),
    raw=inp,
  )
  intent = f"{ma.type} {ma.coordinate or ''} {ma.text or ''}".strip()
  return thought, intent, command, ma, False


_QWEN_ACTION_RE = re.compile(r"^\s*Action:\s*(.*?)(?=\n\s*<tool_call>|\Z)", re.S | re.M)
_QWEN_PARAM_RE = re.compile(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", re.S)


def _parse_qwen(row: dict[str, Any]) -> tuple[str, str, str, ModelAction | None, bool]:
  resp = str(row.get("response") or "")
  m = _QWEN_ACTION_RE.search(resp)
  intent = m.group(1).strip() if m else ""
  params = {k: v.strip() for k, v in _QWEN_PARAM_RE.findall(resp)}
  coord = None
  if params.get("coordinate"):
    try:
      coord = [float(x) for x in json.loads(params["coordinate"])]
    except (ValueError, TypeError):
      coord = None
  ma = ModelAction(type=params.get("action", ""), coordinate=coord, text=params.get("text"), raw=params)
  # Reasoning = everything before "Action:" (Qwen thinking) plus the Action line.
  pre = resp.split("Action:", 1)[0].strip() if "Action:" in resp else ""
  thought = (pre + ("\n" if pre else "") + ("Action: " + intent if intent else "")).strip()
  command = str(row.get("action") or "")
  done = ma.type in {"terminate", "done"} or command.strip().upper() == "DONE"
  return thought, intent, command, ma, done


def _parse_gemini(row: dict[str, Any]) -> tuple[str, str, str, ModelAction | None, bool]:
  thought = str(row.get("executor_plan") or row.get("full_plan") or "")
  intent = str(row.get("plan_code") or "")
  command = str(row.get("action") or "")
  ma = ModelAction(type="", raw=intent) if intent else None
  done = command.strip().upper() in {"DONE", "FAIL"} or "agent.done" in intent
  return thought, intent, command, ma, done


_PARSERS = {
  "claude_4.5_traj_144": _parse_claude,
  "gemini_2.5_pro_traj_30": _parse_gemini,
  "qwen_3.5_traj_30": _parse_qwen,
}


# --- public API -----------------------------------------------------------------


def load_annotation(task_dir: Path) -> Annotation | None:
  p = task_dir / "human_annotation.json"
  if not p.exists():
    return None
  d = json.loads(p.read_text(encoding="utf-8"))
  tag = str(d.get("taxonomy_tag") or "")
  return Annotation(
    root_error_step=int(d.get("root_error_step") or 0),
    taxonomy_tag=tag,
    taxonomy_family=_family_of(tag),
    evidence=str(d.get("evidence") or ""),
    correction=str(d.get("correction") or ""),
    confidence=str(d.get("confidence") or ""),
    source="cuadebug",
  )


def load_trajectory(task_dir: Path, archive: str) -> Trajectory:
  task_dir = Path(task_dir)
  meta = ARCHIVES[archive]
  parse = _PARSERS[archive]
  rows = _read_jsonl(task_dir / "traj.jsonl")
  shots = _screenshot_map(task_dir)

  steps: list[Step] = []
  seen: dict[int, Step] = {}
  for row in rows:
    n = int(row.get("step_num") or (len(steps) + 1))
    thought, intent, command, ma, done = parse(row)
    shot = shots.get(n) or str(task_dir / str(row.get("screenshot_file") or ""))
    reward = row.get("reward")
    try:
      reward_f = float(reward) if reward not in (None, "") else None
    except (TypeError, ValueError):
      reward_f = None
    step = Step(
      step=n,
      screenshot_path=shot,
      action_raw=command,
      action_type=parse_action_type(command) if not done else "done",
      coords=parse_coords(command),
      model_action=ma,
      thought=thought,
      intent=intent,
      reflection=str(row.get("reflection") or "") if archive.startswith("gemini") else "",
      timestamp=str(row.get("action_timestamp") or ""),
      reward=reward_f,
      done=bool(done or str(row.get("done")).lower() == "true"),
    )
    if n in seen:
      # Claude logs one row per tool call within a turn: merge into the existing step.
      prev = seen[n]
      prev.action_raw = (prev.action_raw.rstrip("\n") + "\n" + command).strip()
      prev.extra.setdefault("sub_actions", []).append(
        {"action_raw": command, "model_action": ma.model_dump() if ma else None, "intent": intent}
      )
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
    source="cuadebug",
    package=archive,
    agent=meta["agent"],
    task_id=task_dir.name,
    domain=task_dir.parent.name,
    dir=str(task_dir),
    steps=steps,
    result=result,
    horizon=meta["horizon"],
    model_coord_space=meta["model_coord_space"],
    annotation=load_annotation(task_dir),
  )


def iter_task_dirs(root: Path | str = DEFAULT_ROOT, archives: list[str] | None = None) -> Iterator[tuple[str, Path]]:
  root = Path(root)
  for archive in archives or list(ARCHIVES):
    base = root / archive
    if not base.exists():
      continue
    for traj in sorted(base.glob("*/*/traj.jsonl")):
      yield archive, traj.parent


def load_all(root: Path | str = DEFAULT_ROOT, archives: list[str] | None = None) -> list[Trajectory]:
  return [load_trajectory(d, a) for a, d in iter_task_dirs(root, archives)]
