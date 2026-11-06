"""Helpers for parsing executed ``pyautogui`` code and model responses."""

from __future__ import annotations

import re
from pathlib import Path

from goldjudge.schema import ActionType

_NUM = r"(?:x\s*=\s*)?(-?[0-9]+(?:\.[0-9]+)?)\s*,\s*(?:y\s*=\s*)?(-?[0-9]+(?:\.[0-9]+)?)"
CLICK_RE = re.compile(r"pyautogui\.(click|doubleClick|rightClick|middleClick|tripleClick)\s*\(\s*" + _NUM, re.I)
MOVE_RE = re.compile(r"pyautogui\.moveTo\s*\(\s*" + _NUM, re.I)
DRAG_RE = re.compile(r"pyautogui\.dragTo\s*\(\s*" + _NUM, re.I)
MOUSEDOWN_RE = re.compile(r"pyautogui\.mouseDown\s*\(\s*" + _NUM, re.I)
BUTTON_RE = re.compile(r"button\s*=\s*['\"](left|right|middle)['\"]", re.I)
CLICKS_RE = re.compile(r"clicks\s*=\s*(\d+)", re.I)


def parse_action_type(code: str) -> ActionType:
  """Classify executed pyautogui code into a coarse action type."""
  if not code:
    return "other"
  s = code.strip()
  upper = s.upper()
  if upper in {"DONE", "TERMINATE", "FINISH"} or "computer.terminate" in s:
    return "done"
  if upper == "FAIL":
    return "fail"
  if upper in {"WAIT", "SLEEP"} or re.fullmatch(r"(?:import\s+time;?\s*)?time\.sleep\([^)]*\)\s*;?", s):
    return "wait"
  if "CALL_USER" in upper:
    return "call_user"
  low = s.lower()
  if "dragto" in low or "mousedown" in low and "mouseup" in low:
    return "drag"
  if "scroll" in low:
    return "scroll"
  if "doubleclick" in low or (CLICKS_RE.search(s) and CLICKS_RE.search(s).group(1) == "2"):
    return "double_click"
  if "rightclick" in low or re.search(r"button\s*=\s*['\"]right", low):
    return "right_click"
  if "middleclick" in low:
    return "middle_click"
  if "hotkey" in low:
    return "hotkey"
  if "press(" in low or "keydown" in low or "keyup" in low:
    return "key"
  if "write(" in low or "typewrite(" in low:
    return "type"
  if "click(" in low or "tripleclick" in low:
    return "click"
  if "moveto" in low:
    return "move"
  if "sleep" in low:
    return "wait"
  return "other"


def parse_coords(code: str) -> list[float] | None:
  """Executed target coordinates: click > dragTo > mouseDown > moveTo."""
  if not code:
    return None
  for pat in (CLICK_RE, DRAG_RE, MOUSEDOWN_RE, MOVE_RE):
    m = pat.search(code)
    if m:
      g = m.groups()
      return [float(g[-2]), float(g[-1])]
  return None


def normalize_result(raw: str | None) -> float | None:
  if raw is None:
    return None
  s = raw.strip()
  if not s:
    return None
  try:
    return float(s)
  except ValueError:
    low = s.lower()
    if low in {"success", "passed", "pass", "true"}:
      return 1.0
    if low in {"failure", "failed", "fail", "false"}:
      return 0.0
  return None


STEP_PNG_RE = re.compile(r"^step_(\d+)_(.+)\.png$", re.I)


def step_index_from_png(name: str) -> int | None:
  m = STEP_PNG_RE.match(name)
  return int(m.group(1)) if m else None


def attach_before_after(steps: list, shots: dict[int, str]) -> None:
  """OSWorld saves step_<t>.png after action t: set screen_after = step_t, screen_before = step_(t-1) (step_0 if logged)."""
  for s in steps:
    s.screen_after = shots.get(s.step, "")
    s.screen_before = shots.get(s.step - 1, "")


_TS_DATE_RE = re.compile(r"(20\d{2})(\d{2})(\d{2})")


def run_date(traj) -> str | None:
  """YYYY-MM-DD the trajectory was recorded, from its first step timestamp or screenshot file name."""
  for s in traj.steps[:3]:
    for src in (s.timestamp, s.screen_after, s.screenshot_path):
      m = _TS_DATE_RE.search(Path(str(src)).name if src else "")
      if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
  return None
