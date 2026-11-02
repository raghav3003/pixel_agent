"""Text action logs: one replayable action grammar for every trajectory source.

Executed ``pyautogui`` code (all agents) and OSWorld-Human text steps are both turned
into a list of :class:`Op` per step:

* ``parse_code``   — executed code -> ops. The code is **parsed with ``ast``, never
  executed**; only literal arguments are read. A step that contains anything other than
  ``pyautogui`` / ``time.sleep`` calls (``subprocess``, ``uno``, ``wmctrl``…) becomes a
  single ``CODE`` op carrying the source verbatim, so it still replays.
* ``compact``      — raw pyautogui events -> intent-level ops: ``keyDown/keyUp`` pairs
  become ``PRESS`` / ``HOTKEY``, a held modifier around a click becomes ``mods``,
  ``moveTo`` + ``dragTo`` / ``mouseDown..mouseUp`` become ``DRAG``, runs of
  single-character ``press`` become ``TYPE``. Sleeps that pad another action are dropped
  (harness settle time); a step that is *only* a sleep stays ``WAIT``.
* ``parse_human``  — OSWorld-Human ``"`CLICK` the text box labeled 'Search'"`` -> an op
  whose ``target`` is the element description (no coordinates).

Each op renders to one text line (``render``) for LLM prompts and back to pyautogui
(``to_pyautogui``) for replay. Coordinates are executed screen pixels (1920x1080).
``target`` / ``effect`` are empty here and filled by the VLM captioning pass
(``scripts/caption_actions.py``), which makes a log replayable by a grounding model in an
environment whose layout differs from the recorded one.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from goldjudge.schema import Trajectory

OpName = Literal[
  "CLICK", "DOUBLE_CLICK", "TRIPLE_CLICK", "RIGHT_CLICK", "MIDDLE_CLICK", "MOVE", "DRAG",
  "MOUSE_DOWN", "MOUSE_UP", "SCROLL", "HSCROLL", "TYPE", "PRESS", "HOTKEY", "KEY_DOWN", "KEY_UP",
  "WAIT", "DONE", "FAIL", "CALL_USER", "CODE", "OTHER",
]

MODIFIERS = {"ctrl", "control", "shift", "alt", "option", "cmd", "command", "win", "super", "meta", "ctrlleft",
             "ctrlright", "shiftleft", "shiftright", "altleft", "altright", "winleft", "winright"}
_CLICK_OPS = {"click": "CLICK", "doubleclick": "DOUBLE_CLICK", "tripleclick": "TRIPLE_CLICK",
              "rightclick": "RIGHT_CLICK", "middleclick": "MIDDLE_CLICK"}


class Op(BaseModel):
  op: OpName
  x: float | None = None
  y: float | None = None
  to: list[float] | None = Field(default=None, description="DRAG end point")
  keys: list[str] = Field(default_factory=list, description="PRESS/HOTKEY keys, or click modifiers")
  text: str | None = Field(default=None, description="TYPE text; CODE source; OTHER raw text")
  amount: float | None = Field(default=None, description="scroll clicks (+up) or wait seconds")
  repeat: int = 1
  target: str = Field(default="", description="semantic description of the acted-on element")
  warn: str = ""

  def render(self) -> str:
    o = self.op
    pos = f" @({self.x:g},{self.y:g})" if self.x is not None and self.y is not None else ""
    tgt = f' "{self.target}"' if self.target else ""
    rep = f" x{self.repeat}" if self.repeat > 1 else ""
    if o in {"CLICK", "DOUBLE_CLICK", "TRIPLE_CLICK", "RIGHT_CLICK", "MIDDLE_CLICK", "MOVE", "MOUSE_DOWN", "MOUSE_UP"}:
      mods = f" +{'+'.join(self.keys)}" if self.keys else ""
      s = f"{o}{tgt}{pos}{mods}{rep}"
    elif o == "DRAG":
      end = f" -> ({self.to[0]:g},{self.to[1]:g})" if self.to else ""
      s = f"DRAG{tgt}{pos}{end}" + (f" +{'+'.join(self.keys)}" if self.keys else "")
    elif o in {"SCROLL", "HSCROLL"}:
      s = f"{o} {self.amount:g}{pos}{tgt}" if self.amount is not None else f"{o}{pos}{tgt}"
    elif o == "TYPE":
      s = f"TYPE {json.dumps(self.text or '', ensure_ascii=False)}{tgt}"
    elif o in {"PRESS", "KEY_DOWN", "KEY_UP"}:
      s = f"{o} {'+'.join(self.keys)}{rep}"
    elif o == "HOTKEY":
      s = f"HOTKEY {'+'.join(self.keys)}{rep}"
    elif o == "WAIT":
      s = f"WAIT {self.amount:g}s" if self.amount is not None else "WAIT"
    elif o in {"CODE", "OTHER"}:
      body = (self.text or "").strip().replace("\n", "\\n")
      s = f"{o} {body[:300]}{'…' if len(body) > 300 else ''}{tgt}"
    else:
      s = o + tgt
    return s + (f"  !{self.warn}" if self.warn else "")

  def to_pyautogui(self) -> str:
    o, xy = self.op, (f"{self.x:g}, {self.y:g}" if self.x is not None and self.y is not None else "")
    held = list(self.keys) if o in _CLICK_OPS.values() or o == "DRAG" else []
    pre = "".join(f"pyautogui.keyDown({k!r}); " for k in held)
    post = "".join(f"; pyautogui.keyUp({k!r})" for k in reversed(held))
    if o == "CLICK":
      core = f"pyautogui.click({xy})"
    elif o == "DOUBLE_CLICK":
      core = f"pyautogui.doubleClick({xy})"
    elif o == "TRIPLE_CLICK":
      core = f"pyautogui.tripleClick({xy})"
    elif o == "RIGHT_CLICK":
      core = f"pyautogui.rightClick({xy})"
    elif o == "MIDDLE_CLICK":
      core = f"pyautogui.middleClick({xy})"
    elif o == "MOVE":
      core = f"pyautogui.moveTo({xy})"
    elif o == "MOUSE_DOWN":
      core = (f"pyautogui.moveTo({xy}); " if xy else "") + "pyautogui.mouseDown()"
    elif o == "MOUSE_UP":
      core = (f"pyautogui.moveTo({xy}); " if xy else "") + "pyautogui.mouseUp()"
    elif o == "DRAG":
      end = f"{self.to[0]:g}, {self.to[1]:g}" if self.to else ""
      core = (f"pyautogui.moveTo({xy}); " if xy else "") + f"pyautogui.dragTo({end}, duration=0.5, button='left')"
    elif o in {"SCROLL", "HSCROLL"}:
      fn = "scroll" if o == "SCROLL" else "hscroll"
      core = (f"pyautogui.moveTo({xy}); " if xy else "") + f"pyautogui.{fn}({self.amount:g})"
    elif o == "TYPE":
      core = f"pyautogui.write({self.text or ''!r}, interval=0.02)"
    elif o == "PRESS":
      key = self.keys[0] if len(self.keys) == 1 else self.keys
      core = f"pyautogui.press({key!r}" + (f", presses={self.repeat})" if self.repeat > 1 else ")")
    elif o == "HOTKEY":
      core = "; ".join([f"pyautogui.hotkey({', '.join(repr(k) for k in self.keys)})"] * self.repeat)
    elif o == "KEY_DOWN":
      core = "; ".join(f"pyautogui.keyDown({k!r})" for k in self.keys)
    elif o == "KEY_UP":
      core = "; ".join(f"pyautogui.keyUp({k!r})" for k in self.keys)
    elif o == "WAIT":
      core = f"time.sleep({self.amount if self.amount is not None else 2:g})"
    elif o == "CODE":
      return self.text or ""
    elif o in {"DONE", "FAIL", "CALL_USER"}:
      return o
    else:
      return f"# {self.render()}"
    return pre + core + post


class ActionStep(BaseModel):
  step: int
  ops: list[Op]
  intent: str = Field(default="", description="the agent's own stated intent (Action: line / plan_code)")
  effect: str = Field(default="", description="visible state change after the step (captioning pass)")
  screenshot_path: str = ""

  def render(self) -> str:
    return " ; ".join(o.render() for o in self.ops) or "NOOP"


class ActionLog(BaseModel):
  """Replayable text action log of one trajectory (agent run or gold)."""

  key: str
  source: str
  agent: str
  task_id: str
  domain: str
  result: float | None = None
  screen: tuple[int, int] = (1920, 1080)
  steps: list[ActionStep]
  extra: dict[str, Any] = Field(default_factory=dict)

  def render(self, prefix: str = "s") -> str:
    lines = []
    for s in self.steps:
      line = f"{prefix}{s.step}: {s.render()}"
      if s.effect:
        line += f"  => {s.effect}"
      lines.append(line)
    return "\n".join(lines)

  def to_pyautogui(self) -> list[str]:
    """One code string per step, in order (``DONE``/``FAIL`` are control strings, as in OSWorld)."""
    return ["import pyautogui, time\n" + "\n".join(o.to_pyautogui() for o in s.ops) if s.ops else "" for s in self.steps]


# --- executed code -> raw events ----------------------------------------------------------------

_CONTROL = {"DONE": "DONE", "FAIL": "FAIL", "WAIT": "WAIT", "CALL_USER": "CALL_USER", "TERMINATE": "DONE", "FINISH": "DONE"}


class _NotLiteral(Exception):
  pass


def _lit(node: ast.AST) -> Any:
  try:
    return ast.literal_eval(node)
  except (ValueError, SyntaxError, TypeError) as exc:
    raise _NotLiteral from exc


def _split_statements(code: str) -> list[ast.stmt] | None:
  try:
    return ast.parse(code).body
  except SyntaxError:
    pass
  # Some harnesses join statements with stray separators; try each ';' / newline chunk alone.
  out: list[ast.stmt] = []
  for chunk in re.split(r";|\n", code):
    chunk = chunk.strip()
    if not chunk:
      continue
    try:
      out.extend(ast.parse(chunk).body)
    except SyntaxError:
      return None
  return out


def _call_events(stmts: list[ast.stmt]) -> list[tuple[str, list[Any], dict[str, Any]]] | None:
  """``[(fn, args, kwargs)]`` for pyautogui.* / time.sleep calls; None if anything else is present."""
  events: list[tuple[str, list[Any], dict[str, Any]]] = []
  for st in stmts:
    if isinstance(st, (ast.Import, ast.ImportFrom)):
      continue
    if not (isinstance(st, ast.Expr) and isinstance(st.value, ast.Call)):
      return None
    call = st.value
    f = call.func
    if not (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)):
      return None
    mod, fn = f.value.id, f.attr
    if mod == "time" and fn == "sleep":
      fn = "sleep"
    elif mod == "computer" and fn.lower() in _CLICK_OPS:
      fn = "computer." + fn  # OpenCUA's computer.tripleClick reached the machine untranslated
    elif mod != "pyautogui":
      return None
    try:
      args = [_lit(a) for a in call.args]
      kwargs = {k.arg: _lit(k.value) for k in call.keywords if k.arg}
    except _NotLiteral:
      return None
    events.append((fn, args, kwargs))
  return events


def _xy(args: list[Any], kwargs: dict[str, Any]) -> tuple[float | None, float | None]:
  x = kwargs.get("x", args[0] if len(args) > 0 else None)
  y = kwargs.get("y", args[1] if len(args) > 1 else None)
  if isinstance(x, (tuple, list)) and len(x) == 2:  # click((x, y))
    x, y = x
  if isinstance(x, (int, float)) and isinstance(y, (int, float)):
    return float(x), float(y)
  return None, None


def _key(k: Any) -> str:
  s = str(k)
  return s if len(s) == 1 else s.strip().lower()  # keep ' ' and the case of single characters


def _keys_arg(args: list[Any], kwargs: dict[str, Any], name: str = "keys") -> list[str]:
  if name in kwargs:
    v = kwargs[name]
    return [_key(x) for x in v] if isinstance(v, (list, tuple)) else [_key(v)]
  if len(args) == 1 and isinstance(args[0], (list, tuple)):
    return [_key(x) for x in args[0]]
  return [_key(a) for a in args]


def _bad_key(k: str) -> str:
  return f"key {k!r} is not a single pyautogui key (executed as a no-op)" if " " in k.strip() and k.strip() != " " else ""


class _Hold:
  def __init__(self, key: str, start: int):
    self.key, self.start, self.chord = key, start, []  # chord: keys pressed while this modifier was down


def events_to_ops(events: list[tuple[str, list[Any], dict[str, Any]]]) -> list[Op]:
  """Raw pyautogui events -> intent-level ops (see module docstring)."""
  ops: list[Op] = []
  cur: tuple[float | None, float | None] = (None, None)
  holds: list[_Hold] = []  # keys currently down (keyDown without keyUp yet), outermost first
  down_at: tuple[float | None, float | None] | None = None  # mouseDown position

  def mods() -> list[str]:
    return [h.key for h in holds if h.key in MODIFIERS]

  def pop_move(at: tuple[float | None, float | None] | None = None) -> None:
    """Drop the moveTo that only positioned the next action (looking past padding sleeps)."""
    i = len(ops) - 1
    while i >= 0 and ops[i].op == "WAIT":
      i -= 1
    if i >= 0 and ops[i].op == "MOVE" and (at is None or (ops[i].x, ops[i].y) == at):
      del ops[i:]
      for h in holds:
        h.start = min(h.start, len(ops))

  for fn, args, kw in events:
    alias = fn.startswith("computer.")
    lf = fn.lower().removeprefix("computer.")
    if lf in _CLICK_OPS:
      x, y = _xy(args, kw)
      if x is None:
        x, y = cur
      pop_move((x, y))
      name = _CLICK_OPS[lf]
      btn = str(kw.get("button", "left")).lower()
      clicks = int(kw.get("clicks", 1) or 1)
      if name == "CLICK":
        name = {"right": "RIGHT_CLICK", "middle": "MIDDLE_CLICK"}.get(btn, name)
        name = {2: "DOUBLE_CLICK"}.get(clicks, "TRIPLE_CLICK" if clicks >= 3 else name) if name == "CLICK" else name
      ops.append(Op(op=name, x=x, y=y, keys=mods(),
                    warn=f"executed as computer.{fn[9:]} (not a pyautogui call; likely a no-op)" if alias else ""))
      cur = (x, y)
    elif lf == "moveto":
      cur = _xy(args, kw)
      ops.append(Op(op="MOVE", x=cur[0], y=cur[1]))
    elif lf in {"dragto", "drag"}:
      x, y = _xy(args, kw)
      start = cur
      if lf == "drag" and x is not None and start[0] is not None:  # relative
        x, y = start[0] + x, start[1] + y
      pop_move()
      ops.append(Op(op="DRAG", x=start[0], y=start[1], to=[x, y] if x is not None else None, keys=mods()))
      cur, down_at = (x, y), None
    elif lf == "mousedown":
      x, y = _xy(args, kw)
      pop_move()
      if x is not None:
        cur = (x, y)
      down_at = cur
      ops.append(Op(op="MOUSE_DOWN", x=cur[0], y=cur[1]))
    elif lf == "mouseup":
      x, y = _xy(args, kw)
      if x is not None:
        cur = (x, y)
      if down_at is None and ops and ops[-1].op == "DRAG":
        continue  # dragTo already released the button (gemini: dragTo; mouseUp)
      if down_at is not None:
        pop_move()
        if ops and ops[-1].op == "MOUSE_DOWN":
          ops.pop()
          ops.append(Op(op="DRAG", x=down_at[0], y=down_at[1], to=[cur[0], cur[1]], keys=mods()) if cur != down_at
                     else Op(op="CLICK", x=cur[0], y=cur[1], keys=mods()))
          down_at = None
          continue
      ops.append(Op(op="MOUSE_UP", x=cur[0], y=cur[1]))
      down_at = None
    elif lf in {"scroll", "vscroll", "hscroll"}:
      amt = kw.get("clicks", args[0] if args else None)
      x, y = _xy(args[1:], kw)
      if x is None:
        x, y = cur
      pop_move((x, y))
      ops.append(Op(op="HSCROLL" if lf == "hscroll" else "SCROLL", x=x, y=y, amount=float(amt) if amt is not None else None))
    elif lf in {"write", "typewrite"}:
      v = kw.get("message", args[0] if args else "")
      if isinstance(v, (list, tuple)):  # typewrite(['a', 'enter']) = key presses
        for k in v:
          _press(ops, holds, _key(k))
      else:
        _type(ops, str(v))
    elif lf == "press":
      keys = _keys_arg(args[:1], kw)
      n = int(kw.get("presses", args[1] if len(args) > 1 and isinstance(args[1], int) else 1) or 1)
      for k in keys:
        for _ in range(n):
          _press(ops, holds, k)
    elif lf == "hotkey":
      keys = _keys_arg(args, kw)
      if len(keys) == 1:
        _press(ops, holds, keys[0])
      else:
        _chord(ops, mods() + [k for k in keys if k not in mods()])
    elif lf == "keydown":
      holds.append(_Hold(_key(args[0] if args else kw.get("key", "")), len(ops)))
    elif lf == "keyup":
      _release(ops, holds, _key(args[0] if args else kw.get("key", "")))
    elif lf == "sleep":
      ops.append(Op(op="WAIT", amount=float(args[0]) if args else None))
    else:
      ops.append(Op(op="OTHER", text=f"pyautogui.{fn}({', '.join(map(repr, args))})"))

  # Keys still down at the end of the step (released in a later step): keep them explicit.
  for h in reversed(holds):
    ops.insert(h.start, Op(op="KEY_DOWN", keys=[h.key], warn=_bad_key(h.key)))
  # Sleeps that pad another action are harness settle time, not intent.
  if any(o.op != "WAIT" for o in ops):
    ops = [o for o in ops if o.op != "WAIT"]
  elif len(ops) > 1:
    ops = [Op(op="WAIT", amount=sum(o.amount or 0 for o in ops))]
  return ops


def _type(ops: list[Op], text: str) -> None:
  if ops and ops[-1].op == "TYPE":
    ops[-1].text = (ops[-1].text or "") + text
  else:
    ops.append(Op(op="TYPE", text=text))


def _chord(ops: list[Op], keys: list[str]) -> None:
  if ops and ops[-1].op == "HOTKEY" and ops[-1].keys == keys:
    ops[-1].repeat += 1
  else:
    ops.append(Op(op="HOTKEY", keys=keys))


def _press(ops: list[Op], holds: list[_Hold], k: str) -> None:
  mod_holds = [h for h in holds if h.key in MODIFIERS]
  if mod_holds:  # press while a modifier is held = a chord
    _chord(ops, [h.key for h in mod_holds] + [k])
    mod_holds[-1].chord.append(k)
    return
  if len(k) == 1:  # Claude's xdotool typing: one press() per character
    _type(ops, k)
    return
  canon = {"return": "enter", "esc": "escape"}.get(k, k)
  if ops and ops[-1].op == "PRESS" and ops[-1].keys == [canon]:
    ops[-1].repeat += 1
  else:
    ops.append(Op(op="PRESS", keys=[canon], warn=_bad_key(canon)))


def _release(ops: list[Op], holds: list[_Hold], k: str) -> None:
  """keyUp: fold the matching keyDown into PRESS / HOTKEY / click modifiers."""
  idx = next((i for i in range(len(holds) - 1, -1, -1) if holds[i].key == k), None)
  if idx is None:  # released a key pressed in an earlier step
    ops.append(Op(op="KEY_UP", keys=[k]))
    return
  h = holds.pop(idx)
  between = ops[h.start:]
  outer = [o for o in holds if o.key in MODIFIERS]
  if not between:
    if h.chord:  # keyDown(ctrl) keyDown(a) keyUp(a) keyUp(ctrl)
      if outer:
        outer[-1].chord += [k] + h.chord
      else:
        ops.append(Op(op="HOTKEY", keys=[k] + h.chord))
    elif outer:  # nested inside a modifier hold: a chord member
      outer[-1].chord.append(k)
    else:  # keyDown(k) keyUp(k)
      bad = _bad_key(k)
      ops.append(Op(op="PRESS", keys=[k], warn=bad)) if bad else _press(ops, [], k)
    return
  if k in MODIFIERS:
    return  # already recorded as click/drag modifiers or HOTKEY chords
  ops.insert(h.start, Op(op="KEY_DOWN", keys=[k]))
  ops.append(Op(op="KEY_UP", keys=[k]))


_WRITE_ONLY_RE = re.compile(r"^(?:import pyautogui;?\s*)?pyautogui\.(typewrite|write)\(\s*(['\"])(.*)\2\s*\)\s*;?$", re.S)


def parse_code(code: str) -> list[Op]:
  """Executed step code -> ops. Never executes the code."""
  s = (code or "").strip()
  if not s:
    return []
  if s.upper() in _CONTROL:
    c = _CONTROL[s.upper()]
    return [Op(op=c)] if c != "WAIT" else [Op(op="WAIT")]
  m = _WRITE_ONLY_RE.match(s)
  if m and _split_statements(s) is None:  # a typed multi-line string the harness wrote unescaped
    return [Op(op="TYPE", text=m.group(3), warn="unparseable string literal; text taken verbatim")]
  stmts = _split_statements(s)
  events = _call_events(stmts) if stmts is not None else None
  if events is None:
    return [Op(op="CODE", text=s)]
  return events_to_ops(events)


# --- OSWorld-Human text -------------------------------------------------------------------------

_HUMAN_VERB = {
  "CLICK": "CLICK", "TYPING": "TYPE", "TYPE": "TYPE", "PRESS": "PRESS", "RRESS": "PRESS", "HOTKEY": "HOTKEY",
  "HOTEKEY": "HOTKEY", "HOTEKY": "HOTKEY", "RIGHT_CLICK": "RIGHT_CLICK", "DOUBLE_CLICK": "DOUBLE_CLICK",
  "DOUBLECLICK": "DOUBLE_CLICK", "TRIPLE_CLICK": "TRIPLE_CLICK", "SCROLL": "SCROLL", "SCROLL_LEFT": "HSCROLL",
  "MOVE_TO": "MOVE", "MOVE_ON": "MOVE", "DRAG_TO": "DRAG", "DRAG": "DRAG", "DRAG_AND_DROP": "DRAG",
  "MOUSE_DOWN": "MOUSE_DOWN", "MOUSE_UP": "MOUSE_UP", "KEY_DOWN": "KEY_DOWN", "KEYDOWN": "KEY_DOWN",
  "KEY DOWN": "KEY_DOWN", "KEY_UP": "KEY_UP", "KEYUP": "KEY_UP", "KEY UP": "KEY_UP", "WAIT": "WAIT",
  "FAIL": "FAIL", "DONE": "DONE", "SHELL": "CODE", "CHECK": "CLICK", "SELECT": "CLICK",
}
_BACKTICK_RE = re.compile(r"^\s*(\d+x)?\s*`([^`]*)`?(.*)$", re.S)
_PLAIN_RE = re.compile(r"^\s*(\d+x)?\s*(.*)$", re.S)
_HUMAN_KEYS = {"control": "ctrl", "return": "enter", "esc": "escape", "del": "delete", "page down": "pagedown",
               "page up": "pageup", "down arrow": "down", "up arrow": "up", "left arrow": "left",
               "right arrow": "right", "arrow down": "down", "arrow up": "up", "arrow left": "left",
               "arrow right": "right", "windows": "win", "super": "win", "spacebar": "space"}
_SCROLL_UNITS_RE = re.compile(r"\((-?\d+)\)")
_QUOTED_RE = re.compile(r"^['\"](.*)['\"]\.?$", re.S)


def parse_human(line: str) -> Op:
  """One OSWorld-Human step (``"`VERB` rest"``, loosely formatted) -> an op with ``target``."""
  text = line.replace("``", "`")
  m = _BACKTICK_RE.match(text) if "`" in text else None
  rep, head, tail = (m.group(1), m.group(2), m.group(3)) if m else (*_PLAIN_RE.match(text).groups(), "")
  # The verb is the leading upper-case word(s); some lines put the argument inside the backticks
  # ("`MOVE_TO bottom right corner`") or use no backticks at all ("TYPING Paste each ...").
  words = head.split()
  op = None
  for n in (2, 1):
    cand = " ".join(words[:n])
    if len(words) >= n and cand == cand.upper() and cand in _HUMAN_VERB:
      op, rest = _HUMAN_VERB[cand], (" ".join(words[n:]) + " " + tail).strip().strip("`").strip()
      break
  if op is None:
    return Op(op="OTHER", text=line.strip())
  repeat = int(rep[:-1]) if rep else 1
  if op == "TYPE":
    q = _QUOTED_RE.match(rest)
    return Op(op="TYPE", text=q.group(1) if q else rest, repeat=repeat)
  if op in {"PRESS", "HOTKEY", "KEY_DOWN", "KEY_UP"}:
    keyspec = rest.strip("'\". ").split(" to ")[0].split(" (")[0]
    keys = [_HUMAN_KEYS.get(_key(k), _key(k)).lower()
            for k in re.split(r"\s*[+]\s*|(?<=[a-z])-(?=[a-z])", keyspec, flags=re.I) if k.strip()]
    return Op(op=op, keys=keys, repeat=repeat, target=rest if len(rest) > len(keyspec) + 2 else "")
  if op in {"SCROLL", "HSCROLL"}:
    um = _SCROLL_UNITS_RE.search(rest)
    return Op(op=op, amount=float(um.group(1)) if um else None, target=rest, repeat=repeat)
  if op == "CODE":
    return Op(op="CODE", text=rest)
  if op == "WAIT":
    sec = re.match(r"(\d+(?:\.\d+)?)\s*s", rest)
    return Op(op="WAIT", amount=float(sec.group(1)) if sec else None, target="" if sec else rest)
  return Op(op=op, target=re.sub(r"^(?:on|at)\s+", "", rest), repeat=repeat)


# --- trajectories -> logs -----------------------------------------------------------------------


def action_log(traj: Trajectory) -> ActionLog:
  steps = []
  for s in traj.steps:
    steps.append(ActionStep(step=s.step, ops=parse_code(s.action_raw),
                            intent=(s.intent or "").strip()[:300], screenshot_path=s.screenshot_path))
  return ActionLog(key=traj.key, source=traj.source, agent=traj.agent, task_id=traj.task_id, domain=traj.domain,
                   result=traj.result, screen=traj.screen, steps=steps)


def human_action_log(human: Any, *, result: float | None = 1.0) -> ActionLog:
  """OSWorld-Human reference -> log. Steps follow ``grouped_action`` (one human "step" per group)."""
  groups = human.grouped_action or [[a] for a in human.single_action]
  steps = [ActionStep(step=i, ops=[parse_human(a) for a in g]) for i, g in enumerate(groups, 1)]
  return ActionLog(key=f"osworld-human/{human.domain}/{human.task_id}", source="osworld-human", agent="osworld-human",
                   task_id=human.task_id, domain=human.domain, result=result, steps=steps)
