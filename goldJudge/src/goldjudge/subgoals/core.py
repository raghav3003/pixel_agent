"""Subgoal pipeline logic: request content, reply parsing, verification and localization rules.

Stages (driven by ``scripts/subgoals.py``):

1. caption     per trajectory, windows of steps -> ``target`` / ``effect`` per step (fills the action log)
2. extract     per task, one call per evidence *view* (``task`` = instruction + evaluator only; ``human`` =
               + OSWorld-Human steps; ``ai`` = + one captioned successful AI run) -> candidate subgoal lists
3. consolidate per task: merge the views into one list with support counts and necessity votes
4. verify      track the subgoals over every *successful* run of the task: an essential subgoal that some
               success never reaches is refuted (route-dependent); order edges a success violates are dropped
5. track       track the subgoals over each *failed* run
6. localize    rule: first essential subgoal (in order) never reached, or reached and lost while it had to
               persist -> error window; LLM picks the root step inside the window
"""

from __future__ import annotations

import base64
import io
import json
import re
from typing import Any

from PIL import Image, ImageDraw

from goldjudge.actions import ActionLog, Op
from goldjudge.judge.images import image_path_ok
from goldjudge.judge.verdict import _fix_escapes
from goldjudge.schema import TaskMeta, Trajectory
from goldjudge.subgoals.schema import StepCaption, StepStatus, Subgoal, SubgoalSet, Track

_FENCED_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)
_POINT_OPS = {"CLICK", "DOUBLE_CLICK", "TRIPLE_CLICK", "RIGHT_CLICK", "MIDDLE_CLICK", "MOVE", "DRAG", "SCROLL",
              "HSCROLL", "MOUSE_DOWN", "MOUSE_UP"}


# --- generic helpers ---------------------------------------------------------------------------


def parse_json(text: str, key: str) -> dict[str, Any] | None:
  """Last JSON object in ``text`` that has ``key`` (fenced blocks first; repairs bad escapes)."""
  for t in (text, _fix_escapes(text)):
    for m in reversed(list(_FENCED_RE.finditer(t))):
      try:
        obj = json.loads(m.group(1))
      except json.JSONDecodeError:
        continue
      if isinstance(obj, dict) and key in obj:
        return obj
    dec, last = json.JSONDecoder(), None
    for i, ch in enumerate(t):
      if ch == "{":
        try:
          obj, _ = dec.raw_decode(t, i)
        except json.JSONDecodeError:
          continue
        if isinstance(obj, dict) and key in obj:
          last = obj
    if last is not None:
      return last
  return None


def txt(s: str) -> dict[str, Any]:
  return {"type": "text", "text": s}


def marked_image(path: str, points: list[tuple[float, float]], screen: tuple[int, int] = (1920, 1080),
                 max_width: int = 1280, quality: int = 80) -> dict[str, Any]:
  """Screenshot with numbered red circles at executed pointer positions, as an image_url part."""
  img = Image.open(path).convert("RGB")
  sx, sy = img.width / screen[0], img.height / screen[1]
  if points:
    d = ImageDraw.Draw(img)
    r = max(10, img.width // 90)
    for i, (x, y) in enumerate(points, 1):
      cx, cy = x * sx, y * sy
      d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 0, 0), width=4)
      d.line([cx - r // 2, cy, cx + r // 2, cy], fill=(255, 0, 0), width=2)
      d.line([cx, cy - r // 2, cx, cy + r // 2], fill=(255, 0, 0), width=2)
      if len(points) > 1:
        d.text((cx + r + 2, cy - r), str(i), fill=(255, 0, 0))
  if img.width > max_width:
    img = img.resize((max_width, round(img.height * max_width / img.width)), Image.LANCZOS)
  buf = io.BytesIO()
  img.save(buf, format="JPEG", quality=quality)
  return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}}


def op_points(ops: list[Op]) -> list[tuple[float, float]]:
  pts = []
  for o in ops:
    if o.op in _POINT_OPS and o.x is not None and o.y is not None:
      pts.append((o.x, o.y))
    if o.op == "DRAG" and o.to and o.to[0] is not None:
      pts.append((o.to[0], o.to[1]))
  return pts


def trunc(s: str, n: int) -> str:
  s = (s or "").strip()
  return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def task_text(meta: TaskMeta, osw: Any, *, eval_source_chars: int = 2500, config_chars: int = 1500) -> str:
  lines = ["# TASK", f"Domain: {meta.domain}", f"Instruction: {meta.instruction}", "",
           "## Initial setup (OSWorld config: files downloaded/opened, apps launched before the agent starts)",
           trunc(json.dumps(meta.config, ensure_ascii=False), config_chars), "",
           "## What the OSWorld evaluator checks", f"evaluator spec: {trunc(json.dumps(meta.evaluator, ensure_ascii=False), 2500)}"]
  for fn in osw.evaluator_funcs(meta):
    src = osw.metric_source(fn)
    if src:
      lines += ["", f"### source of metric `{fn}` (truncated)", "```python", trunc(src, eval_source_chars), "```"]
  return "\n".join(lines)


# --- 1. captions -------------------------------------------------------------------------------


def caption_windows(n_steps: int, window: int) -> list[list[int]]:
  return [list(range(i, min(i + window, n_steps + 1))) for i in range(1, n_steps + 1, window)]


def caption_content(meta: TaskMeta, traj: Trajectory, log: ActionLog, steps: list[int], *, max_width: int) -> list[dict[str, Any]]:
  by = {s.step: s for s in traj.steps}
  lsteps = {s.step: s for s in log.steps}
  parts = [txt(f"Task the agent was given: {meta.instruction}\nSteps {steps[0]}-{steps[-1]} of {traj.n_steps}.")]
  for t in steps:
    st, ls = by.get(t), lsteps.get(t)
    if st and image_path_ok(st.screenshot_path):
      parts.append(txt(f"(screenshot before step {t})"))
      parts.append(marked_image(st.screenshot_path, op_points(ls.ops) if ls else [], traj.screen, max_width))
    intent = trunc(st.intent if st else "", 200)
    thought = trunc(st.thought if st else "", 300)
    parts.append(txt(f"step {t} executed: {ls.render() if ls else '?'}"
                     + (f"\n  agent intent: {intent}" if intent else "") + (f"\n  agent thought: {thought}" if thought else "")))
  nxt = by.get(steps[-1] + 1)
  if nxt and image_path_ok(nxt.screenshot_path):
    parts.append(txt(f"(screenshot after step {steps[-1]}, before step {steps[-1] + 1})"))
    parts.append(marked_image(nxt.screenshot_path, [], traj.screen, max_width))
  else:
    parts.append(txt(f"(no screenshot after step {steps[-1]}; give its effect as 'unknown (no later screenshot)' "
                     "unless the action makes it obvious)"))
  return parts


def parse_captions(text: str, steps: list[int]) -> list[StepCaption]:
  obj = parse_json(text, "steps") or {}
  out = []
  for r in obj.get("steps") or []:
    try:
      t = int(r.get("step"))
    except (TypeError, ValueError):
      continue
    if t in steps:
      hi = r.get("hit_intended")
      out.append(StepCaption(step=t, target=str(r.get("target") or ""), effect=str(r.get("effect") or ""),
                             hit_intended=hi if isinstance(hi, bool) else None))
  return out


def apply_captions(log: ActionLog, caps: dict[int, StepCaption]) -> ActionLog:
  """Fill ``target`` on the step's mouse/typing ops and ``effect`` on the step."""
  log = log.model_copy(deep=True)
  for s in log.steps:
    c = caps.get(s.step)
    if not c:
      continue
    s.effect = c.effect
    acted = [o for o in s.ops if o.op not in {"WAIT", "DONE", "FAIL", "CALL_USER", "KEY_DOWN", "KEY_UP"}]
    if acted and c.target:
      acted[0].target = c.target
    if c.hit_intended is False:
      s.intent = (s.intent + " [caption: target does NOT match intent]").strip()
  return log


# --- 2/3. extraction & consolidation -----------------------------------------------------------


def extract_content(task: str, view: str, *, human_log: ActionLog | None = None, ai_log: ActionLog | None = None) -> list[dict[str, Any]]:
  parts = [txt(task)]
  if view == "human" and human_log is not None:
    parts.append(txt("# REFERENCE — steps a human wrote down for this task (one possible route; may be incomplete)\n"
                     + human_log.render(prefix="h")))
  if view == "ai" and ai_log is not None:
    parts.append(txt(f"# REFERENCE — a successful run by an AI agent ({ai_log.agent}; evaluator score {ai_log.result}). "
                     "One possible route. Each line: action(s) \"element acted on\" @pixels => visible effect.\n"
                     + ai_log.render(prefix="g")))
  parts.append(txt("Write the subgoal list for this task following the rules. JSON only."))
  return parts


def parse_subgoals(text: str) -> tuple[list[Subgoal], str]:
  obj = parse_json(text, "subgoals") or {}
  out = []
  for i, r in enumerate(obj.get("subgoals") or [], 1):
    if not isinstance(r, dict):
      continue
    nec = str(r.get("necessity") or "essential").lower()
    votes = r.get("necessity_votes") if isinstance(r.get("necessity_votes"), dict) else {}
    out.append(Subgoal(
      id=str(r.get("id") or f"S{i}"), description=str(r.get("description") or ""),
      observable_forms=[str(x) for x in r.get("observable_forms") or [] if x],
      success_check=str(r.get("success_check") or ""),
      necessity=nec if nec in {"essential", "conditional", "optional"} else "essential",
      necessity_reason=str(r.get("necessity_reason") or ""),
      depends_on=[str(d) for d in r.get("depends_on") or [] if d],
      persistent=bool(r.get("persistent")), alternatives=str(r.get("alternatives") or ""),
      support=[str(x) for x in r.get("support") or []],
      necessity_votes={str(k): int(v) for k, v in votes.items() if isinstance(v, (int, float))},
    ))
  return out, str(obj.get("notes") or "")


def render_subgoals(sgs: list[Subgoal], *, full: bool = True) -> str:
  lines = []
  for s in sgs:
    line = f"{s.id} [{s.necessity}{', persistent' if s.persistent else ''}] {s.description}"
    if s.observable_forms:
      line += "\n    any of: " + " | ".join(trunc(f, 160) for f in s.observable_forms)
    if s.success_check:
      line += f"\n    check: {s.success_check}"
    if full:
      if s.depends_on:
        line += f"\n    after: {', '.join(s.depends_on)}"
      if s.necessity_reason:
        line += f"\n    why: {trunc(s.necessity_reason, 300)}"
      if s.alternatives:
        line += f"\n    alternatives: {trunc(s.alternatives, 200)}"
    lines.append(line)
  return "\n".join(lines)


def consolidate_content(task: str, candidates: dict[str, list[Subgoal]], view_names: dict[str, str]) -> list[dict[str, Any]]:
  parts = [txt(task)]
  for letter, sgs in candidates.items():
    parts.append(txt(f"# CANDIDATE LIST {letter} (written from: {view_names[letter]})\n{render_subgoals(sgs)}"))
  parts.append(txt("Merge the candidate lists into one canonical list following the rules. JSON only."))
  return parts


def finalize_consolidated(sgs: list[Subgoal], n_views: int) -> list[Subgoal]:
  ids = {s.id for s in sgs}
  for s in sgs:
    s.support = sorted(set(s.support))
    s.support_frac = round(len(s.support) / n_views, 3) if n_views else 0.0
    s.depends_on = [d for d in s.depends_on if d in ids and d != s.id]
  return sgs


# --- 4/5. tracking -----------------------------------------------------------------------------


def track_windows(steps: list[int], window: int) -> list[list[int]]:
  return [steps[i:i + window] for i in range(0, len(steps), window)]


def check_content(meta: TaskMeta, sgs: list[Subgoal], traj: Trajectory, log: ActionLog, steps: list[int],
                  prev: StepStatus | None, *, max_width: int) -> list[dict[str, Any]]:
  by = {s.step: s for s in traj.steps}
  lsteps = {s.step: s for s in log.steps}
  parts = [txt(f"Task: {meta.instruction}\n\n# SUBGOALS\n{render_subgoals(sgs, full=False)}\n\n"
               f"Trajectory has {traj.n_steps} steps; this chunk covers screenshots before steps {steps[0]}-{steps[-1]}.")]
  if prev is None:
    parts.append(txt("Status before this chunk: start of the trajectory (nothing done by the agent yet; the initial "
                     "setup may already satisfy some subgoals)."))
  else:
    parts.append(txt(f"Status at the screenshot before step {prev.step} (end of previous chunk): holds={prev.holds} "
                     f"unsure={prev.unsure}"))
  for t in steps:
    st = by.get(t)
    if st and image_path_ok(st.screenshot_path):
      parts.append(txt(f"(screenshot before step {t})"))
      parts.append(marked_image(st.screenshot_path, [], traj.screen, max_width))
    else:
      parts.append(txt(f"(screenshot before step {t} missing)"))
    ls = lsteps.get(t)
    parts.append(txt(f"step {t} executed: {ls.render() if ls else '?'}"))
  ids = ", ".join(s.id for s in sgs)
  parts.append(txt(f"Return one entry per screenshot (steps {steps[0]}-{steps[-1]}); subgoal ids: {ids}. JSON only."))
  return parts


def parse_states(text: str, steps: list[int], valid: set[str]) -> list[StepStatus]:
  obj = parse_json(text, "states") or {}
  got: dict[int, StepStatus] = {}
  for r in obj.get("states") or []:
    try:
      t = int(r.get("step"))
    except (TypeError, ValueError):
      continue
    if t in steps:
      got[t] = StepStatus(step=t, holds=[s for s in map(str, r.get("holds") or []) if s in valid],
                          unsure=[s for s in map(str, r.get("unsure") or []) if s in valid], note=str(r.get("note") or "")[:300])
  return [got[t] for t in steps if t in got]


def screenshot_steps(traj: Trajectory) -> list[int]:
  return [s.step for s in traj.steps if image_path_ok(s.screenshot_path)]


# --- 4. verification on successful runs --------------------------------------------------------


def verify(sset: SubgoalSet, gold_tracks: list[Track], *, seen_keys: set[str] | None = None) -> SubgoalSet:
  """Use successful runs as counterexamples: an essential subgoal a success never reaches is refuted.

  A subgoal that holds anywhere in the success (including only as 'unsure' at the last screenshot) counts as
  reached. Order edges A->B where a success reaches B strictly before A are dropped.
  """
  sset = sset.model_copy(deep=True)
  seen_keys = seen_keys or set()
  for s in sset.subgoals:
    s.gold_hits = {}
  dropped: set[tuple[str, str]] = set()
  for tr in gold_tracks:
    first = tr.first_hold()
    unsure_any = {u for st in tr.statuses for u in st.unsure}
    for s in sset.subgoals:
      s.gold_hits[tr.key] = True if s.id in first else (None if s.id in unsure_any else False)
      for d in s.depends_on:
        if s.id in first and d in first and first[s.id] < first[d]:
          dropped.add((d, s.id))
  for s in sset.subgoals:
    hits = list(s.gold_hits.values())
    if not hits:
      s.verification = "unverified"
    elif any(h is False for h in hits):
      s.verification = "refuted"
    elif all(h is True for h in hits):
      s.verification = "verified"
    else:
      s.verification = "unclear"
    s.depends_on = [d for d in s.depends_on if (d, s.id) not in dropped]
  sset.dropped_edges = sorted([list(e) for e in dropped])
  sset.extra["gold_keys"] = [t.key for t in gold_tracks]
  sset.extra["gold_seen_in_extraction"] = sorted(k for k in seen_keys if k in {t.key for t in gold_tracks})
  return sset


# --- 6. localization ---------------------------------------------------------------------------


def locate_failure(sset: SubgoalSet, track: Track, *, use_verification: bool = True) -> dict[str, Any]:
  """Rule-based error window from tracked subgoal status.

  Returns ``mode`` (never_reached | lost | all_reached | no_subgoals), the failed subgoal, the window
  ``[lo, hi]`` of action steps that contains the error (±3 around the undoing step for ``lost``), and
  ``rule_step`` (the rule's point estimate).
  Screenshot t is the observation before action t, so a state first seen at screenshot t was produced by
  action t-1, and a state lost at screenshot t was destroyed by action t-1.
  """
  order = sset.ordered(essential_only=True, use_verification=use_verification)
  n = track.n_steps
  if not order:
    return {"mode": "no_subgoals", "failed": None, "lo": 1, "hi": n, "rule_step": 1}
  statuses = sorted(track.statuses, key=lambda s: s.step)
  first = track.first_hold()
  reached_at: dict[str, int] = {}
  for s in order:
    if s.id not in first:
      deps = [reached_at[d] for d in s.depends_on if d in reached_at]
      prior = [reached_at[o.id] for o in order[: order.index(s)] if o.id in reached_at]
      start_obs = max(deps or prior or [1])
      lo = max(1, start_obs)  # first action taken once the preceding progress was visible
      return {"mode": "never_reached", "failed": s.id, "lo": lo, "hi": n, "rule_step": lo}
    reached_at[s.id] = first[s.id]
  # every essential subgoal was reached at some point: did a persistent one get undone?
  for s in order:
    if not s.persistent:
      continue
    seen = False
    for st in statuses:
      if s.id in st.holds:
        seen = True
      elif seen and s.id not in st.unsure:
        lost_obs = st.step
        later = [x for x in statuses if x.step > lost_obs]
        if not any(s.id in x.holds for x in later):  # lost for good
          step = max(1, lost_obs - 1)
          return {"mode": "lost", "failed": s.id, "lo": max(1, step - 3), "hi": min(n, step + 3), "rule_step": step}
  last = max(reached_at.values())
  return {"mode": "all_reached", "failed": None, "lo": max(1, last - 1), "hi": n, "rule_step": max(1, last - 1)}


def localize_content(meta: TaskMeta, sset: SubgoalSet, track: Track, traj: Trajectory, log: ActionLog,
                     loc: dict[str, Any], *, image_budget: int, max_width: int) -> list[dict[str, Any]]:
  from goldjudge.judge.images import select_steps_for_images

  lo, hi = loc["lo"], loc["hi"]
  sgs = sset.ordered(essential_only=False, use_verification=False)
  failed = loc.get("failed")
  status_lines = []
  first = track.first_hold()
  for s in sset.ordered(essential_only=True):
    status_lines.append(f"{s.id}: " + (f"first seen at screenshot {first[s.id]}" if s.id in first else "never reached"))
  if loc["mode"] == "never_reached":
    why = f"The agent never reached {failed}. Window: steps {lo}-{hi} (after the last subgoal progress)."
  elif loc["mode"] == "lost":
    why = f"The agent reached {failed} and later undid it around step {loc['rule_step']}. Window: steps {lo}-{hi}."
  else:
    why = (f"Every required subgoal was reached at some point, yet the evaluator failed the run: the error is a "
           f"subtle wrong value/state or a late regression. Window: steps {lo}-{hi}.")
  parts = [txt(f"Task: {meta.instruction}\nEvaluator spec: {trunc(json.dumps(meta.evaluator, ensure_ascii=False), 1200)}\n\n"
               f"# SUBGOALS\n{render_subgoals(sgs, full=False)}\n\n# TRACKER RESULT\n" + "\n".join(status_lines)
               + f"\n{why}\nAgent: {traj.agent}; {traj.n_steps} steps; evaluator score {traj.result}.")]
  by = {s.step: s for s in traj.steps}
  lsteps = {s.step: s for s in log.steps}
  win = list(range(lo, hi + 1))
  rel = select_steps_for_images(len(win), image_budget, must=[1])
  img_steps = {win[i - 1] for i in rel}
  for t in win:
    st, ls = by.get(t), lsteps.get(t)
    if st and t in img_steps and image_path_ok(st.screenshot_path):
      parts.append(txt(f"(screenshot before step {t})"))
      parts.append(marked_image(st.screenshot_path, op_points(ls.ops) if ls else [], traj.screen, max_width))
    line = f"step {t}: {ls.render() if ls else '?'}"
    if ls and ls.effect:
      line += f"  => {ls.effect}"
    if st and st.thought:
      line += f"\n  thought: {trunc(st.thought, 400)}"
    parts.append(txt(line))
  parts.append(txt(f"Give the root-cause step (between {lo} and {hi}). JSON only."))
  return parts


def parse_localization(text: str, lo: int, hi: int) -> dict[str, Any]:
  obj = parse_json(text, "root_step") or {}
  try:
    step = int(obj.get("root_step"))
  except (TypeError, ValueError):
    return {"root_step": None, "parse_ok": False}
  return {"root_step": min(max(step, lo), hi), "raw_root_step": step, "parse_ok": True,
          "failed_subgoal": obj.get("failed_subgoal"), "evidence": str(obj.get("evidence") or "")[:1500],
          "correction": str(obj.get("correction") or "")[:800]}
