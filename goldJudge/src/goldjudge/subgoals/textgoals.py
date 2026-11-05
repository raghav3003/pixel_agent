"""Subgoals from text alone, one method for the agent and the gold (see docs/subgoals.md, section 7).

1. extract  instruction + per-step text -> ordered subgoals. Same prompt for agent and gold, and no screenshots,
            evaluator, captions or outcome: it reports what the text shows the person was trying to do.
2. align    gold subgoals vs agent subgoals -> which gold subgoals the agent finished, by its own text.
3. classify rule: the first unfinished gold subgoal (or an earlier conflicting goal) is the divergence, which
            gives a predicted error step and a candidate failure label.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from goldjudge.actions import ActionLog
from goldjudge.judge.images import select_steps_for_images
from goldjudge.schema import Trajectory
from goldjudge.subgoals.core import marked_image, parse_json, trunc, txt

TEXTGOAL_PROMPT_VERSION = "textgoal-v0.3"

Kind = Literal["find", "do"]
Ended = Literal["says_done", "gave_up", "ran_out", "no_statement"]
Status = Literal["finished", "unfinished", "missing"]
Relation = Literal["serves", "extra", "conflicts"]

EXTRACT_SYSTEM = """You read the text log of someone working on a computer task and write down the subgoals they were \
pursuing, in the order they pursued them. You have only the text: never guess what was on the screen.

Rules:
- State the PURPOSE, not the button or route: "quoting in replies is turned off", not "uncheck the box". The same \
subgoal should read the same if it were reached a different way.
- "kind": "find" if the purpose is to work out where or how something is done, or to locate something (the person is \
searching, exploring menus, unsure of the way); "do" if the purpose is to change or produce something, or to confirm it.
- A subgoal is one stage on the way to the task, never the task restated. When the log goes through several stages, \
such as getting to the right place, locating the target, making each separate change, and saving or applying it, each \
stage is its own subgoal, even if the log is short.
- Merge consecutive steps of one stage; start a new subgoal when the stage changes. Record a stage again if the person \
returns to it after doing something else. Typically 2-8 subgoals; one is right only if the log has a single stage.
- Report what they were trying to do even if it was a bad idea or unrelated to the task. Do not correct or judge it.
- "ended": "says_done" if their text says it was achieved, "gave_up" if their text gives it up, otherwise \
"no_statement". Use only what the text states. Logs without reasoning have no statements, so use "no_statement".
- Some logs contain only actions: infer the purpose from them. Coordinates in actions carry no meaning.

Reply with JSON only:
{"subgoals": [{"id": "1", "purpose": "...", "kind": "find|do", "first_step": 1, "last_step": 3, \
"ended": "says_done|gave_up|no_statement", "evidence": "<short quote from the log>"}]}"""

ALIGN_SYSTEM = """You compare two ordered subgoal lists for the same computer task, both written from text logs. The GOLD \
list comes from a successful way of doing the task. The AGENT list is what another agent was trying to do. You see no \
screenshots: judge only from the lists.

For each GOLD subgoal return "status":
- "finished": an agent subgoal with the same purpose ended "says_done", or the agent went on to a later stage that \
could only start once this one was done (it found the option and then set it). A different route to the same purpose counts.
- "unfinished": the agent worked towards it, but neither of those shows it was achieved.
- "missing": no agent subgoal worked towards it.
and "agent_goals": the ids of the agent subgoals that worked towards it. A "find" subgoal counts as working towards the \
gold subgoal it was searching for, because gold often skips searching when the way is already known.

For each AGENT subgoal return "relation":
- "serves": works towards some gold subgoal; list them in "gold".
- "extra": harmless, not part of the gold list.
- "conflicts": works against the instruction or the gold list (a different target or value than the task asks for, a \
change nobody asked for, undoing a gold subgoal).

Reply with JSON only:
{"gold": [{"id": "1", "status": "finished|unfinished|missing", "agent_goals": ["2"]}], \
"agent": [{"id": "2", "relation": "serves|extra|conflicts", "gold": ["1"]}]}"""


class TextGoal(BaseModel):
  id: str
  purpose: str
  kind: Kind = "do"
  first_step: int
  last_step: int
  ended: Ended = "no_statement"
  evidence: str = ""


class GoldMatch(BaseModel):
  id: str
  status: Status = "missing"
  agent_goals: list[str] = Field(default_factory=list)


class AgentMatch(BaseModel):
  id: str
  relation: Relation = "extra"
  gold: list[str] = Field(default_factory=list)


def _int(v: Any) -> int | None:
  try:
    return int(v)
  except (TypeError, ValueError):
    return None


# --- 1. extract --------------------------------------------------------------------------------


def agent_content(instruction: str, traj: Trajectory, log: ActionLog, *, thought_chars: int = 900) -> list[dict[str, Any]]:
  lsteps = {s.step: s for s in log.steps}
  lines = []
  for s in traj.steps:
    ls = lsteps.get(s.step)
    reasoning = trunc(s.thought, thought_chars) or trunc(s.intent, 200) or "(none logged)"
    lines.append(f"[step {s.step}] reasoning: {reasoning}\n  action: {trunc(ls.render(), 200) if ls else '?'}")
  return [txt(f"Task: {instruction}\n\n# LOG ({traj.n_steps} steps)\n" + "\n".join(lines)), txt("List the subgoals. JSON only.")]


def gold_content(instruction: str, log: ActionLog) -> list[dict[str, Any]]:
  lines = [f"[step {s.step}] reasoning: (none logged)\n  action: {trunc(s.render(), 300)}" for s in log.steps]
  return [txt(f"Task: {instruction}\n\n# LOG ({len(log.steps)} steps)\n" + "\n".join(lines)), txt("List the subgoals. JSON only.")]


def parse_goals(text: str, n_steps: int, *, is_agent: bool = True) -> list[TextGoal]:
  """Reply -> subgoals with steps clamped to the log. An agent's last open subgoal at the final step ``ran_out``."""
  obj = parse_json(text, "subgoals") or {}
  out = []
  for i, r in enumerate(obj.get("subgoals") or [], 1):
    if not isinstance(r, dict):
      continue
    first, last = _int(r.get("first_step")), _int(r.get("last_step"))
    if first is None:
      continue
    last = first if last is None else last
    first, last = max(1, min(first, n_steps)), max(1, min(last, n_steps))
    kind, ended = str(r.get("kind") or "").lower(), str(r.get("ended") or "").lower()
    out.append(TextGoal(id=str(r.get("id") or i), purpose=str(r.get("purpose") or ""),
                        kind=kind if kind in {"find", "do"} else "do", first_step=first, last_step=max(first, last),
                        ended=ended if ended in {"says_done", "gave_up"} else "no_statement",
                        evidence=str(r.get("evidence") or "")[:300]))
  if is_agent and out and out[-1].ended == "no_statement" and out[-1].last_step >= n_steps:
    out[-1].ended = "ran_out"
  return out


# --- 2. align ----------------------------------------------------------------------------------


def render_goals(goals: list[TextGoal]) -> str:
  return "\n".join(f"{g.id} [{g.kind}, steps {g.first_step}-{g.last_step}, ended: {g.ended}] {g.purpose}" for g in goals)


def align_content(instruction: str, gold: list[TextGoal], agent: list[TextGoal]) -> list[dict[str, Any]]:
  return [txt(f"Task: {instruction}\n\n# GOLD SUBGOALS\n{render_goals(gold)}\n\n# AGENT SUBGOALS\n{render_goals(agent)}"),
          txt("Compare the lists. JSON only.")]


def parse_alignment(text: str, gold: list[TextGoal], agent: list[TextGoal]) -> tuple[list[GoldMatch], list[AgentMatch]] | None:
  """Every listed subgoal gets a row, in list order; None if the reply has no gold rows at all."""
  obj = parse_json(text, "agent")  # not "gold": the agent rows also carry a "gold" key
  if not obj or not isinstance(obj.get("gold"), list):
    return None
  gids, aids = {g.id for g in gold}, {a.id for a in agent}
  grows = {str(r.get("id")): r for r in obj["gold"] if isinstance(r, dict)}
  arows = {str(r.get("id")): r for r in obj.get("agent") or [] if isinstance(r, dict)}
  gm, am = [], []
  for g in gold:
    r = grows.get(g.id, {})
    st = str(r.get("status") or "").lower()
    gm.append(GoldMatch(id=g.id, status=st if st in {"finished", "unfinished", "missing"} else "missing",
                        agent_goals=[x for x in map(str, r.get("agent_goals") or []) if x in aids]))
  for a in agent:
    r = arows.get(a.id, {})
    rel = str(r.get("relation") or "").lower()
    am.append(AgentMatch(id=a.id, relation=rel if rel in {"serves", "extra", "conflicts"} else "extra",
                         gold=[x for x in map(str, r.get("gold") or []) if x in gids]))
  return gm, am


# --- 3. rule -----------------------------------------------------------------------------------


def classify(agent: list[TextGoal], gm: list[GoldMatch], am: list[AgentMatch], n_steps: int) -> dict[str, Any]:
  """First divergence of the agent from the gold list, from text only.

  ``knowledge``: the agent only ever searched for the first unfinished gold subgoal and never says it found the way (a
  candidate label: only the screen can tell it from missing something that was visible). ``attempted``: got past
  searching or tried it directly, with no sign of success.
  ``planning``: never worked on it. ``conflict``: an earlier goal works against the task. ``unexplained``: the text
  says every gold subgoal was finished, so text cannot explain the failure and ``step`` is None.
  """
  by = {a.id: a for a in agent}
  label, step, hi, gold_id, goal_id = "unexplained", None, None, None, None
  finished_until = 0
  for m in gm:
    serving = sorted((by[i] for i in m.agent_goals if i in by), key=lambda g: g.first_step)
    if m.status == "finished":
      finished_until = max([finished_until] + [g.last_step for g in serving])
      continue
    gold_id = m.id
    if m.status == "unfinished" and serving:
      finds = [g for g in serving if g.kind == "find"]
      if finds and len(finds) == len(serving) and not any(g.ended == "says_done" for g in finds):
        label, first = "knowledge", finds[0]
        hi = finds[-1].last_step
      else:
        label = "attempted"
        first = next((g for g in serving if g.kind == "do"), serving[0])
        hi = serving[-1].last_step
      step, goal_id = first.first_step, first.id
    else:  # the agent moved on without it: the error is the step where it should have started
      label, step, hi = "planning", min(finished_until + 1, n_steps), n_steps
    break
  conflicts = sorted((by[a.id] for a in am if a.relation == "conflicts" and a.id in by), key=lambda g: g.first_step)
  if conflicts and (step is None or conflicts[0].first_step < step):
    c = conflicts[0]
    label, step, hi, goal_id = "conflict", c.first_step, c.last_step, c.id
    gold_id = None
  statuses = [m.status for m in gm]
  return {"label": label, "step": step, "hi": hi, "gold_subgoal": gold_id, "agent_goal": goal_id,
          "n_gold": len(gm), "n_agent": len(agent), "n_finished": statuses.count("finished"),
          "n_unfinished": statuses.count("unfinished"), "n_missing": statuses.count("missing"),
          "n_conflicts": len(conflicts), "n_extra": sum(a.relation == "extra" for a in am),
          "n_find": sum(a.kind == "find" for a in agent)}


# --- 4. screen check at the divergence -----------------------------------------------------------

SCREEN_PROMPT_VERSION = "textgoal-screen-v0.2"

CLAIMS_SYSTEM = """You check claims made by a computer-use agent against the screen. For each claim you get what the \
agent was trying to do, its actions for that stage, and the screenshot taken just after the stage (the observation \
before the next step). The agent believes the stage succeeded.

For each claim answer "happened": "yes" if the screenshot shows the stage's result, "no" if it shows the result is \
absent or wrong (a different value, the wrong item, an error, nothing changed), "unclear" if the screenshot cannot \
show it. Judge only from the screenshot; the agent's belief is not evidence.

Reply with JSON only: {"claims": [{"id": "<agent subgoal id>", "happened": "yes|no|unclear", "evidence": "..."}]}"""

SEARCH_SYSTEM = """A computer-use agent spent several steps searching for how or where to do something. You get what \
it was looking for, how a human did that part of the task (one valid way), screenshots from the agent's search, and \
the agent's action at every step of the search and just after it.

1. "visible_step": the first step whose screenshot clearly shows the thing it needed (the option, control, menu entry, \
field or information that would let it do this), or null if none does. Visible means a person looking at that \
screenshot could have seen and used it, not that it was reachable through some menu.
2. "used_when_visible": if it was visible, did the agent act on it at that step or the next one or two (click it, open \
it, read and use it)? true if so; false if the agent went elsewhere and kept searching. null if it was never visible.

Reply with JSON only: {"visible_step": <int or null>, "used_when_visible": true|false|null, "what": "<the control \
you saw, or what was missing>"}"""


class ClaimCheck(BaseModel):
  id: str
  happened: Literal["yes", "no", "unclear"] = "unclear"
  evidence: str = ""
  screenshot_step: int | None = None


class SearchCheck(BaseModel):
  id: str
  first_step: int
  last_step: int
  visible_step: int | None = None
  used_when_visible: bool | None = None
  what: str = ""
  shown_steps: list[int] = Field(default_factory=list)


def claims_to_check(agent: list[TextGoal], am: list[AgentMatch], budget: int) -> list[TextGoal]:
  """Claimed-done subgoals that serve the gold list, in order (evenly thinned past the budget)."""
  serving = {a.id for a in am if a.relation == "serves"}
  claimed = [g for g in agent if g.ended == "says_done" and g.id in serving]
  if len(claimed) <= budget:
    return claimed
  idx = sorted({round(i * (len(claimed) - 1) / (budget - 1)) for i in range(budget)}) if budget > 1 else [len(claimed) - 1]
  return [claimed[i] for i in idx]


def searches_to_check(agent: list[TextGoal], am: list[AgentMatch], *, min_steps: int, max_n: int) -> list[TextGoal]:
  """The longest find subgoals that serve the gold list."""
  serving = {a.id for a in am if a.relation == "serves"}
  finds = [g for g in agent if g.kind == "find" and g.id in serving and g.last_step - g.first_step + 1 >= min_steps]
  return sorted(sorted(finds, key=lambda g: g.first_step - g.last_step)[:max_n], key=lambda g: g.first_step)


def after_step(g: TextGoal, have: set[int]) -> int | None:
  """Screenshot showing the result of a subgoal: the one after its last step, else its last step's own."""
  return next((t for t in (g.last_step + 1, g.last_step) if t in have), None)


def parse_claims(text: str, goals: list[TextGoal]) -> list[ClaimCheck]:
  obj = parse_json(text, "claims") or {}
  rows = {str(r.get("id")): r for r in obj.get("claims") or [] if isinstance(r, dict)}
  out = []
  for g in goals:
    r = rows.get(g.id, {})
    h = str(r.get("happened") or "").lower()
    out.append(ClaimCheck(id=g.id, happened=h if h in {"yes", "no", "unclear"} else "unclear", evidence=str(r.get("evidence") or "")[:400]))
  return out


def parse_search(text: str, g: TextGoal, shown: list[int]) -> SearchCheck:
  obj = parse_json(text, "visible_step") or {}
  v = _int(obj.get("visible_step"))
  return SearchCheck(id=g.id, first_step=g.first_step, last_step=g.last_step, shown_steps=shown,
                     visible_step=v if v is not None and g.first_step <= v <= g.last_step else None,
                     used_when_visible=obj.get("used_when_visible") if isinstance(obj.get("used_when_visible"), bool) else None,
                     what=str(obj.get("what") or "")[:300])


def classify_with_screen(text_summary: dict[str, Any], agent: list[TextGoal], claims: list[ClaimCheck],
                         searches: list[SearchCheck]) -> dict[str, Any]:
  """Text divergence plus two screen events; the earliest wins.

  ``false_claim``: the agent says a stage it needed succeeded and the screen shows it did not (error in that stage).
  ``perception``: during a search the needed control was visible and the agent did not use it but kept searching.
  A text ``knowledge`` candidate whose target was never visible stays ``knowledge``.
  """
  by = {g.id: g for g in agent}
  seen = {s.id: s for s in searches if s.visible_step is not None and s.used_when_visible is False}
  events: list[tuple[int, str, int | None, str]] = []
  if text_summary.get("step") is not None:
    if text_summary["label"] == "knowledge" and text_summary.get("agent_goal") in seen:
      pass  # the searched-for control was on screen: replaced by the perception event below
    else:
      events.append((int(text_summary["step"]), text_summary["label"], text_summary.get("hi"), "text"))
  for c in claims:
    if c.happened == "no" and c.id in by:
      g = by[c.id]
      events.append((g.first_step, "false_claim", g.last_step, c.id))
  for s in searches:
    if s.id in seen:
      events.append((s.visible_step, "perception", s.last_step, s.id))
  events.sort(key=lambda e: e[0])
  label, step, hi, source = (events[0][1], events[0][0], events[0][2], events[0][3]) if events else ("unexplained", None, None, None)
  return {"label": label, "step": step, "hi": hi, "source": source, "text_label": text_summary.get("label"),
          "n_claims": len(claims), "n_false_claims": sum(c.happened == "no" for c in claims),
          "n_unclear_claims": sum(c.happened == "unclear" for c in claims), "n_searches": len(searches),
          "n_visible_searches": sum(s.visible_step is not None for s in searches), "n_missed_searches": len(seen)}


def claims_content(instruction: str, traj: Trajectory, log: ActionLog, goals: list[TextGoal], *,
                   max_width: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
  by, lsteps = {s.step: s for s in traj.steps}, {s.step: s for s in log.steps}
  have = {t for t, s in by.items() if s.screenshot_path}
  parts, shots = [txt(f"Task: {instruction}\nAgent: {traj.agent}; {traj.n_steps} steps.\n# CLAIMS")], {}
  for g in goals:
    acts = " ; ".join(trunc(lsteps[t].render(), 80) for t in range(g.first_step, g.last_step + 1) if t in lsteps)
    parts.append(txt(f"{g.id} (steps {g.first_step}-{g.last_step}): {g.purpose}\n  actions: {trunc(acts, 600)}"))
    t = after_step(g, have)
    if t is not None:
      shots[g.id] = t
      parts.append(txt(f"(screenshot before step {t})" if t > g.last_step else f"(screenshot before step {t}, the stage's last step; no later one)"))
      parts.append(marked_image(by[t].screenshot_path, [], traj.screen, max_width))
  parts.append(txt("Check every claim. JSON only."))
  return parts, shots


def search_content(instruction: str, traj: Trajectory, log: ActionLog, g: TextGoal, gold_purpose: str,
                   human_steps: list[str], *, image_budget: int, max_width: int) -> tuple[list[dict[str, Any]], list[int]]:
  by, lsteps = {s.step: s for s in traj.steps}, {s.step: s for s in log.steps}
  span = [t for t in range(g.first_step, g.last_step + 1) if t in by and by[t].screenshot_path]
  shown = [span[i - 1] for i in select_steps_for_images(len(span), image_budget)]
  parts = [txt(f"Task: {instruction}\nThe agent was looking for: {g.purpose} (steps {g.first_step}-{g.last_step}).\n"
               f"That serves this part of the task: {gold_purpose}\nHow a human did that part:\n"
               + ("\n".join(f"- {h}" for h in human_steps) or "- (not recorded)"))]
  for t in shown:
    parts.append(txt(f"(screenshot before step {t}; agent thought: {trunc(by[t].thought, 200) or '-'})"))
    parts.append(marked_image(by[t].screenshot_path, [], traj.screen, max_width))
  acts = [f"step {t}: {trunc(lsteps[t].render(), 120)}" for t in range(g.first_step, min(g.last_step + 2, traj.n_steps) + 1) if t in lsteps]
  parts.append(txt("# AGENT ACTIONS\n" + "\n".join(acts)))
  parts.append(txt("Was the needed control visible, and did the agent use it? JSON only."))
  return parts, shown
