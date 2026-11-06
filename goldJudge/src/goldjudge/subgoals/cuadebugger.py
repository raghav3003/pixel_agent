"""Reproduction of CUADebug's own RCA harnesses (arXiv 2608.02643, Appendix A.1), to test label anchoring.

CUADebug annotators could see a machine RCA proposal while labelling (``chosen_debugger`` in human_annotation.json:
54 of the 204 failures, essentially the Gemini and Qwen runs). If labels lean towards that procedure, a faithful
reproduction should agree with them more than our judge does, especially on the seeded cases.

* ``tool``    the tool-augmented debugger: ReAct loop with get_step_details(step_num) (action code, reasoning,
              reward/done, the screen before and after the step) and finish(...); walks backward from the terminal
              failure step F. Prompts follow Appendix A.1.
* ``prompt``  the prompt-only baseline: one call with the task, the full textual trajectory, the taxonomy and the
              result screenshots of the last steps.

Deviations: subtype names for R/S come from our taxonomy notes (the PDF table did not extract cleanly), there is no
episodic memory (the paper's Table 2 rows are reported without it as the main setting, unverified), the execution-error
field is not available in our loader, and older screenshots are not removed from the context.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from goldjudge.actions import ActionLog
from goldjudge.judge.images import encode_screenshot
from goldjudge.schema import Trajectory
from goldjudge.subgoals.core import marked_image, trunc, txt

CUADEBUGGER_PROMPT_VERSION = "cuadebugger-repro-v0.1"

TAXONOMY = """P (Perception)
P1 Visual hallucination: perceives objects, text, or UI elements that are not present.
P2 Misrecognition / OCR error: relevant content is visible but identified or parsed incorrectly.
P3 Cross-modal misbinding: incorrectly associates information across modalities, screen regions, or UI elements.
P4 Observation omission: fails to notice necessary visible information.
P5 Semantic misunderstanding: sees the content correctly but misinterprets its meaning for the task.
G (Grounding & Interaction)
G1 Coordinate / element grounding error: targets the wrong coordinates, UI element, DOM node, or spatial region.
G2 Visibility / accessibility error: the intended element is off-screen, occluded, hidden, disabled, or not interactable.
G3 Interaction mechanics error: wrong click type, drag behavior, gesture, text-entry method, or input sequence.
G4 Distraction / adversarial misdirection: redirected by ads, overlays, pop-ups, decoys, or other distractors.
R (Task Reasoning & Control)
R1 Constraint violation. R2 Impossible plan / action. R3 Decomposition failure. R4 Inefficient / redundant actions.
R5 Action-intent misalignment. R6 Invalid / malformed action. R7 Parameter / argument error. R8 Context loss.
R9 Memory hallucination. R10 Progress misjudgment. R11 Outcome misinterpretation. R12 Failed self-correction.
R13 Causal misattribution.
S (External / System)
S1 Rendering / layout. S2 Timing / race. S3 Unexpected system behavior. S4 Step / resource limit.
S5 Tool / API failure. S6 Environment instability. S7 Benchmark / evaluation artifact.
O (Others)
O1 Infeasible task."""

ANNOTATION_RULES = """<annotation_rules>
- Label the root-cause error, even when downstream effects manifest differently.
- Perception vs. Grounding: misunderstanding visible content is P; understanding the content but targeting the wrong
location or interacting incorrectly is G.
- Prefer fine-grained subtype codes (P1, G2, R10, S4, etc.).
</annotation_rules>"""

TOOL_SYSTEM = f"""<role>
You are an expert GUI agent trajectory debugger. Use the provided tools to analyze a trajectory step by step, then
submit your findings.
</role>
<workflow>
1. The user first message includes a step index showing each step's action_type and whether a screenshot exists.
Use it to pick which steps to inspect.
2. For each suspicious step, such as a step after a failed step, a repeated action, or the terminal step of a failed
trajectory, call get_step_details.
3. get_step_details returns the input screenshot (what the agent saw before acting) and the result screenshot (screen
state after the action). Compare them to identify perception, grounding, interaction, or outcome errors.
4. Look for patterns across steps: repeated errors without correction, stale variable references, wrong UI targets, or
an environment that was not ready.
5. When you have inspected all relevant steps, call finish() with the complete structured RCA record.
</workflow>
<taxonomy>
{TAXONOMY}
</taxonomy>
{ANNOTATION_RULES}
Your task is Root Cause Analysis, not general debugging.
Goal: identify the single earliest step N such that N <= F, where F is the Terminal Failure Step (the last step of a
failed trajectory), and the mistake at step N is the direct cause of subsequent failures.
Backward-tracing procedure:
1. Use get_step_details to inspect steps. It returns textual details plus the input screenshot and result screenshot.
2. Identify F = the terminal failure step from the trajectory summary.
3. Walk backwards from F. For each candidate step, call get_step_details to check whether it contains an independent
mistake or merely inherited the failure from an earlier step. Compare the input screenshot against the action to
identify perception or grounding errors.
4. Stop when you find the earliest step that introduced a new mistake. That is step N = root_error_step.
5. Call finish() with root_error_step=N, taxonomy_tag, evidence, correction, confidence.
After each get_step_details result, write a short plain-text observation before requesting more tools.
Confidence guidance:
- 0.9-1.0: clear single root cause with a strong causal chain.
- 0.7-0.89: likely root cause with some ambiguity.
- 0.5-0.69: multiple plausible candidates; this is the best guess.
- < 0.5: weak evidence; still call finish and note the uncertainty."""

PROMPT_SYSTEM = f"""You are a debugging assistant analysing a failed GUI-agent trajectory.
Your job is to identify:
(a) the root error step: the single step number where the failure originates, not just where it surfaces;
(b) the taxonomy_tag: the subtype code from the taxonomy table that best describes the root cause.
Error Taxonomy:
{TAXONOMY}
{ANNOTATION_RULES}
## RCA Mode: Find the Single Root Error Step
Return ONLY a single valid ASCII JSON object with no markdown fences, no prose, and no trailing text:
{{"root_error_step": [int], "taxonomy_tag": [str; one subtype code], "evidence": [str; 1-2 sentences citing
action/error/state evidence], "correction": [str; what the agent should have done at that step],
"confidence": [float between 0 and 1]}}"""

TAGS = ["P1", "P2", "P3", "P4", "P5", "G1", "G2", "G3", "G4", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9",
        "R10", "R11", "R12", "R13", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "O1", "P", "G", "R", "S", "O"]


def step_index(traj: Trajectory) -> str:
  return "\n".join(f"step {s.step}: action_type={s.action_type}, screenshot={'yes' if s.screen_after else 'no'}"
                   + (", done=true" if s.done else "") for s in traj.steps)


def tool_user(instruction: str, traj: Trajectory) -> list[dict[str, Any]]:
  return [txt("Perform Root Cause Analysis on this failed trajectory:\n"
              f"Task: {instruction}\nApp/domain: {traj.domain}\nAgent: {traj.agent}\nTerminal status: failed "
              f"(evaluator score {traj.result})\nTerminal Failure Step F = {traj.n_steps}\n# STEP INDEX\n{step_index(traj)}\n\n"
              "The trajectory is already loaded. You can directly call get_step_details to inspect steps. Work backwards "
              "from the Terminal Failure Step, then call finish() with the root_error_step, taxonomy_tag, evidence, "
              "correction, and confidence.")]


def prompt_user(instruction: str, traj: Trajectory, log: ActionLog, *, last_n: int, max_width: int) -> list[dict[str, Any]]:
  lsteps = {s.step: s for s in log.steps}
  lines = []
  for s in traj.steps:
    lines.append(f"Step {s.step}: action_type={s.action_type}; action code: {trunc(s.action_raw, 300)}; "
                 f"reward={s.reward}; done={s.done}\n  reasoning: {trunc(s.thought, 800) or '-'}")
  parts = [txt(f"{instruction}\n\n# TRAJECTORY ({traj.n_steps} steps)\n" + "\n".join(lines))]
  for s in traj.steps[-last_n:]:
    if s.screen_after:
      parts.append(txt(f"(result screenshot of step {s.step})"))
      parts.append(marked_image(s.screen_after, [], traj.screen, max_width))
  parts.append(txt("Identify the root error step and taxonomy tag. Output JSON only."))
  return parts


@dataclass
class DebuggerTools:
  traj: Trajectory
  max_width: int = 1024
  inspected: list[int] = field(default_factory=list)
  finished: dict[str, Any] | None = None

  def definitions(self) -> list[dict[str, Any]]:
    return [
      {"name": "get_step_details", "description": (
        "Get full details for a specific step: action code, agent reasoning, reward, done status, plus two screenshots: "
        "the input screenshot before the action and the result screenshot after the action."),
       "input_schema": {"type": "object", "properties": {"step_num": {"type": "integer", "description": "The step number to inspect."}},
                        "required": ["step_num"]}},
      {"name": "finish", "description": "Submit the final RCA result after identifying the root error step.",
       "input_schema": {"type": "object", "properties": {
         "root_error_step": {"type": "integer", "description": "Earliest root-cause step N, where N <= terminal step F."},
         "taxonomy_tag": {"type": "string", "enum": TAGS, "description": "Prefer a fine-grained subtype code."},
         "evidence": {"type": "string", "description": "Grounded evidence from action code, errors, or screenshots."},
         "correction": {"type": "string", "description": "What the agent should have done at that step."},
         "confidence": {"type": "number", "description": "Confidence in [0, 1]."}},
         "required": ["root_error_step", "taxonomy_tag", "evidence", "correction", "confidence"]}},
    ]

  def _img(self, path: str, caption: str) -> list[dict[str, Any]]:
    data = encode_screenshot(path, max_width=self.max_width).split(",", 1)[1]
    return [{"type": "text", "text": caption}, {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}]

  def __call__(self, name: str, args: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    if name == "finish":
      self.finished = args
      return [{"type": "text", "text": "RCA recorded. Reply with the same RCA as a JSON object and stop."}], False
    if name != "get_step_details":
      return [{"type": "text", "text": f"unknown tool {name}"}], True
    try:
      n = int(args.get("step_num"))
    except (TypeError, ValueError):
      return [{"type": "text", "text": "step_num must be an integer"}], True
    s = next((x for x in self.traj.steps if x.step == n), None)
    if s is None:
      return [{"type": "text", "text": f"step {n} does not exist (1..{self.traj.n_steps})"}], True
    self.inspected.append(n)
    blocks = [{"type": "text", "text": f"Step {n} details:\naction_type: {s.action_type}\naction code: {trunc(s.action_raw, 600)}\n"
                                       f"reward: {s.reward}; done: {s.done}\nreasoning: {trunc(s.thought, 1500) or '-'}"}]
    if s.screen_before:
      blocks += self._img(s.screen_before, f"(input screenshot: before step {n})")
    else:
      blocks.append({"type": "text", "text": "(no input screenshot logged for this step)"})
    if s.screen_after:
      blocks += self._img(s.screen_after, f"(result screenshot: after step {n})")
    return blocks, False


def final_step(tools: DebuggerTools, text: str) -> tuple[int | None, dict[str, Any]]:
  """Root step from the finish() call, else from a JSON reply."""
  from goldjudge.subgoals.core import parse_json

  import re

  rec = tools.finished or parse_json(text, "root_error_step") or {}
  try:
    return int(rec.get("root_error_step")), rec
  except (TypeError, ValueError):
    m = re.search(r"\bstep\s*(\d+)", text or "", re.I)  # plain-text answer when finish() was not called
    if m:
      return int(m.group(1)), {**rec, "evidence": (text or "")[:800], "from_text": True}
    return None, rec


def dumps(rec: dict[str, Any]) -> str:
  return json.dumps(rec, ensure_ascii=False)[:3000]
