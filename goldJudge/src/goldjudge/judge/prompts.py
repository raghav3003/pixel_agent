"""System prompt for the failure-analysis judge (taxonomy v2)."""

from __future__ import annotations

from goldjudge.taxonomy import Taxonomy

PROMPT_VERSION = "judge-v0.3"  # v0.3: taxonomy v2.1 (G1+G2 merged, G leaves renumbered)

OUTPUT_SCHEMA = """{
  "root_step": <int, 1-based step index>,
  "primary": "<leaf id, e.g. G2>",
  "secondary": [{"leaf": "<leaf id>", "step": <int>, "relation": "co-occurring|downstream|independent"}],
  "evidence": "<2-5 sentences citing specific steps, screenshots, thoughts and actions of the AGENT's own trajectory>",
  "correction": "<what the agent should have done at root_step>",
  "confidence": "high|mid|low",
  "flags": {"evaluator_doubt": <bool>}
}"""


def screenshot_rules(*, budget: int, per_call: int, has_gold: bool) -> str:
  """System-prompt section for the on-demand screenshot harness (empty for the fixed harness)."""
  which = "the agent trajectory" + (" or the reference run" if has_gold else "")
  return f"""
# Screenshots on demand
Only some screenshots are included up front; the message says which steps have one. Call
view_screenshots to see the screenshot before any other step of {which} (at most {per_call} steps
per call, {budget} extra screenshots in total). Use it whenever your judgement depends on what was on
screen, in particular to check the step you suspect is the root step and its neighbours before
committing. Do not request screenshots you do not need. When you are done, reply with ONLY the JSON verdict.
"""


def build_system_prompt(tax: Taxonomy, *, has_reference: bool, screenshots: str = "") -> str:
  ref_rules = (
    "ADDITIONAL INFORMATION: a reference for the same task is provided (a human-written step list or a "
    "successful run by another agent). It is GUIDANCE ONLY. Use it to understand what the task actually "
    "requires (which state must be reached, which settings/fields/options matter, what a working approach "
    "looks like). The agent may legitimately take a completely different route; a different route is NOT "
    "an error, and you must NOT compare the two trajectories step by step or report where they diverge. "
    "Locate the agent's error from the agent's OWN trajectory (its screenshots, reasoning and actions); "
    "the reference only helps you judge whether an agent action or belief was wrong for this task."
    if has_reference
    else "No reference is provided; reason from the task, the evaluator semantics, and the trajectory alone."
  )
  return f"""You are an expert analyst of computer-use agent (CUA) failures. You will receive an OSWorld task,
what its evaluator checks, and a FAILED agent trajectory (screenshots, reasoning, executed actions).
Your job: find the ROOT CAUSE of the failure and label it with the taxonomy below.

# Root step
The root step is the EARLIEST step at which the agent's executed action, given what was observable,
takes the run off any viable path to success AND the deviation is never repaired later. Earlier
suboptimal-but-recovered steps are not the root. If the environment, harness, or evaluator (not the
agent) caused the failure, the root step is the first step where that fault bit, and the primary leaf
is in family E. Downstream consequences (looping, premature DONE) are NEVER the primary.

# Labelling procedure at the root step — ask in this order
E: was something outside the agent at fault (needs screenshot/log evidence; burden of proof is on E)?
P: did the agent mis-see the observation?
G: did its reasoning name the CORRECT target/effect, but the executed action missed it (wrong coordinates,
   wrong element hit, inaccessible target, wrong click/key mechanics)? G applies ONLY when the reasoning is
   right and the execution is wrong.
R: otherwise the error is in what it decided (goal, plan, chosen target, action content, belief, stopping,
   recovery). If the reasoning itself names the wrong element or the wrong approach, that is R (or P if the
   wrong element came from misreading the screen), never G.

Exactly ONE primary leaf at the root step. Add secondary leaves only when supported by evidence:
"co-occurring" (also true at the root step), "downstream" (caused by the root error, later step),
"independent" (a separate later error that would have failed the task on its own).

# Reference (guidance only)
{ref_rules}
{screenshots}
# Taxonomy
{tax.render_for_prompt(with_rules=True)}

# Output
Respond with ONLY a JSON object matching this schema (no prose before or after):
{OUTPUT_SCHEMA}
"""
