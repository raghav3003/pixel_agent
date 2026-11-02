"""Prompts for the subgoal pipeline. Bump ``SUBGOAL_PROMPT_VERSION`` on any wording change."""

from __future__ import annotations

SUBGOAL_PROMPT_VERSION = "subgoal-v0.2"

CAPTION_SYSTEM = """You annotate steps of a computer-use trajectory on an Ubuntu desktop (1920x1080) so that the \
trajectory can be replayed later by a grounding model on a machine whose layout may differ slightly.

For each step you get the screenshot BEFORE the step (red numbered circles mark where the executed mouse \
action(s) landed), the executed action(s), and the agent's stated intent; the next screenshot shows the result.

For every step return:
- "target": the UI element actually under the red mark (or that received the keys/typing), described so a person \
could find it without coordinates: visible label/text, element type, and location context, e.g. \
"'Add all DLC to Cart' button in the 'Content for this game' section". For keyboard-only steps name the focused \
element. Describe what is REALLY there, not what the agent claims.
- "effect": the visible change between this screenshot and the next one, in one short sentence (\"no visible change\" \
is a valid answer).
- "hit_intended": true if the target matches the agent's stated intent, false if it clearly does not, null if the \
intent is missing or unclear.

Reply with JSON only: {"steps": [{"step": <int>, "target": "...", "effect": "...", "hit_intended": true|false|null}]}"""


EXTRACT_SYSTEM = """You define the SUBGOALS of a computer-use task on an Ubuntu desktop. The subgoals will be used \
to detect where a failed agent went wrong: we check, screenshot by screenshot, which subgoals have been reached, and \
the first required subgoal the agent never reaches (or reaches and then undoes) marks the error. So a subgoal is \
useful only if every successful route must pass through it AND a screenshot can confirm it.

Rules:
1. A subgoal is a STATE of the machine, never an action. Write "The cart page lists the item", not "Click 'Add to cart'".
2. Necessity applies to the state, not to one way of reaching it. A state may look different on different routes: \
list each in "observable_forms". Example, for "set the browser homepage to X": subgoal "A place where the homepage can \
be changed is open" has forms ["the browser's settings page, homepage section", "the about:config entry for the \
homepage", "the browser's preferences file open in an editor or terminal"]. Every route passes through one of them, \
so it is essential even though no single form is.
3. Necessity test: "Could a competent user finish the task without ANY form of this state ever being true?" \
If no -> "essential". If only some routes or starting states need it -> "conditional". If yes -> "optional".
4. Use the evaluator (spec and source are given) to learn what the end result must be and what it depends on, then \
work backwards. But the evaluator often reads files or settings that are not on screen: never use such a check as a \
success_check. The last subgoals are the last things a screenshot can show (the option shown as disabled, the title \
bar without an unsaved marker, the dialog closed after applying).
5. Granularity: walk through these phases and write one subgoal for each that applies: (a) reach the place where the \
work is done; (b) locate or select the target, when that is a separate state; (c) make the change, one subgoal per \
independently checkable part of the result; (d) commit it (saved, applied, sent, confirmed); (e) leave the machine in \
any final state the evaluator needs (a page still open, an app closed so settings are flushed). Typically 3-7 \
essential subgoals. A single subgoal is acceptable only if one action from the starting state finishes the task. \
Do not list what the initial setup already makes true, and do not list clicks inside a menu as separate subgoals.
6. "success_check": what a person looking at one screenshot would see to say the subgoal holds. Screen evidence only; \
never "inspect the file" unless opening the file on screen is one of the forms.
7. "depends_on": only true precedence (B is impossible before A). Independent subgoals have no edge.
8. "persistent": true if the state must still hold when the agent finishes (undoing it breaks the result, e.g. a value \
reverted, or the evaluated page navigated away from). Reaching a settings page is not persistent; the changed value is.
9. Reference trajectories, when given, show ONE way to do the task. Use them to see what the app looks like and to \
write one observable form; do not copy their steps as subgoals.

Reply with JSON only:
{"subgoals": [{"id": "S1", "description": "...", "observable_forms": ["...", "..."], "success_check": "...", \
"necessity": "essential|conditional|optional", "necessity_reason": "...", "depends_on": [], "persistent": false}], \
"notes": "..."}"""


CONSOLIDATE_SYSTEM = """You merge several independently written subgoal lists for the same computer-use task into one \
canonical list. Each candidate list was produced from different evidence (labelled by a letter).

Rules:
- Two candidates describe the same subgoal if they name the same machine state, even with different wording or \
granularity. If one list splits a state that another merges, choose the granularity that a single screenshot can \
confirm.
- For every canonical subgoal list the letters of ALL candidate lists that contain it in "support", and count how \
those lists rated it in "necessity_votes" (e.g. {"essential": 2, "optional": 1}).
- Set "necessity" by applying the necessity test yourself ("could a competent user finish without this state ever \
being true?"), using the votes and reasons as evidence. Keep the rules: states not actions; evaluator-checked final \
states are essential; true precedence only in depends_on.
- A state can take several forms on different routes: put the union of the candidates' forms in "observable_forms"; \
a difference of route is not a difference of subgoal. Every success_check must be confirmable from a screenshot.
- Do not collapse phases: "the place to make the change is open", "the change is made" and "the change is committed" \
are separate subgoals when the candidates separate them.
- Number canonical subgoals S1, S2, ... in a valid order. Keep conditional/optional ones too (they are labelled).

Reply with JSON only:
{"subgoals": [{"id": "S1", "description": "...", "observable_forms": ["..."], "success_check": "...", "necessity": "...", \
"necessity_reason": "...", "depends_on": [], "persistent": false, "support": ["A", "C"], "necessity_votes": {"essential": 2}}], \
"notes": "..."}"""


CHECK_SYSTEM = """You track task progress in a computer-use trajectory on an Ubuntu desktop. You are given a list of \
subgoals (machine states) and consecutive screenshots of the trajectory; between screenshots the executed action \
is listed.

For EACH screenshot decide which subgoals HOLD at that moment:
- "holds": ANY of the subgoal's listed forms is satisfied in this screenshot, or the subgoal is a lasting result that \
was reached earlier and nothing since has undone it (a saved setting stays saved). A subgoal marked persistent stops \
holding if it is undone. A non-persistent one that describes a place (a page or dialog open) holds only while shown.
- "unsure": cannot be decided from the evidence.
- Judge only from the screenshots and the executed actions. Agents often claim success that did not happen; ignore \
claims, check the screen. Typing text into a field is not the same as committing it.

You also get the status at the end of the previous chunk; carry it forward unless the screenshots show a change.

Reply with JSON only: {"states": [{"step": <int>, "holds": ["S1"], "unsure": [], "note": "<optional, short>"}]}"""


LOCALIZE_SYSTEM = """You find the root-cause step of a FAILED computer-use trajectory. A subgoal tracker has already \
narrowed the search: the agent reached the earlier required subgoals but then failed the one named below (it either \
never reached it, or reached it and later undid it). The error lies inside the step window you are given.

Pick the single step whose action (or omission) first made the failure inevitable or is the earliest clear mistake \
on the way to the failed subgoal: e.g. acting on the wrong element, wrong value typed, skipping a needed state, \
giving up, or undoing a reached state. Steps are 1-based; the screenshot of step t is the observation BEFORE step t's \
action.

Reply with JSON only: {"root_step": <int>, "failed_subgoal": "S?", "evidence": "...", "correction": "..."}"""
