from goldjudge.subgoals import intents
from goldjudge.subgoals.intents import AgentGoal
from goldjudge.subgoals.schema import StepStatus, Subgoal, SubgoalSet, Track


def _set(*ids):
  return SubgoalSet(task_id="t", domain="d", instruction="x", version="v", subgoals=[Subgoal(id=i, description=i) for i in ids])


def _track(n, holds):
  return Track(key="k", task_id="t", agent="a", result=0.0, n_steps=n, version="v",
               statuses=[StepStatus(step=t, holds=holds.get(t, [])) for t in range(1, n + 1)])


def _goal(id, first, last, **kw):
  return AgentGoal(id=id, description=id, first_step=first, last_step=last, **kw)


def test_parse_goals_clamps_steps_and_skips_bad_rows():
  text = ('{"goals": [{"id": "A1", "description": "d", "first_step": 2, "last_step": 99, "claimed_done": true, "claimed_step": 99},'
          ' {"description": "no start"}, {"description": "e", "first_step": 7}]}')
  g = intents.parse_goals(text, 10)
  assert [(x.first_step, x.last_step) for x in g] == [(2, 10), (7, 7)]
  assert g[0].claimed_done and g[0].claimed_step is None and g[1].id == "A3"


def test_alignment_keeps_goals_and_drops_unknown_values():
  goals = [_goal("A1", 1, 2), _goal("A2", 3, 4)]
  text = '{"goals": [{"id": "A1", "alignment": "REQUIRED", "aligned_to": ["S1", "S9"], "completed": "maybe"}]}'
  out = intents.apply_alignment(goals, text, {"S1"})
  assert out[0].alignment == "required" and out[0].aligned_to == ["S1"] and out[0].completed is None
  assert out[1].alignment is None and goals[0].alignment is None  # input not mutated


def test_classify_outcomes_of_required_subgoals():
  goals = [_goal("A1", 1, 2, aligned_to=["S1"], alignment="required", completed="yes"),
           _goal("A2", 3, 5, aligned_to=["S2"], alignment="required", completed="no", claimed_done=True, claimed_step=5),
           _goal("A3", 6, 7, aligned_to=["S3"], alignment="required", completed="no")]
  out = intents.classify(goals, _set("S1", "S2", "S3", "S4"), _track(8, {3: ["S1"]}))
  assert out["outcomes"] == {"S1": "reached", "S2": "false_completion", "S3": "attempted", "S4": "omitted"}
  assert out["primary"] == "false_completion" and out["first_unreached"] == "S2"
  assert (out["divergence_step"], out["divergence_kind"]) == (5, "false_claim") and out["n_false_claims"] == 1


def test_divergence_is_earliest_event_and_absent_when_on_task():
  goals = [_goal("A1", 2, 4, alignment="contradicts"), _goal("A2", 5, 6, claimed_done=True, claimed_step=6, completed="no")]
  out = intents.classify(goals, _set("S1"), _track(8, {}))
  assert (out["divergence_step"], out["divergence_kind"], out["divergence_goal"]) == (2, "contradicting_goal", "A1")
  on_task = [_goal("A1", 1, 8, alignment="required", aligned_to=["S1"], completed="yes")]
  out = intents.classify(on_task, _set("S1"), _track(8, {4: ["S1"]}))
  assert out["divergence_step"] is None and out["primary"] == "all_reached"


def test_claim_of_a_completed_goal_is_not_false():
  assert not _goal("A", 1, 2, claimed_done=True, completed="yes").false_claim
  assert not _goal("A", 1, 2, claimed_done=True, completed="unclear").false_claim
  assert _goal("A", 1, 2, claimed_done=True, completed="no").false_claim
