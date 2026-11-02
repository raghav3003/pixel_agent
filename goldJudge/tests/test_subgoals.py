from goldjudge.subgoals import core
from goldjudge.subgoals.schema import StepStatus, Subgoal, SubgoalSet, Track


def _set(*sgs):
  return SubgoalSet(task_id="t", domain="chrome", instruction="x", version="v", subgoals=list(sgs))


def _track(n, holds_by_step, key="k"):
  return Track(key=key, task_id="t", agent="a", result=0.0, n_steps=n, version="v",
               statuses=[StepStatus(step=t, holds=holds_by_step.get(t, [])) for t in range(1, n + 1)])


S1 = Subgoal(id="S1", description="page open")
S2 = Subgoal(id="S2", description="item in cart", depends_on=["S1"], persistent=True)
S3 = Subgoal(id="S3", description="nice to have", necessity="optional")


def test_never_reached_window_starts_after_last_progress():
  tr = _track(10, {t: ["S1"] for t in range(4, 11)})  # S1 first visible at screenshot 4
  loc = core.locate_failure(_set(S1, S2, S3), tr)
  assert loc["mode"] == "never_reached" and loc["failed"] == "S2" and (loc["lo"], loc["hi"], loc["rule_step"]) == (4, 10, 4)


def test_lost_persistent_subgoal_blames_the_action_before():
  holds = {t: ["S1"] for t in range(2, 11)}
  for t in range(5, 8):
    holds[t] = ["S1", "S2"]  # reached at screenshot 5, gone from 8 on -> action 7 undid it
  loc = core.locate_failure(_set(S1, S2), _track(10, holds))
  assert loc["mode"] == "lost" and loc["failed"] == "S2" and loc["rule_step"] == 7 and (loc["lo"], loc["hi"]) == (4, 10)


def test_all_reached():
  loc = core.locate_failure(_set(S1, S2), _track(6, {t: ["S1", "S2"] for t in range(3, 7)}))
  assert loc["mode"] == "all_reached"


def test_verify_refutes_skipped_subgoal_and_drops_violated_edge():
  a = Subgoal(id="A", description="a")
  b = Subgoal(id="B", description="b", depends_on=["A"])
  c = Subgoal(id="C", description="c")
  gold = _track(5, {2: ["B"], 3: ["A", "B"]}, key="g1")  # reaches B before A, never C
  out = core.verify(_set(a, b, c), [gold])
  v = {s.id: s.verification for s in out.subgoals}
  assert v == {"A": "verified", "B": "verified", "C": "refuted"}
  assert out.subgoals[1].depends_on == [] and out.dropped_edges == [["A", "B"]]
  assert [s.id for s in out.ordered()] == ["A", "B"]  # refuted C excluded
  assert [s.id for s in out.ordered(use_verification=False)] == ["A", "B", "C"]


def test_parse_json_takes_last_matching_object():
  text = ('thinking {"x": 1} ... ```json\n{"subgoals": [{"id": "S1", "description": "d", "necessity": "weird", '
          '"observable_forms": ["a", "b"]}]}\n```')
  sgs, _ = core.parse_subgoals(text)
  assert len(sgs) == 1 and sgs[0].necessity == "essential" and sgs[0].observable_forms == ["a", "b"]
  assert "any of: a | b" in core.render_subgoals(sgs, full=False)
  st = core.parse_states('{"states": [{"step": 3, "holds": ["S1", "S9"]}, {"step": 99, "holds": []}]}', [3, 4], {"S1"})
  assert [(s.step, s.holds) for s in st] == [(3, ["S1"])]
