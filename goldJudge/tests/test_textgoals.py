from goldjudge.subgoals import textgoals as tg
from goldjudge.subgoals.textgoals import AgentMatch, GoldMatch, TextGoal


def _g(id, first, last, kind="do", ended="no_statement"):
  return TextGoal(id=id, purpose=id, kind=kind, first_step=first, last_step=last, ended=ended)


def _rel(*ids, relation="serves"):
  return [AgentMatch(id=i, relation=relation) for i in ids]


def test_parse_goals_clamps_and_marks_ran_out_for_agents_only():
  text = ('{"subgoals": [{"id": "1", "purpose": "a", "kind": "FIND", "first_step": 2, "last_step": 3, "ended": "says_done"},'
          ' {"purpose": "no start"}, {"id": "2", "purpose": "b", "kind": "weird", "first_step": 4, "last_step": 99}]}')
  g = tg.parse_goals(text, 10)
  assert [(x.kind, x.first_step, x.last_step, x.ended) for x in g] == [("find", 2, 3, "says_done"), ("do", 4, 10, "ran_out")]
  assert tg.parse_goals(text, 10, is_agent=False)[1].ended == "no_statement"


def test_alignment_gives_every_subgoal_a_row_and_drops_unknown_ids():
  gold, agent = [_g("1", 1, 1), _g("2", 2, 2)], [_g("a", 1, 3)]
  text = '{"gold": [{"id": "1", "status": "FINISHED", "agent_goals": ["a", "zz"]}], "agent": [{"id": "a", "relation": "serves", "gold": ["1", "9"]}]}'
  gm, am = tg.parse_alignment(text, gold, agent)
  assert [(m.id, m.status, m.agent_goals) for m in gm] == [("1", "finished", ["a"]), ("2", "missing", [])]
  assert am[0].gold == ["1"]
  assert tg.parse_alignment("no json", gold, agent) is None


def test_find_goal_that_never_succeeds_is_a_knowledge_candidate():
  agent = [_g("a", 1, 2, ended="says_done"), _g("b", 3, 9, kind="find", ended="gave_up")]
  gm = [GoldMatch(id="1", status="finished", agent_goals=["a"]), GoldMatch(id="2", status="unfinished", agent_goals=["b"])]
  out = tg.classify(agent, gm, _rel("a", "b"), 12)
  assert (out["label"], out["step"], out["hi"], out["gold_subgoal"]) == ("knowledge", 3, 9, "2")


def test_searching_then_acting_without_a_statement_is_attempted_not_knowledge():
  agent = [_g("f", 1, 3, kind="find"), _g("d", 4, 8)]
  out = tg.classify(agent, [GoldMatch(id="1", status="unfinished", agent_goals=["f", "d"])], _rel("f", "d"), 8)
  assert (out["label"], out["step"]) == ("attempted", 4)


def test_found_the_way_but_not_done_is_attempted():
  agent = [_g("f", 1, 3, kind="find", ended="says_done"), _g("d", 4, 8)]
  gm = [GoldMatch(id="1", status="unfinished", agent_goals=["f", "d"])]
  out = tg.classify(agent, gm, _rel("f", "d"), 8)
  assert (out["label"], out["step"]) == ("attempted", 4)


def test_missing_subgoal_is_planning_at_the_step_after_last_progress():
  agent = [_g("a", 1, 5, ended="says_done"), _g("x", 6, 9)]
  gm = [GoldMatch(id="1", status="finished", agent_goals=["a"]), GoldMatch(id="2", status="missing")]
  out = tg.classify(agent, gm, [AgentMatch(id="a", relation="serves"), AgentMatch(id="x", relation="extra")], 9)
  assert (out["label"], out["step"], out["n_missing"], out["n_extra"]) == ("planning", 6, 1, 1)
  # nothing finished: the error is at the start
  assert tg.classify([], [GoldMatch(id="1", status="missing")], [], 9)["step"] == 1


def test_earlier_conflict_wins_and_later_conflict_does_not():
  agent = [_g("c", 2, 4), _g("d", 5, 9)]
  gm = [GoldMatch(id="1", status="unfinished", agent_goals=["d"])]
  out = tg.classify(agent, gm, [AgentMatch(id="c", relation="conflicts"), AgentMatch(id="d", relation="serves")], 9)
  assert (out["label"], out["step"], out["gold_subgoal"]) == ("conflict", 2, None)
  late = [_g("d", 1, 4), _g("c", 5, 9)]
  out = tg.classify(late, gm, [AgentMatch(id="c", relation="conflicts"), AgentMatch(id="d", relation="serves")], 9)
  assert (out["label"], out["step"]) == ("attempted", 1)


def test_all_finished_is_unexplained_unless_there_is_a_conflict():
  agent = [_g("a", 1, 5, ended="says_done")]
  gm = [GoldMatch(id="1", status="finished", agent_goals=["a"])]
  out = tg.classify(agent, gm, _rel("a"), 5)
  assert (out["label"], out["step"]) == ("unexplained", None)
  assert tg.classify(agent, gm, _rel("a", relation="conflicts"), 5)["label"] == "conflict"


def test_screen_events_false_claim_and_perception():
  from goldjudge.subgoals.textgoals import ClaimCheck, SearchCheck
  agent = [_g("f", 1, 8, kind="find"), _g("d", 9, 12, ended="says_done")]
  text = {"label": "unexplained", "step": None}
  out = tg.classify_with_screen(text, agent, [ClaimCheck(id="d", happened="no")], [])
  assert (out["label"], out["step"], out["hi"]) == ("false_claim", 9, 12)
  missed = SearchCheck(id="f", first_step=1, last_step=8, visible_step=3, used_when_visible=False)
  out = tg.classify_with_screen(text, agent, [ClaimCheck(id="d", happened="no")], [missed])
  assert (out["label"], out["step"], out["n_missed_searches"]) == ("perception", 3, 1)
  # visible and then used: the search simply succeeded, no perception error
  used = SearchCheck(id="f", first_step=1, last_step=8, visible_step=3, used_when_visible=True)
  assert tg.classify_with_screen(text, agent, [], [used])["label"] == "unexplained"


def test_knowledge_becomes_perception_when_the_target_was_visible():
  from goldjudge.subgoals.textgoals import SearchCheck
  agent = [_g("f", 2, 9, kind="find")]
  text = {"label": "knowledge", "step": 2, "hi": 9, "agent_goal": "f"}
  assert tg.classify_with_screen(text, agent, [], [SearchCheck(id="f", first_step=2, last_step=9, visible_step=5, used_when_visible=False)])["label"] == "perception"
  assert tg.classify_with_screen(text, agent, [], [SearchCheck(id="f", first_step=2, last_step=9, visible_step=5, used_when_visible=True)])["label"] == "knowledge"
  assert tg.classify_with_screen(text, agent, [], [SearchCheck(id="f", first_step=2, last_step=9)])["label"] == "knowledge"


def test_claims_and_searches_selected_from_serving_goals():
  agent = [_g("a", 1, 2, ended="says_done"), _g("b", 3, 4, ended="says_done"), _g("f", 5, 12, kind="find"), _g("g", 13, 14, kind="find")]
  am = [AgentMatch(id="a", relation="serves"), AgentMatch(id="b", relation="extra"), AgentMatch(id="f", relation="serves"), AgentMatch(id="g", relation="serves")]
  assert [g.id for g in tg.claims_to_check(agent, am, 12)] == ["a"]
  assert [g.id for g in tg.searches_to_check(agent, am, min_steps=4, max_n=3)] == ["f"]
  assert tg.after_step(agent[0], {3}) == 3 and tg.after_step(agent[0], {2}) == 2 and tg.after_step(agent[0], set()) is None
