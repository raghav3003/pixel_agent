import sys
from pathlib import Path

from goldjudge.subgoals import localizers as L

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from localize_ideas import vote  # noqa: E402


def test_parse_root_clamps_and_keeps_extra_fields():
  out = L.parse_root('x ```json\n{"root_step": 99, "span": "3", "evidence": "e"}\n```', 20)
  assert out["root_step"] == 20 and out["span"] == "3" and out["parse_ok"]
  assert not L.parse_root("no json", 20)["parse_ok"]


def test_vote_picks_the_densest_pm2_window():
  assert vote([3, 4, 5, 20]) == 4  # 3 votes within +-2 of 3..5 -> median of the tied best
  assert vote([10]) == 10
  assert vote([float("nan")]) is None


def test_label_definition_is_in_every_prompt():
  for p in (L.JUDGE_SYSTEM, L.JUDGE_G1_SYSTEM, L.CHECKPOINT_SYSTEM, L.SPANSEL_SYSTEM, L.DOCHECK_SYSTEM):
    assert "ROOT-CAUSE STEP" in p and "JSON only" in p
  assert "human wrote down" in L.JUDGE_G1_SYSTEM


def test_candidate_steps_pool_neighbours_and_rank_by_support():
  assert L.candidate_steps([5, 5, 6, 20, 20, 2, 9], max_n=3) == [2, 5, 20]  # ties in support: earliest first
  assert L.candidate_steps([5, 5, 6, 20, 20, 9], max_n=2) == [5, 20]
  assert L.candidate_steps([]) == []
