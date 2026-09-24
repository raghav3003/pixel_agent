import json
from pathlib import Path

from goldjudge.loaders import cuadebug, osworld_verified


def _png(p: Path) -> None:
  p.write_bytes(b"\x89PNG\r\n\x1a\n")


def _mk_task(root: Path, archive: str, domain: str, task: str, rows: list[dict], result: str, ann: dict | None) -> Path:
  d = root / archive / domain / task
  d.mkdir(parents=True)
  with (d / "traj.jsonl").open("w") as fh:
    for r in rows:
      fh.write(json.dumps(r) + "\n")
      _png(d / r["screenshot_file"])
  (d / "result.txt").write_text(result)
  if ann:
    (d / "human_annotation.json").write_text(json.dumps(ann))
  return d


def test_cuadebug_claude_merges_multi_row_steps(tmp_path):
  rows = [
    {"step_num": 1, "action_timestamp": "t1", "screenshot_file": "step_1_t1.png", "reward": 0, "done": False,
     "action": {"name": "computer", "input": {"action": "left_click", "coordinate": [100, 50]}, "action_type": "tool_use",
                "command": "pyautogui.click(150, 75)\n", "raw_response": "[THINKING] think a\n[TOOL_USE] computer: {}"},
     "response": "clicking"},
    {"step_num": 1, "action_timestamp": "t1b", "screenshot_file": "step_1_t1.png", "reward": 0, "done": False,
     "action": {"name": "computer", "input": {"action": "key", "text": "Return"}, "action_type": "tool_use",
                "command": "pyautogui.press('enter')\n", "raw_response": "[TEXT] press enter"},
     "response": "press enter"},
    {"step_num": 2, "action_timestamp": "t2", "screenshot_file": "step_2_t2.png", "reward": 0, "done": True,
     "action": {"action_type": "DONE", "raw_response": "[TEXT] done"}, "response": "done"},
  ]
  ann = {"root_error_step": 1, "taxonomy_tag": "G1", "evidence": "e", "correction": "c", "confidence": "high"}
  _mk_task(tmp_path, "claude_4.5_traj_144", "chrome", "t1", rows, "0.0", ann)
  trajs = cuadebug.load_all(tmp_path, ["claude_4.5_traj_144"])
  assert len(trajs) == 1
  t = trajs[0]
  assert t.agent == "claude-sonnet-4-5" and t.model_coord_space == (1280, 720)
  assert t.n_steps == 2
  s1 = t.steps[0]
  assert s1.action_type == "click" and s1.coords == [150.0, 75.0]
  assert s1.model_action.coordinate == [100.0, 50.0]
  assert "press" in s1.action_raw and len(s1.extra["sub_actions"]) == 1
  assert "think a" in s1.thought
  assert t.steps[1].action_type == "done" and t.steps[1].done
  assert t.result == 0.0 and not t.success
  assert t.annotation.taxonomy_tag == "G1" and t.annotation.taxonomy_family == "Grounding & Interaction"


def test_cuadebug_qwen_parses_tool_call(tmp_path):
  resp = ("Action: Click on cell B2.\n\n<tool_call>\n<function=computer_use>\n<parameter=action>\nleft_click\n"
          "</parameter>\n<parameter=coordinate>\n[125, 306]\n</parameter>\n</function>\n</tool_call>")
  rows = [{"step_num": 1, "action_timestamp": "t", "screenshot_file": "step_1_t.png", "reward": 0, "done": False,
           "action": "pyautogui.click(240, 330)", "actions": ["pyautogui.click(240, 330)"], "response": resp}]
  _mk_task(tmp_path, "qwen_3.5_traj_30", "libreoffice_calc", "t2", rows, "0", None)
  t = cuadebug.load_all(tmp_path, ["qwen_3.5_traj_30"])[0]
  s = t.steps[0]
  assert s.coords == [240.0, 330.0]
  assert s.model_action.coordinate == [125.0, 306.0] and s.model_action.type == "left_click"
  assert s.intent == "Click on cell B2."
  assert t.annotation is None


def test_osworld_verified_opencua_layout(tmp_path):
  pkg = tmp_path / "opencua_agent-opencua_7b-cot_l2-action_history-3image-Ubuntu-50steps"
  d = pkg / "turn_2" / "gimp" / "task-x"
  d.mkdir(parents=True)
  resp = ("# Step 1:\n## Thought:\nI should click the tool.\n\n## Action:\nClick the move tool.\n\n"
          "## Code:\n```python\npyautogui.click(x=0.1, y=0.2)\n```\n")
  row = {"step_num": 1, "action_timestamp": "t", "action": "pyautogui.click(192, 216)", "response": resp,
         "reward": 0, "done": False, "info": {}, "screenshot_file": "step_1_t.png"}
  (d / "traj.jsonl").write_text(json.dumps(row) + "\n")
  _png(d / "step_1_t.png")
  (d / "result.txt").write_text("1.0")
  eps = list(osworld_verified.iter_episode_dirs(tmp_path))
  assert len(eps) == 1 and eps[0]["run"] == "turn_2" and eps[0]["agent"] == "opencua-7b" and eps[0]["horizon"] == 50
  t = osworld_verified.load_trajectory(d, package=pkg.name, run="turn_2", agent="opencua-7b", horizon=50)
  assert t.success and t.domain == "gimp" and t.task_id == "task-x"
  s = t.steps[0]
  assert s.coords == [192.0, 216.0] and s.model_action.coordinate == [0.1, 0.2]
  assert s.intent == "Click the move tool." and "click the tool" in s.thought
  assert t.model_coord_space == (1, 1)
