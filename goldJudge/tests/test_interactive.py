from types import SimpleNamespace

from PIL import Image

from goldjudge.judge.client import JudgeClient, JudgeClientConfig, _AnthropicBackend
from goldjudge.judge.interactive import ScreenshotTool
from goldjudge.schema import Step, Trajectory


def _traj(tmp_path, name, n):
  steps = []
  for i in range(1, n + 1):
    p = tmp_path / f"{name}_{i}.png"
    Image.new("RGB", (64, 36), (i, i, i)).save(p)
    steps.append(Step(step=i, screenshot_path=str(p), action_raw=f"pyautogui.click({i}, {i})"))
  return Trajectory(source="t", package="p", agent=name, task_id="t1", domain="chrome", dir=str(tmp_path), steps=steps)


def _texts(blocks):
  return " ".join(b["text"] for b in blocks if b["type"] == "text")


def test_tool_budget_dedupe_and_limits(tmp_path):
  tool = ScreenshotTool(agent=_traj(tmp_path, "a", 10), budget=3, per_call=4, shown={"agent": [1, 10]})
  assert tool.definition()["input_schema"]["properties"]["trajectory"]["enum"] == ["agent"]
  blocks, err = tool("view_screenshots", {"trajectory": "agent", "steps": [1, 4, 4, 5, 99]})
  assert not err and sum(b["type"] == "image" for b in blocks) == 2
  assert "already shown" in _texts(blocks) and "99" in _texts(blocks)
  blocks, err = tool("view_screenshots", {"trajectory": "agent", "steps": [6, 7]})
  assert sum(b["type"] == "image" for b in blocks) == 1 and "exhausted" in _texts(blocks)
  assert tool.used == 3 and tool.viewed() == [1, 4, 5, 6, 10]
  assert tool.requests[1]["granted"] == [6] and tool.requests[1]["over_budget"] == [7]
  _, err = tool("view_screenshots", {"trajectory": "gold", "steps": [1]})
  assert err  # no gold in this condition


def test_tool_per_call_and_gold(tmp_path):
  tool = ScreenshotTool(agent=_traj(tmp_path, "a", 10), gold=_traj(tmp_path, "g", 5), budget=10, per_call=2)
  assert tool.definition()["input_schema"]["properties"]["trajectory"]["enum"] == ["agent", "gold"]
  blocks, _ = tool("view_screenshots", {"trajectory": "gold", "steps": [2, 3, 4]})
  assert sum(b["type"] == "image" for b in blocks) == 2 and tool.viewed("gold") == [2, 3]
  assert "(screenshot before g2, gold trajectory)" in _texts(blocks)


def _resp(content, stop, i=10):
  usage = SimpleNamespace(input_tokens=i, cache_read_input_tokens=5, cache_creation_input_tokens=1, output_tokens=3)
  return SimpleNamespace(content=content, stop_reason=stop, usage=usage, model="fake")


class _FakeMessages:
  def __init__(self, script):
    self.script, self.calls = list(script), []

  def create(self, **kw):
    self.calls.append({**kw, "messages": list(kw["messages"])})
    return self.script.pop(0)


def _client(script, max_tokens=100):
  cfg = JudgeClientConfig(provider="anthropic", model="fake", max_tokens=max_tokens, thinking="adaptive", effort="high")
  backend = object.__new__(_AnthropicBackend)
  backend.cfg = cfg
  backend._client = SimpleNamespace(messages=_FakeMessages(script))
  client = object.__new__(JudgeClient)
  client.cfg, client._backend = cfg, backend
  return client, backend._client.messages


def test_interactive_loop(tmp_path):
  use = SimpleNamespace(type="tool_use", id="tu1", name="view_screenshots", input={"trajectory": "agent", "steps": [3]})
  final = SimpleNamespace(type="text", text='{"root_step": 3, "primary": "G1"}')
  client, fake = _client([_resp([use], "tool_use"), _resp([final], "end_turn")])
  tool = ScreenshotTool(agent=_traj(tmp_path, "a", 5), budget=4)
  r = client.chat_interactive("sys", [{"type": "text", "text": "hi"}], [tool.definition()], tool, max_turns=5)
  assert r.text == final.text and r.n_turns == 2 and tool.viewed() == [3]
  assert r.usage["input_tokens"] == 20 and r.usage["prompt_tokens"] == 32 and r.usage["completion_tokens"] == 6
  first, second = fake.calls
  assert first["thinking"] == {"type": "adaptive"} and first["output_config"] == {"effort": "high"}
  assert first["cache_control"] == {"type": "ephemeral"} and "tool_choice" not in first
  tr = second["messages"][2]["content"][0]
  assert tr["type"] == "tool_result" and tr["tool_use_id"] == "tu1" and any(b["type"] == "image" for b in tr["content"])


def test_interactive_last_turn_forces_answer(tmp_path):
  use = SimpleNamespace(type="tool_use", id="tu1", name="view_screenshots", input={"trajectory": "agent", "steps": [2]})
  final = SimpleNamespace(type="text", text="{}")
  client, fake = _client([_resp([use], "tool_use"), _resp([final], "end_turn")])
  tool = ScreenshotTool(agent=_traj(tmp_path, "a", 5))
  client.chat_interactive("sys", [{"type": "text", "text": "hi"}], [tool.definition()], tool, max_turns=2)
  last = fake.calls[1]
  assert last["tool_choice"] == {"type": "none"}
  assert "No more tool calls" in last["messages"][-1]["content"][-1]["text"]


# --- OpenAI-compatible backend (vLLM) ---------------------------------------------------------

def _oa_resp(content=None, calls=None, finish="stop"):
  msg = SimpleNamespace(content=content, tool_calls=calls)
  return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish)], model="judge",
                         usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3))


class _FakeCompletions:
  def __init__(self, script):
    self.script, self.calls = list(script), []

  def create(self, **kw):
    self.calls.append(kw)
    return self.script.pop(0)


def _oa_client(script):
  from goldjudge.judge.client import _OpenAIBackend

  cfg = JudgeClientConfig(provider="openai", model="judge", max_tokens=100)
  backend = object.__new__(_OpenAIBackend)
  backend.cfg = cfg
  backend._client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions(script)))
  client = object.__new__(JudgeClient)
  client.cfg, client._backend = cfg, backend
  return client, backend._client.chat.completions


def _call(steps, cid="c1"):
  import json

  fn = SimpleNamespace(name="view_screenshots", arguments=json.dumps({"trajectory": "agent", "steps": steps}))
  return SimpleNamespace(id=cid, function=fn)


def test_openai_interactive_loop(tmp_path):
  client, fake = _oa_client([_oa_resp("looking", [_call([3, 4])]), _oa_resp('{"root_step": 3}')])
  tool = ScreenshotTool(agent=_traj(tmp_path, "a", 5), budget=4, shown={"agent": [4]})
  r = client.chat_interactive("sys", [{"type": "text", "text": "hi"}], [tool.definition()], tool, max_turns=5)
  assert r.text == '{"root_step": 3}' and r.n_turns == 2 and tool.viewed() == [3, 4]
  assert r.usage == {"prompt_tokens": 20, "completion_tokens": 6, "stop_reason": "stop"}
  first, second = fake.calls
  assert first["tools"][0]["function"]["name"] == "view_screenshots" and first["tool_choice"] == "auto"
  assert "parameters" in first["tools"][0]["function"]
  msgs = second["messages"]
  assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool", "user"]
  assert msgs[2]["tool_calls"][0]["id"] == "c1" and msgs[3]["tool_call_id"] == "c1"
  assert isinstance(msgs[3]["content"], str) and "1 screenshot(s) follow" in msgs[3]["content"] and "already shown" in msgs[3]["content"]
  imgs = msgs[4]["content"]
  assert [p["type"] for p in imgs] == ["text", "image_url"] and imgs[0]["text"].startswith("(screenshot before step 3")
  assert imgs[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_openai_last_turn_has_no_tools_and_flags_truncation(tmp_path):
  client, fake = _oa_client([_oa_resp(None, [_call([2])]), _oa_resp("{...", finish="length")])
  tool = ScreenshotTool(agent=_traj(tmp_path, "a", 5))
  r = client.chat_interactive("sys", [{"type": "text", "text": "hi"}], [tool.definition()], tool, max_turns=2)
  last = fake.calls[1]
  assert "tools" not in last and "tool_choice" not in last
  assert "No more tool calls" in last["messages"][-1]["content"][-1]["text"]
  assert r.usage["stop_reason"] == "max_tokens"


def test_openai_bad_arguments_do_not_crash(tmp_path):
  bad = SimpleNamespace(id="c1", function=SimpleNamespace(name="view_screenshots", arguments="{not json"))
  client, fake = _oa_client([_oa_resp(None, [bad]), _oa_resp("{}")])
  tool = ScreenshotTool(agent=_traj(tmp_path, "a", 5))
  client.chat_interactive("sys", [{"type": "text", "text": "hi"}], [tool.definition()], tool, max_turns=3)
  assert "JSON object" in fake.calls[1]["messages"][3]["content"]
