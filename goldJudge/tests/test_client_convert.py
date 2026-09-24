from goldjudge.judge.client import load_env_file, openai_parts_to_anthropic


def test_parts_conversion():
  parts = [
    {"type": "text", "text": "hello"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
    {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
  ]
  out = openai_parts_to_anthropic(parts)
  assert out[0] == {"type": "text", "text": "hello"}
  assert out[1]["type"] == "image" and out[1]["source"] == {"type": "base64", "media_type": "image/jpeg", "data": "AAAA"}
  assert out[2]["source"] == {"type": "url", "url": "https://x/y.png"}


def test_env_file(tmp_path, monkeypatch):
  p = tmp_path / "judge.env"
  p.write_text("# c\nJUDGE_MODEL=claude-haiku-4-5\nJUDGE_PROVIDER='anthropic'\n")
  monkeypatch.delenv("JUDGE_MODEL", raising=False)
  monkeypatch.setenv("JUDGE_PROVIDER", "openai")
  load_env_file(p)
  import os
  assert os.environ["JUDGE_MODEL"] == "claude-haiku-4-5"
  assert os.environ["JUDGE_PROVIDER"] == "openai"  # existing value wins
