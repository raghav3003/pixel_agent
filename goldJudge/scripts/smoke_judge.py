#!/usr/bin/env python3
"""Smoke-test the judge endpoint with one real trajectory step (one screenshot).

  export JUDGE_BASE_URL=http://<node>:8000/v1 JUDGE_MODEL=judge
  python scripts/smoke_judge.py
"""

from __future__ import annotations

import time

from goldjudge.judge.client import JudgeClient, JudgeClientConfig, load_env_file
from goldjudge.judge.images import encode_screenshot
from goldjudge.loaders import cuadebug


def main() -> None:
  archive, task_dir = next(cuadebug.iter_task_dirs(archives=["gemini_2.5_pro_traj_30"]))
  traj = cuadebug.load_trajectory(task_dir, archive)
  step = traj.steps[0]
  load_env_file("config/judge.env")
  client = JudgeClient(JudgeClientConfig(max_tokens=200))
  t0 = time.time()
  resp = client.chat(
    "You are a concise assistant.",
    [
      {"type": "text", "text": f"Task domain: {traj.domain}. Describe this desktop screenshot in one sentence, then say which application is in focus."},
      {"type": "image_url", "image_url": {"url": encode_screenshot(step.screenshot_path, max_width=1280)}},
    ],
  )
  print(f"model={resp.model} latency={time.time() - t0:.1f}s usage={resp.usage}")
  print(resp.text)


if __name__ == "__main__":
  main()
