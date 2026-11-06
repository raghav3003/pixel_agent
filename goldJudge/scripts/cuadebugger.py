#!/usr/bin/env python3
"""Reproduce CUADebug's RCA harnesses (goldjudge/subgoals/cuadebugger.py) and test label anchoring.

  python scripts/cuadebugger.py run   --mode tool|prompt --version V --env-file config/openai.env --workers 6
  python scripts/cuadebugger.py score --version V

Outputs data/cuadebugger/<V>/<mode>/<agent>/<task>.json (resumable). ``score`` compares the reproductions with our
judges overall and split by whether the annotator started from a machine proposal (``chosen_debugger``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from score_subgoals import load_judge  # noqa: E402
from subgoal_cost import usages  # noqa: E402
from subgoals import ROOT, Ctx, call, load_episode, load_log, run_units, write_json  # noqa: E402
from goldjudge.subgoals import cuadebugger as C  # noqa: E402
from goldjudge.subgoals.core import parse_json  # noqa: E402


def run(ctx: Ctx) -> int:
  a = ctx.args

  def one(r) -> str | None:
    p = ctx.out / a.mode / r.agent / f"{r.task_id}.json"
    if p.exists():
      return None
    traj, log, meta = load_episode(r), load_log(r), ctx.meta(r.task_id)
    if a.mode == "tool":
      tools = C.DebuggerTools(traj, max_width=a.max_width)
      resp = ctx.fast.chat_interactive(C.TOOL_SYSTEM, C.tool_user(meta.instruction, traj), tools.definitions(), tools,
                                       max_turns=a.max_turns, keep_last_images=a.keep_images)
      step, rec = C.final_step(tools, resp.text)
      extra = {"inspected": tools.inspected, "finish_called": tools.finished is not None, "n_turns": resp.n_turns}
    else:
      resp = ctx.fast.chat(C.PROMPT_SYSTEM, C.prompt_user(meta.instruction, traj, log, last_n=a.last_n, max_width=a.max_width))
      rec = parse_json(resp.text, "root_error_step") or {}
      try:
        step = int(rec.get("root_error_step"))
      except (TypeError, ValueError):
        step = None
      extra = {}
    if step is None:
      raise RuntimeError(f"no root_error_step: {resp.text[:300]}")
    step = max(1, min(step, traj.n_steps))
    write_json(p, {"agent": r.agent, "task_id": r.task_id, "n_steps": traj.n_steps, "mode": a.mode,
                   "prompt_version": C.CUADEBUGGER_PROMPT_VERSION, "out": {"root_step": step, "taxonomy_tag": rec.get("taxonomy_tag"),
                   "evidence": str(rec.get("evidence") or "")[:800], "confidence": rec.get("confidence")},
                   "usage": {**resp.usage, "latency_s": round(resp.latency_s, 1)}, **extra})
    return "ok"

  return run_units(f"cuadebugger-{a.mode}", list(ctx.fails.itertuples()), one, a.workers)


def score(args: argparse.Namespace) -> None:
  d = Path(args.out) / args.version
  lab = pd.read_parquet(Path(args.index) / "episodes.parquet")
  lab = lab[(lab.source == "cuadebug") & lab.root_error_step.notna()].reset_index(drop=True)
  seeded = []
  for r in lab.itertuples():
    ann = json.loads((Path(r.dir) / "human_annotation.json").read_text(encoding="utf-8"))
    seeded.append(ann.get("chosen_debugger") or "")
  lab["seeded_by"] = seeded
  lab["seeded"] = lab.seeded_by != ""
  key = list(zip(lab.agent, lab.task_id))
  g = lab.root_error_step.astype(float).values
  preds = {}
  for name, spec in [("orig judge G0", "gpt6luna-fixed-v0.3/G0"), ("orig judge G1", "gpt6luna-fixed-v0.3/G1")]:
    j = load_judge(spec)
    preds[name] = np.array([j.get(k, np.nan) if j.get(k) is not None else np.nan for k in key], float)
  cost = 0.0
  for mode in ("prompt", "tool"):
    vals = {}
    for f in (d / mode).rglob("*.json"):
      r = json.loads(f.read_text(encoding="utf-8"))
      vals[(r["agent"], r["task_id"])] = r["out"]["root_step"]
      for u in usages(r):
        cost += ((u.get("prompt_tokens") or 0) * 0.1 + (u.get("completion_tokens") or 0) * 0.5) / 1e6
    if vals:
      preds[f"repro CUADebug {mode}"] = np.array([vals.get(k, np.nan) for k in key], float)
  preds["const 3"] = np.minimum(3, lab.n_steps.values).astype(float)
  ref = np.abs(preds["orig judge G1"] - g) <= 2
  rows = []
  for name, p in preds.items():
    e = np.abs(p - g)
    hit = e <= 2
    row = {"predictor": name, "have": int(np.isfinite(p).sum()), "exact": (e == 0).mean(), "pm2": hit.mean()}
    for who, m in [("seeded", lab.seeded.values), ("unseeded", ~lab.seeded.values)]:
      row[f"pm2_{who}"] = hit[m].mean()
      row[f"exact_{who}"] = (e[m] == 0).mean()
    for ag in ("claude-sonnet-4-5", "gemini-2.5-pro", "qwen3.5-vl"):
      row[f"pm2_{ag.split('-')[0]}"] = hit[lab.agent.values == ag].mean()
    if name != "orig judge G1":
      a_, b_ = int((hit & ~ref).sum()), int((ref & ~hit).sum())
      row["vs_G1"] = f"+{a_}/-{b_} p={binomtest(a_, a_ + b_).pvalue:.2f}" if a_ + b_ else "same"
    rows.append(row)
  t = pd.DataFrame(rows).round(3)
  pd.set_option("display.width", 250)
  print(f"{len(lab)} labelled failures; seeded by a machine proposal: {int(lab.seeded.sum())} "
        f"({lab[lab.seeded].seeded_by.value_counts().to_dict()}); by agent seeded share: "
        f"{lab.groupby('agent').seeded.mean().round(2).to_dict()}")
  print(t.to_string(index=False))
  print(f"reproduction call cost ${cost:.2f}")
  (d / "metrics").mkdir(parents=True, exist_ok=True)
  t.to_csv(d / "metrics" / "anchoring.csv", index=False)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("cmd", choices=["run", "score"])
  ap.add_argument("--mode", choices=["tool", "prompt"], default="tool")
  ap.add_argument("--version", required=True)
  ap.add_argument("--index", default=str(ROOT / "data/index"))
  ap.add_argument("--out", default=str(ROOT / "data/cuadebugger"))
  ap.add_argument("--pilot")
  ap.add_argument("--limit", type=int, default=None)
  ap.add_argument("--max-turns", type=int, default=40)
  ap.add_argument("--keep-images", type=int, default=4, help="tool mode: screenshots kept in context (older ones dropped)")
  ap.add_argument("--last-n", type=int, default=3, help="prompt mode: result screenshots of the last N steps")
  ap.add_argument("--env-file", default=str(ROOT / "config/openai.env"))
  ap.add_argument("--provider", default=None)
  ap.add_argument("--model", default=None)
  ap.add_argument("--reasoning-effort", default="none")
  ap.add_argument("--think-effort", default=None)
  ap.add_argument("--max-tokens", type=int, default=3000)
  ap.add_argument("--think-max-tokens", type=int, default=16000)
  ap.add_argument("--workers", type=int, default=6)
  ap.add_argument("--max-width", type=int, default=1024)
  ap.add_argument("--dry-run", action="store_true")
  args = ap.parse_args()
  if args.cmd == "score":
    score(args)
    return
  ctx = Ctx(args)
  print(f"mode={args.mode} version={args.version} failures={len(ctx.fails)}", flush=True)
  if args.dry_run:
    return
  ctx.out.mkdir(parents=True, exist_ok=True)
  sys.exit(1 if run(ctx) else 0)


if __name__ == "__main__":
  main()
