#!/usr/bin/env python3
"""Idea 4: required states + method constraints per task, verified on successes, checked against failed runs.

  python scripts/constraint_check.py run   --version V --env-file config/openai.env --reasoning-effort none --workers 8
  python scripts/constraint_check.py score --version V

Uses the captioned successful runs and candidate states of the evaluator-anchored pipeline (--from-version, default
gpt6luna-sg-v0.2). Outputs data/constraints/<V>/{tasks,verify,check}/ (resumable). Labels are used only to score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_actions import log_path  # noqa: E402
from subgoals import ROOT, Ctx, call, load_episode, load_log, run_units, write_json  # noqa: E402
from goldjudge.actions import ActionLog, human_action_log  # noqa: E402
from goldjudge.subgoals import constraints as K, core  # noqa: E402
from goldjudge.subgoals.localizers import parse_root  # noqa: E402


def gold_log(ctx: Ctx, r) -> ActionLog:
  p = log_path(ROOT / "data/subgoals" / ctx.args.from_version / "captions", r)
  if p.exists():
    return ActionLog.model_validate(json.loads(p.read_text(encoding="utf-8"))["action_log"])
  return load_log(r)


def candidate_states(ctx: Ctx, task_id: str) -> list[tuple[str, str]]:
  base = ROOT / "data/subgoals" / ctx.args.from_version
  for sub in ("sets_verified", "sets"):
    p = base / sub / f"{task_id}.json"
    if p.exists():
      sgs = json.loads(p.read_text(encoding="utf-8"))["subgoals"]
      return [(s["id"], s["description"]) for s in sgs if s.get("necessity") == "essential" and s.get("verification") != "refuted"]
  return []


def task_ref(ctx: Ctx, t: str) -> tuple[list[K.StateMark], list[K.Constraint]]:
  d = json.loads((ctx.out / "tasks" / f"{t}.json").read_text(encoding="utf-8"))
  states = [K.StateMark(**s) for s in d["states"]]
  cons = [K.Constraint(**c) for c in d["constraints"]]
  for v in sorted((ctx.out / "verify" / t).glob("*.json")) if (ctx.out / "verify" / t).exists() else []:
    broken = json.loads(v.read_text(encoding="utf-8"))["broken"]
    for c in cons:
      if broken.get(c.id):
        c.broken_by.append(v.stem)
  return states, [c for c in cons if not c.broken_by]


def run(ctx: Ctx) -> int:
  a = ctx.args

  def extract_one(t) -> str | None:
    p = ctx.out / "tasks" / f"{t}.json"
    if p.exists():
      return None
    meta = ctx.meta(t)
    golds = [gold_log(ctx, r) for r in ctx.golds[ctx.golds.task_id == t].itertuples()][: a.max_golds]
    human = human_action_log(meta.human).render(prefix="h") if meta.human else None
    states = candidate_states(ctx, t)
    text, usage = call(ctx.fast, K.EXTRACT_SYSTEM, K.extract_content(core.task_text(meta, ctx.osw), human, golds, states))
    parsed = K.parse_extract(text, states)
    if parsed is None:
      raise RuntimeError(f"no constraints parsed: {text[:300]}")
    write_json(p, {"task_id": t, "prompt_version": K.CONSTRAINT_PROMPT_VERSION, "n_golds": len(golds),
                   "states": [s.model_dump() for s in parsed[0]], "constraints": [c.model_dump() for c in parsed[1]], "usage": usage})
    return "ok"

  def verify_one(r) -> str | None:
    p = ctx.out / "verify" / r.task_id / f"{r.agent}__{hashlib.md5(str(r.run).encode()).hexdigest()[:8]}.json"
    if p.exists():
      return None
    cons = [K.Constraint(**c) for c in json.loads((ctx.out / "tasks" / f"{r.task_id}.json").read_text(encoding="utf-8"))["constraints"]]
    if not cons:
      write_json(p, {"task_id": r.task_id, "agent": r.agent, "run": r.run, "broken": {}})
      return "ok"
    text, usage = call(ctx.fast, K.VERIFY_SYSTEM, K.verify_content(core.task_text(ctx.meta(r.task_id), ctx.osw), gold_log(ctx, r), cons))
    write_json(p, {"task_id": r.task_id, "agent": r.agent, "run": r.run, "broken": K.parse_verify(text), "usage": usage})
    return "ok"

  def check_one(r) -> str | None:
    p = ctx.out / "check" / r.agent / f"{r.task_id}.json"
    if p.exists():
      return None
    states, cons = task_ref(ctx, r.task_id)
    traj, log = load_episode(r), load_log(r)
    ref = K.render_reference(states, cons)
    text, usage = call(ctx.fast, K.CHECK_SYSTEM, K.check_content(core.task_text(ctx.meta(r.task_id), ctx.osw), ref, traj, log))
    out = parse_root(text, traj.n_steps)
    if not out["parse_ok"]:
      raise RuntimeError(f"no root_step parsed: {text[:300]}")
    write_json(p, {"agent": r.agent, "task_id": r.task_id, "n_steps": traj.n_steps, "reference": ref, "out": out, "usage": usage})
    return "ok"

  failed = run_units("extract", ctx.tasks, extract_one, a.workers)
  failed += run_units("verify", list(ctx.golds.itertuples()), verify_one, a.workers)
  failed += run_units("check", list(ctx.fails.itertuples()), check_one, a.workers)
  return failed


def score(args: argparse.Namespace) -> None:
  d = Path(args.out) / args.version
  lab = pd.read_parquet(Path(args.index) / "episodes.parquet")
  lab = lab[(lab.source == "cuadebug") & lab.root_error_step.notna()].set_index(["agent", "task_id"])
  dev = {(r["agent"], r["task_id"]) for r in json.loads((Path(args.index) / "h50.json").read_text())["rows"]}
  rows = []
  for f in (d / "check").rglob("*.json"):
    r = json.loads(f.read_text(encoding="utf-8"))
    k = (r["agent"], r["task_id"])
    if k in lab.index:
      g = lab.root_error_step[k]
      rows.append({"agent": k[0], "split": "dev" if k in dev else "test", "err": abs(r["out"]["root_step"] - g),
                   "violation": str(r["out"].get("violation", ""))[:20]})
  df = pd.DataFrame(rows)
  df["pm2"], df["exact"] = df.err <= 2, df.err == 0
  print(df.groupby(["split", "agent"]).agg(n=("err", "size"), pm2=("pm2", "mean"), exact=("exact", "mean")).round(3).to_string())
  print(f"all n={len(df)} pm2 {df.pm2.mean():.3f} exact {df.exact.mean():.3f}; no violation found in "
        f"{df.violation.str.lower().str.startswith('none').mean():.2f} of runs")
  tasks = [json.loads(f.read_text(encoding="utf-8")) for f in (d / "tasks").glob("*.json")]
  n_c = sum(len(t["constraints"]) for t in tasks)
  broken = sum(any(json.loads(v.read_text())["broken"].get(c["id"]) for v in (d / "verify" / t["task_id"]).glob("*.json"))
               for t in tasks for c in t["constraints"] if (d / "verify" / t["task_id"]).exists())
  req = [s["required"] for t in tasks for s in t["states"]]
  print(f"tasks {len(tasks)}: constraints {n_c} ({n_c / max(1, len(tasks)):.1f}/task), broken by a success {broken}; "
        f"candidate states marked required {sum(req)}/{len(req)}")


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("cmd", choices=["run", "score"])
  ap.add_argument("--version", required=True)
  ap.add_argument("--from-version", default="gpt6luna-sg-v0.2")
  ap.add_argument("--index", default=str(ROOT / "data/index"))
  ap.add_argument("--out", default=str(ROOT / "data/constraints"))
  ap.add_argument("--pilot")
  ap.add_argument("--limit", type=int, default=None)
  ap.add_argument("--max-golds", type=int, default=3)
  ap.add_argument("--env-file", default=str(ROOT / "config/openai.env"))
  ap.add_argument("--provider", default=None)
  ap.add_argument("--model", default=None)
  ap.add_argument("--reasoning-effort", default="none")
  ap.add_argument("--think-effort", default=None)
  ap.add_argument("--max-tokens", type=int, default=4000)
  ap.add_argument("--think-max-tokens", type=int, default=16000)
  ap.add_argument("--workers", type=int, default=8)
  ap.add_argument("--max-width", type=int, default=1280)
  ap.add_argument("--dry-run", action="store_true")
  args = ap.parse_args()
  if args.cmd == "score":
    score(args)
    return
  ctx = Ctx(args)
  print(f"version={args.version} failures={len(ctx.fails)} tasks={len(ctx.tasks)} golds={len(ctx.golds)}", flush=True)
  if args.dry_run:
    return
  ctx.out.mkdir(parents=True, exist_ok=True)
  sys.exit(1 if run(ctx) else 0)


if __name__ == "__main__":
  main()
