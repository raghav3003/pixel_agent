#!/usr/bin/env python3
"""Idea 1: step-level intent vs outcome for every step of every failed run (goldjudge/subgoals/stepcheck.py).

  python scripts/step_checks.py run   --version V --env-file config/openai.env --reasoning-effort none --workers 8
  python scripts/step_checks.py score --version V

Outputs data/stepcheck/<V>/<agent>/<task>.json (resumable). Scores "first mismatch" and "first harmful mismatch" as
root-step predictors on all labelled failures, by split (h50 dev / test154) and agent. Labels are used only to score.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from subgoals import ROOT, Ctx, call, load_episode, load_log, run_units, write_json  # noqa: E402
from goldjudge.subgoals import stepcheck as sc  # noqa: E402


def run(ctx: Ctx) -> int:
  def one(r) -> str | None:
    p = ctx.out / r.agent / f"{r.task_id}.json"
    if p.exists():
      return None
    traj, log, meta = load_episode(r), load_log(r), ctx.meta(r.task_id)
    verdicts, usage, unparsed = [], [], []
    for win in sc.windows(traj.n_steps, ctx.args.window):
      text, u = call(ctx.fast, sc.STEPCHECK_SYSTEM, sc.window_content(meta.instruction, traj, log, win, max_width=ctx.args.max_width))
      got = sc.parse_window(text, win)
      verdicts += got
      usage.append(u)
      if len(got) < len(win):
        unparsed.append({"window": win, "text": text[:2000]})
    if not verdicts:
      raise RuntimeError("no step verdicts parsed")
    write_json(p, {"agent": r.agent, "task_id": r.task_id, "n_steps": traj.n_steps, "prompt_version": sc.STEPCHECK_PROMPT_VERSION,
                   "verdicts": [v.model_dump() for v in verdicts], "unparsed": unparsed, "usage": usage})
    return "ok"

  return run_units("stepcheck", list(ctx.fails.itertuples()), one, ctx.args.workers)


def load(d: Path) -> dict[tuple[str, str], list[sc.StepVerdict]]:
  out = {}
  for f in d.rglob("*.json"):
    r = json.loads(f.read_text(encoding="utf-8"))
    if "verdicts" in r:
      out[(r["agent"], r["task_id"])] = [sc.StepVerdict(**v) for v in r["verdicts"]]
  return out


def score(args: argparse.Namespace) -> None:
  d = Path(args.out) / args.version
  ver = load(d)
  lab = pd.read_parquet(Path(args.index) / "episodes.parquet")
  lab = lab[(lab.source == "cuadebug") & lab.root_error_step.notna()]
  dev = {(r["agent"], r["task_id"]) for r in json.loads((Path(args.index) / "h50.json").read_text())["rows"]}
  rows = []
  for r in lab.itertuples():
    v = ver.get((r.agent, r.task_id))
    if v is None:
      continue
    miss = [x.step for x in v if x.match == "no"]
    harm = [x.step for x in v if x.match == "no" and x.harm]
    rows.append({"agent": r.agent, "split": "dev" if (r.agent, r.task_id) in dev else "test", "g": r.root_error_step,
                 "n_steps": r.n_steps, "n_miss": len(miss), "n_harm": len(harm),
                 "first_miss": miss[0] if miss else None, "first_harm": harm[0] if harm else None,
                 "any_miss_pm2": any(abs(m - r.root_error_step) <= 2 for m in miss)})
  df = pd.DataFrame(rows)
  print(f"{len(df)} labelled failures with step checks; mean flagged steps {df.n_miss.mean():.1f}, harmful {df.n_harm.mean():.1f}")
  for col in ("first_miss", "first_harm"):
    err = (df[col].astype(float) - df.g).abs()
    df[f"{col}_pm2"] = err <= 2
    df[f"{col}_exact"] = err == 0
  g = df.groupby(["split", "agent"]).agg(n=("g", "size"), first_miss_pm2=("first_miss_pm2", "mean"), first_harm_pm2=("first_harm_pm2", "mean"),
                                         first_harm_exact=("first_harm_exact", "mean"), some_miss_pm2=("any_miss_pm2", "mean"))
  tot = df.agg({"first_miss_pm2": "mean", "first_harm_pm2": "mean", "first_harm_exact": "mean", "any_miss_pm2": "mean"})
  print(g.round(3).to_string())
  print(f"all: first mismatch +-2 {tot.first_miss_pm2:.3f}; first harmful +-2 {tot.first_harm_pm2:.3f} (exact {tot.first_harm_exact:.3f}); "
        f"some mismatch within +-2 {tot.any_miss_pm2:.3f}")
  (d / "metrics").mkdir(parents=True, exist_ok=True)
  df.to_csv(d / "metrics" / "per_trajectory.csv", index=False)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("cmd", choices=["run", "score"])
  ap.add_argument("--version", required=True)
  ap.add_argument("--index", default=str(ROOT / "data/index"))
  ap.add_argument("--out", default=str(ROOT / "data/stepcheck"))
  ap.add_argument("--pilot")
  ap.add_argument("--limit", type=int, default=None)
  ap.add_argument("--window", type=int, default=6)
  ap.add_argument("--env-file", default=str(ROOT / "config/openai.env"))
  ap.add_argument("--provider", default=None)
  ap.add_argument("--model", default=None)
  ap.add_argument("--reasoning-effort", default="none")
  ap.add_argument("--think-effort", default=None)
  ap.add_argument("--max-tokens", type=int, default=3000)
  ap.add_argument("--think-max-tokens", type=int, default=16000)
  ap.add_argument("--workers", type=int, default=8)
  ap.add_argument("--max-width", type=int, default=1280)
  ap.add_argument("--dry-run", action="store_true")
  args = ap.parse_args()
  if args.cmd == "score":
    score(args)
    return
  ctx = Ctx(args)
  print(f"version={args.version} failures={len(ctx.fails)}", flush=True)
  if args.dry_run:
    return
  ctx.out.mkdir(parents=True, exist_ok=True)
  sys.exit(1 if run(ctx) else 0)


if __name__ == "__main__":
  main()
