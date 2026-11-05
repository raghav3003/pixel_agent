#!/usr/bin/env python3
"""Procedure study: compare root-step localizers on the same failed runs (docs/subgoals.md section 8).

  python scripts/localize_ideas.py run   --method judge --version V --pilot data/index/h50.json --samples 1
  python scripts/localize_ideas.py score --version V --pilot data/index/h50.json

Methods: judge, judge_g1, checkpoint, spansel, docheck (see goldjudge/subgoals/localizers.py). checkpoint/spansel/docheck
read subgoals from a text-subgoal version (--textgoals, default gpt6luna-text-v0.3), so that version must cover the pilot.
Outputs: data/ideas/<V>/<method>/s<k>/<agent>/<task>.json (resumable). Human labels are used only to score.
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
from goldjudge.actions import human_action_log  # noqa: E402
from goldjudge.subgoals import core, localizers as L  # noqa: E402
from goldjudge.subgoals.textgoals import TextGoal  # noqa: E402

METHODS = {"judge": L.JUDGE_SYSTEM, "judge_g1": L.JUDGE_G1_SYSTEM, "checkpoint": L.CHECKPOINT_SYSTEM,
           "spansel": L.SPANSEL_SYSTEM, "docheck": L.DOCHECK_SYSTEM, "counterfactual": L.COUNTERFACTUAL_SYSTEM}
OLD_JUDGES = ["gpt6luna-fixed-v0.3/G0", "gpt6luna-fixed-v0.3/G1"]


def _goals(p: Path) -> list[TextGoal]:
  return [TextGoal(**g) for g in json.loads(p.read_text(encoding="utf-8"))["subgoals"]]


def content_for(ctx: Ctx, method: str, r, traj, log) -> list[dict]:
  meta = ctx.meta(r.task_id)
  task = core.task_text(meta, ctx.osw)
  a = ctx.args
  tgd = ROOT / "data/textgoals" / a.textgoals
  if method == "judge":
    return L.judge_content(task, traj, log, image_budget=a.images, max_width=a.max_width)
  if method == "judge_g1":
    human = human_action_log(meta.human).render(prefix="h") if meta.human else None
    return L.judge_content(task, traj, log, image_budget=a.images, max_width=a.max_width, human=human)
  if method == "checkpoint":
    return L.judge_content(task, traj, log, image_budget=a.images, max_width=a.max_width,
                           checkpoints=_goals(tgd / "gold_human" / f"{r.task_id}.r1.json"))
  spans = _goals(tgd / "agent" / r.agent / f"{r.task_id}.r1.json")
  if method == "spansel":
    return L.spans_content(task, traj, log, spans, image_budget=a.images, max_width=a.max_width)
  if method == "docheck":
    return L.spans_content(task, traj, log, spans, image_budget=a.images, max_width=a.max_width, kinds={"do"})
  raise ValueError(method)


def candidates_for(ctx: Ctx, r) -> list[int]:
  """Every saved prediction of the --candidates-from methods (all samples) for this run."""
  preds = []
  for m in ctx.args.candidates_from:
    for f in (ctx.out / m).glob(f"s*/{r.agent}/{r.task_id}.json"):
      step = json.loads(f.read_text(encoding="utf-8"))["out"].get("root_step")
      if step is not None:
        preds.append(step)
  return L.candidate_steps(preds, max_n=ctx.args.max_candidates)


def run(ctx: Ctx) -> int:
  a = ctx.args
  units = [(r, k) for r in ctx.fails.itertuples() for k in range(1, a.samples + 1)]

  def one(u) -> str | None:
    r, k = u
    p = ctx.out / a.method / f"s{k}" / r.agent / f"{r.task_id}.json"
    if p.exists():
      return None
    traj, log = load_episode(r), load_log(r)
    extra = {}
    if a.method == "counterfactual":
      cands = candidates_for(ctx, r)
      if not cands:
        raise RuntimeError("no candidate predictions from --candidates-from")
      content = L.counterfactual_content(core.task_text(ctx.meta(r.task_id), ctx.osw), traj, log, cands,
                                         image_budget=a.images, max_width=a.max_width)
      extra = {"candidates": cands, "candidates_from": a.candidates_from}
    else:
      content = content_for(ctx, a.method, r, traj, log)
    text, usage = call(ctx.think, METHODS[a.method], content)
    out = L.parse_root(text, traj.n_steps)
    if not out["parse_ok"]:
      raise RuntimeError(f"no root_step parsed: {text[:300]}")
    write_json(p, {"agent": r.agent, "task_id": r.task_id, "method": a.method, "sample": k, "n_steps": traj.n_steps,
                   "prompt_version": L.LOCALIZER_PROMPT_VERSION, "textgoals": a.textgoals, "out": out, "usage": usage, **extra})
    return "ok"

  return run_units(f"{a.method}", units, one, a.workers)


def vote(preds: list[float]) -> float | None:
  """The step whose +-2 window holds the most predictions (ties: the median of the tied candidates)."""
  ps = [p for p in preds if p == p]
  if not ps:
    return None
  cands = range(int(min(ps)), int(max(ps)) + 1)
  scores = {c: sum(abs(p - c) <= 2 for p in ps) for c in cands}
  best = max(scores.values())
  return float(np.median([c for c, v in scores.items() if v == best]))


def score(args: argparse.Namespace) -> None:
  d = Path(args.out) / args.version
  labels = pd.read_parquet(Path(args.index) / "episodes.parquet")
  labels = labels[labels.source == "cuadebug"].set_index(["agent", "task_id"])
  keys = [(r["agent"], r["task_id"]) for r in json.loads(Path(args.pilot).read_text())["rows"]] if args.pilot else list(labels.index)
  gold = labels.root_error_step.reindex(keys).astype(float)
  keys = [k for k in keys if gold.get(k) == gold.get(k)]
  gold = gold.reindex(keys)
  preds: dict[str, pd.Series] = {}
  cost = 0.0
  for mdir in sorted(p for p in d.iterdir() if p.is_dir() and p.name in METHODS):
    samples = sorted(mdir.glob("s*"))
    per_sample = []
    for sd in samples:
      vals = {}
      for f in sd.rglob("*.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        vals[(r["agent"], r["task_id"])] = r["out"]["root_step"]
        cost += sum(((u.get("prompt_tokens") or 0) * args.price_in + (u.get("completion_tokens") or 0) * args.price_out) / 1e6
                    for u in usages(r))
      per_sample.append(pd.Series([vals.get(k) for k in keys], index=pd.MultiIndex.from_tuples(keys), dtype=float))
    if not per_sample:
      continue
    preds[mdir.name] = per_sample[0]
    if len(per_sample) > 1:
      m = pd.concat(per_sample, axis=1)
      preds[f"{mdir.name} vote{len(per_sample)}"] = pd.Series([vote(list(row)) for row in m.values], index=m.index, dtype=float)
  firsts = [m for m in METHODS if m in preds]
  for combo in [firsts] + [c.split("+") for c in args.ensembles]:
    if len(combo) > 1 and all(m in preds for m in combo):
      m = pd.concat([preds[c] for c in combo], axis=1)
      preds["ensemble " + "+".join(combo)] = pd.Series([vote(list(row)) for row in m.values], index=m.index, dtype=float)
  for b in OLD_JUDGES:
    j = load_judge(b)
    preds[f"old {b}"] = pd.Series([j.get(k) for k in keys], index=pd.MultiIndex.from_tuples(keys), dtype=float)
  n_steps = labels.n_steps.reindex(keys).astype(float).values
  preds["const3"] = pd.Series(np.minimum(3.0, n_steps), index=pd.MultiIndex.from_tuples(keys))
  gold.index = pd.MultiIndex.from_tuples(keys)

  ref_name = args.reference if args.reference in preds else "judge"
  ref = (preds[ref_name] - gold).abs() <= 2 if ref_name in preds else None
  rows = []
  for name, p in preds.items():
    err = (p - gold).abs()
    hit = err <= 2
    row = {"method": name, "n": len(keys), "predicted": int(p.notna().sum()), "exact": round((err == 0).mean(), 3),
           "pm2": round(hit.mean(), 3), "pm1": round((err <= 1).mean(), 3), "pm5": round((err <= 5).mean(), 3)}
    if ref is not None and name != ref_name:
      a_, b_ = int((hit & ~ref).sum()), int((ref & ~hit).sum())
      row[f"vs_{ref_name}"] = f"+{a_}/-{b_} p={binomtest(a_, a_ + b_).pvalue:.2f}" if a_ + b_ else "same"
    rows.append(row)
  table = pd.DataFrame(rows)
  mdir = d / "metrics" / (Path(args.pilot).stem if args.pilot else "all")
  mdir.mkdir(parents=True, exist_ok=True)
  table.to_csv(mdir / "localization.csv", index=False)
  print(f"== {len(keys)} labelled failures; +-2 is primary; sign test vs '{ref_name}' on +-2 (uncorrected) ==")
  print(table.to_string(index=False))
  print(f"cost so far ${cost:.2f}  -> {mdir}")


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("cmd", choices=["run", "score"])
  ap.add_argument("--method", choices=list(METHODS))
  ap.add_argument("--version", required=True)
  ap.add_argument("--textgoals", default="gpt6luna-text-v0.3")
  ap.add_argument("--index", default=str(ROOT / "data/index"))
  ap.add_argument("--out", default=str(ROOT / "data/ideas"))
  ap.add_argument("--pilot")
  ap.add_argument("--limit", type=int, default=None)
  ap.add_argument("--samples", type=int, default=1)
  ap.add_argument("--images", type=int, default=16)
  ap.add_argument("--candidates-from", nargs="*", default=["judge_g1", "checkpoint", "docheck"],
                  help="counterfactual: methods whose saved predictions (all samples) are the candidates")
  ap.add_argument("--max-candidates", type=int, default=4)
  ap.add_argument("--reference", default="judge", help="score: method the others are paired against")
  ap.add_argument("--ensembles", nargs="*", default=["judge_g1+checkpoint", "judge_g1+checkpoint+docheck"],
                  help="score: vote across these methods' first samples")
  ap.add_argument("--env-file", default=str(ROOT / "config/openai.env"))
  ap.add_argument("--provider", default=None)
  ap.add_argument("--model", default=None)
  ap.add_argument("--reasoning-effort", default=None)
  ap.add_argument("--think-effort", default="medium")
  ap.add_argument("--max-tokens", type=int, default=4000)
  ap.add_argument("--think-max-tokens", type=int, default=16000)
  ap.add_argument("--workers", type=int, default=10)
  ap.add_argument("--max-width", type=int, default=1280)
  ap.add_argument("--price-in", type=float, default=0.10)
  ap.add_argument("--price-out", type=float, default=0.50)
  ap.add_argument("--dry-run", action="store_true")
  args = ap.parse_args()
  if args.cmd == "score":
    score(args)
    return
  if not args.method:
    ap.error("--method is required for run")
  ctx = Ctx(args)
  print(f"method={args.method} version={args.version} failures={len(ctx.fails)} samples={args.samples}", flush=True)
  if args.dry_run:
    return
  ctx.out.mkdir(parents=True, exist_ok=True)
  sys.exit(1 if run(ctx) else 0)


if __name__ == "__main__":
  main()
