#!/usr/bin/env python3
"""The agent's own subgoals from its reasoning: were they the task's, and did they complete?

Runs on top of a finished subgoal version (needs its sets and fail tracks; see docs/subgoals.md):

  python scripts/agent_goals.py run   --version gpt6luna-sg-v0.2 --env-file config/openai.env \
      --reasoning-effort none --think-effort medium --workers 8 [--pilot data/index/h50.json]
  python scripts/agent_goals.py score --version gpt6luna-sg-v0.2 [--pilot data/index/h50.json]

Output: data/subgoals/<V>/agent_goals/<agent>.jsonl (one row per failed trajectory, resumable),
metrics in data/subgoals/<V>/metrics/<pilot|all>/agent_goals_*.csv. Human labels are used only to score.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from subgoals import ROOT, Ctx, call, captioned_log, load_episode, load_set, run_units  # noqa: E402
from score_subgoals import metrics  # noqa: E402
from goldjudge.subgoals import intents  # noqa: E402
from goldjudge.subgoals.schema import Track  # noqa: E402

_lock = threading.Lock()
FAMILY = {"Perception": "P", "Grounding & Interaction": "G", "Task Reasoning & Control": "R", "External / System": "S",
          "Infeasible Task": "IF"}


def run(ctx: Ctx) -> int:
  out = ctx.out / "agent_goals"
  out.mkdir(parents=True, exist_ok=True)
  done = set()
  for f in out.glob("*.jsonl"):
    done |= {json.loads(x)["key"] for x in f.read_text(encoding="utf-8").splitlines() if x.strip()}

  def one(r) -> str | None:
    tp = ctx.ep_path("tracks/fail", r)
    if not tp.exists():
      raise RuntimeError("no track")
    track = Track(**json.loads(tp.read_text(encoding="utf-8")))
    if track.key in done:
      return None
    sset = load_set(ctx, r.task_id, verified=True) or load_set(ctx, r.task_id, verified=False)
    traj, meta, log = load_episode(r), ctx.meta(r.task_id), captioned_log(ctx, r)
    t1, u1 = call(ctx.think, intents.EXTRACT_INTENTS_SYSTEM, intents.extract_content(meta, traj, log))
    goals = intents.parse_goals(t1, traj.n_steps)
    if not goals:
      raise RuntimeError(f"no goals parsed: {t1[:300]}")
    content = intents.align_content(meta, sset, track, goals, traj, image_budget=ctx.args.localize_images,
                                    max_width=ctx.args.max_width)
    t2, u2 = call(ctx.think, intents.ALIGN_SYSTEM, content)
    goals = intents.apply_alignment(goals, t2, {s.id for s in sset.subgoals})
    if all(g.alignment is None for g in goals):
      raise RuntimeError(f"alignment parse failed: {t2[:300]}")
    row = {"key": track.key, "agent": r.agent, "task_id": r.task_id, "domain": r.domain, "n_steps": track.n_steps,
           "version": ctx.args.version, "prompt_version": intents.INTENT_PROMPT_VERSION,
           "human_root": None if pd.isna(r.root_error_step) else int(r.root_error_step), "human_tag": r.taxonomy_tag,
           "human_family": FAMILY.get(r.taxonomy_family, ""), "goals": [g.model_dump() for g in goals],
           "summary": intents.classify(goals, sset, track), "usage": [u1, u2]}
    with _lock, (out / f"{r.agent}.jsonl").open("a", encoding="utf-8") as fh:
      fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return "ok"

  return run_units("agent_goals", list(ctx.fails.itertuples()), one, ctx.args.workers)


def load_rows(d: Path, pilot: str | None) -> pd.DataFrame:
  rows = []
  for f in sorted((d / "agent_goals").glob("*.jsonl")):
    rows += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
  df = pd.DataFrame(rows)
  if df.empty:
    return df
  df = df.drop_duplicates("key", keep="last")
  if pilot:
    keys = {(r["agent"], r["task_id"]) for r in json.loads(Path(pilot).read_text())["rows"]}
    df = df[[(a, t) in keys for a, t in zip(df.agent, df.task_id)]]
  return df


def score(args: argparse.Namespace) -> None:
  d = Path(args.out) / args.version
  df = load_rows(d, args.pilot)
  if df.empty:
    print("no agent_goals rows")
    return
  df = df.join(pd.DataFrame(list(df["summary"]), index=df.index))  # keeps the per-alignment dicts as cells
  labelled = df[df.human_root.notna()]
  gold = labelled.human_root.astype(float)

  # 1. Are intents and beliefs on task? share of agent steps by alignment, false claims, per agent.
  per_agent = []
  for a, g in df.groupby("agent"):
    tot = sum(sum(x.values()) for x in g.alignment_steps) or 1
    rec = {"agent": a, "n": len(g), "goals_per_traj": round(g.n_goals.mean(), 1), "inferred_share": round(g.n_inferred.sum() / max(1, g.n_goals.sum()), 2)}
    for k in sorted(intents.ALIGNMENTS):
      rec[f"steps_{k}"] = round(sum(x[k] for x in g.alignment_steps) / tot, 2)
    rec |= {"traj_with_false_claim": round((g.n_false_claims > 0).mean(), 2), "traj_with_contradiction": round(g.divergence_kind.notna().mean(), 2)}
    per_agent.append(rec)
  per_agent = pd.DataFrame(per_agent)

  # 2. What happened to the first required subgoal the agent missed, against the human error family.
  crosstab = pd.crosstab(df.primary, df.human_family, margins=True)

  # 3. Is the first departure of intent/belief where the human put the error?
  preds = {"divergence_step": labelled.divergence_step.astype(float)}
  loc = pd.DataFrame([{"predictor": k, **metrics(v, gold)} for k, v in preds.items()])
  have = labelled.divergence_step.notna()
  sub = metrics(labelled.divergence_step[have].astype(float), gold[have]) if have.any() else {}
  by_kind = pd.DataFrame([{"kind": k, **metrics(g.divergence_step.astype(float), g.human_root.astype(float))}
                          for k, g in labelled[have].groupby("divergence_kind")])

  mdir = d / "metrics" / (Path(args.pilot).stem if args.pilot else "all")
  mdir.mkdir(parents=True, exist_ok=True)
  per_agent.to_csv(mdir / "agent_goals_by_agent.csv", index=False)
  crosstab.to_csv(mdir / "agent_goals_primary_vs_family.csv")
  loc.to_csv(mdir / "agent_goals_localization.csv", index=False)
  print(f"== agent intent and beliefs ({len(df)} failures) ==\n{per_agent.T.to_string(header=False)}")
  print(f"\n== fate of the first missed required subgoal vs human error family ==\n{crosstab.to_string()}")
  print(f"\n== first departure of intent/belief vs human root step ==\n{loc.to_string(index=False)}")
  print(f"when a departure exists (n={int(have.sum())}): {sub}")
  if not by_kind.empty:
    print(by_kind.to_string(index=False))
  print(f"\nwritten to {mdir}")


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("cmd", choices=["run", "score"])
  ap.add_argument("--version", required=True)
  ap.add_argument("--index", default=str(ROOT / "data/index"))
  ap.add_argument("--out", default=str(ROOT / "data/subgoals"))
  ap.add_argument("--pilot")
  ap.add_argument("--limit", type=int, default=None)
  ap.add_argument("--env-file", default=str(ROOT / "config/openai.env"))
  ap.add_argument("--provider", default=None)
  ap.add_argument("--model", default=None)
  ap.add_argument("--reasoning-effort", default=None)
  ap.add_argument("--think-effort", default=None)
  ap.add_argument("--max-tokens", type=int, default=4000)
  ap.add_argument("--think-max-tokens", type=int, default=16000)
  ap.add_argument("--workers", type=int, default=4)
  ap.add_argument("--max-width", type=int, default=1280)
  ap.add_argument("--localize-images", type=int, default=12, help="screenshots per align call")
  ap.add_argument("--dry-run", action="store_true")
  args = ap.parse_args()
  if args.cmd == "score":
    score(args)
    return
  ctx = Ctx(args)
  print(f"version={args.version} failures={len(ctx.fails)}", flush=True)
  if args.dry_run:
    return
  sys.exit(1 if run(ctx) else 0)


if __name__ == "__main__":
  main()
