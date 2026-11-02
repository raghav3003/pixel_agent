#!/usr/bin/env python3
"""Score subgoal-based localization against CUADebug human root steps, next to judge baselines.

Predictors (per failed trajectory):
  rule_verified     first essential subgoal not reached / lost, after refuting subgoals a success skipped
  rule_unverified   same rule without the verification step (ablation)
  llm               LLM root step inside the rule window
  judge:<V>/<cond>  earlier whole-trajectory judges on the same trajectories (e.g. gpt6luna-fixed-v0.3/G0)
  const3            always step 3 (trivial baseline: scores ±2 ≈ .43 on the 204)

Metrics: exact, pm2 (|pred - human| <= 2), and for rules the window hit rate (human root in [lo, hi]) and
mean window width. Also subgoal-set statistics (size, support, verification).

  python scripts/score_subgoals.py --version gpt6luna-sg-v0.2 [--pilot data/index/h50.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINES = ["gpt6luna-fixed-v0.3/G0", "gpt6luna-fixed-v0.3/G1", "gpt6luna-fixed-v0.3/G2",
                     "gpt6luna-think-fixed-v0.3/G0", "gpt6luna-think-fixed-v0.3/G1",
                     "qwen3vl32b-fixed-v0.3/G0", "qwen3vl32b-fixed-v0.3/G1"]


def load_localize(d: Path) -> pd.DataFrame:
  rows = []
  for f in sorted((d / "localize").glob("*.jsonl")):
    for line in f.read_text(encoding="utf-8").splitlines():
      if line.strip():
        rows.append(json.loads(line))
  df = pd.DataFrame(rows)
  return df.drop_duplicates("key", keep="last") if not df.empty else df


def load_judge(base: str) -> dict[tuple[str, str], int | None]:
  v, cond = base.split("/")
  out = {}
  for f in (ROOT / "data/judge" / v / cond).glob("*.jsonl"):
    for line in f.read_text(encoding="utf-8").splitlines():
      try:
        r = json.loads(line)
      except json.JSONDecodeError:
        continue
      step = (r.get("verdict") or {}).get("root_step")
      out[(r["agent"], r["task_id"])] = int(step) if isinstance(step, (int, float)) else None
  return out


def metrics(pred: pd.Series, gold: pd.Series) -> dict[str, float]:
  ok = pred.notna()
  diff = (pred[ok] - gold[ok]).abs()
  n = len(pred)
  return {"n": n, "coverage": round(ok.mean(), 3), "exact": round((diff == 0).sum() / n, 3),
          "pm2": round((diff <= 2).sum() / n, 3), "mae": round(diff.mean(), 2) if len(diff) else None}


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--version", required=True)
  ap.add_argument("--out", default=str(ROOT / "data/subgoals"))
  ap.add_argument("--pilot")
  ap.add_argument("--baselines", nargs="*", default=DEFAULT_BASELINES)
  args = ap.parse_args()
  d = Path(args.out) / args.version
  df = load_localize(d)
  if df.empty:
    print("no localize rows")
    return
  if args.pilot:
    keys = {(r["agent"], r["task_id"]) for r in json.loads(Path(args.pilot).read_text())["rows"]}
    df = df[[(a, t) in keys for a, t in zip(df.agent, df.task_id)]]
  df = df[df.human_root.notna()].copy()
  gold = df.human_root.astype(float)

  preds: dict[str, pd.Series] = {}
  rows = []
  for v in ("rule_verified", "rule_unverified"):
    loc = df[v]
    preds[v] = loc.map(lambda x: x["rule_step"]).astype(float)
    lo, hi = loc.map(lambda x: x["lo"]), loc.map(lambda x: x["hi"])
    m = metrics(preds[v], gold)
    m |= {"window_hit": round(((lo <= gold) & (gold <= hi)).mean(), 3), "window_width": round((hi - lo + 1).mean(), 1)}
    rows.append({"predictor": v, **m})
  preds["llm"] = df.llm.map(lambda x: x.get("root_step")).astype(float)
  rows.append({"predictor": "llm", **metrics(preds["llm"], gold)})
  preds["const3"] = pd.Series(3.0, index=df.index).clip(upper=df.n_steps.astype(float))
  rows.append({"predictor": "const3", **metrics(preds["const3"], gold)})
  for b in args.baselines:
    j = load_judge(b)
    if not j:
      continue
    s = pd.Series([j.get((a, t)) for a, t in zip(df.agent, df.task_id)], index=df.index, dtype=float)
    rows.append({"predictor": f"judge:{b}", **metrics(s, gold), "judge_rows_present": int(s.notna().sum())})
  table = pd.DataFrame(rows)

  modes = df.rule_verified.map(lambda x: x["mode"])
  by_mode = []
  for mode, idx in modes.groupby(modes).groups.items():
    r = {"mode": mode, "n": len(idx)}
    for p in ("rule_verified", "llm"):
      r |= {f"{p}_{k}": v for k, v in metrics(preds[p][idx], gold[idx]).items() if k in {"exact", "pm2"}}
    loc = df.loc[idx, "rule_verified"]
    r["window_hit"] = round(((loc.map(lambda x: x["lo"]) <= gold[idx]) & (gold[idx] <= loc.map(lambda x: x["hi"]))).mean(), 3)
    by_mode.append(r)
  by_mode = pd.DataFrame(by_mode)
  by_agent = pd.DataFrame([{"agent": a, **{f"{p}_{k}": v for p in ("rule_verified", "llm")
                                           for k, v in metrics(preds[p][idx], gold[idx]).items() if k in {"n", "exact", "pm2"}}}
                           for a, idx in df.groupby("agent").groups.items()])

  sets = []
  for f in sorted((d / "sets_verified").glob("*.json")) or sorted((d / "sets").glob("*.json")):
    s = json.loads(f.read_text(encoding="utf-8"))
    sg = s["subgoals"]
    ess = [x for x in sg if x["necessity"] == "essential"]
    sets.append({"task_id": s["task_id"], "n_subgoals": len(sg), "n_essential": len(ess),
                 "n_views": len(s.get("views") or []),
                 "mean_support_frac": sum(x.get("support_frac", 0) for x in sg) / max(1, len(sg)),
                 "essential_refuted": sum(x.get("verification") == "refuted" for x in ess),
                 "essential_verified": sum(x.get("verification") == "verified" for x in ess),
                 "n_golds": len((s.get("extra") or {}).get("gold_keys") or []),
                 "dropped_edges": len(s.get("dropped_edges") or [])})
  sets = pd.DataFrame(sets)

  mdir = d / "metrics" / (Path(args.pilot).stem if args.pilot else "all")
  mdir.mkdir(parents=True, exist_ok=True)
  table.to_csv(mdir / "localization.csv", index=False)
  by_mode.to_csv(mdir / "by_mode.csv", index=False)
  by_agent.to_csv(mdir / "by_agent.csv", index=False)
  sets.to_csv(mdir / "subgoal_sets.csv", index=False)
  print(f"== localization ({len(df)} labelled failures) ==\n{table.to_string(index=False)}")
  print(f"\n== by failure mode (rule_verified) ==\n{by_mode.to_string(index=False)}")
  print(f"\n== by agent ==\n{by_agent.to_string(index=False)}")
  if not sets.empty:
    print(f"\n== subgoal sets ({len(sets)} tasks) ==\n{sets.drop(columns='task_id').describe().round(2).to_string()}")
    ess = sets.n_essential.sum()
    print(f"essential subgoals: {ess}; verified {sets.essential_verified.sum()}, refuted {sets.essential_refuted.sum()} "
          f"(tasks with >=1 gold: {(sets.n_golds > 0).sum()})")
  print(f"\nwritten to {mdir}")


if __name__ == "__main__":
  main()
