#!/usr/bin/env python3
"""Score root-step predictors side by side on all labelled CUADebug failures (docs/subgoals.md section 10).

Predictor specs:
  judge:<judge_version>/<cond>      rows in data/judge/ (verdict.root_step)
  idea:<version>/<method>           data/ideas/<version>/<method>/s1
  stepcheck:<version>[:harm]        first step whose outcome mismatched intent (":harm" = first harmful one)
  constraints:<version>             data/constraints/<version>/check
  const:<k>                         always step k

  python scripts/compare_localizers.py --ref judge:gpt6luna-fixed-after-v0.3/G0 judge:gpt6luna-fixed-v0.3/G0 ...
Reports exact / +-1 / +-2 / +-5 overall, by split (h50 dev vs held-out), and by agent; paired sign test on +-2 vs --ref,
on the trajectories both predicted (and counting a missing prediction as a miss).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_subgoals import load_judge  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_pred(spec: str) -> dict[tuple[str, str], float]:
  kind, _, rest = spec.partition(":")
  out: dict[tuple[str, str], float] = {}
  if kind == "judge":
    return {k: float(v) for k, v in load_judge(rest).items() if v is not None}
  if kind == "idea":
    v, m = rest.split("/")
    files = (ROOT / "data/ideas" / v / m / "s1").rglob("*.json")
  elif kind == "constraints":
    files = (ROOT / "data/constraints" / rest / "check").rglob("*.json")
  elif kind == "stepcheck":
    v, _, mode = rest.partition(":")
    for f in (ROOT / "data/stepcheck" / v).rglob("*.json"):
      r = json.loads(f.read_text(encoding="utf-8"))
      if "verdicts" not in r:
        continue
      steps = [x["step"] for x in r["verdicts"] if x["match"] == "no" and (mode != "harm" or x["harm"])]
      if steps:
        out[(r["agent"], r["task_id"])] = float(steps[0])
    return out
  elif kind == "const":
    return {}
  else:
    raise ValueError(spec)
  for f in files:
    r = json.loads(f.read_text(encoding="utf-8"))
    s = r["out"].get("root_step")
    if s is not None:
      out[(r["agent"], r["task_id"])] = float(s)
  return out


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("specs", nargs="+")
  ap.add_argument("--ref", required=True)
  ap.add_argument("--index", default=str(ROOT / "data/index"))
  ap.add_argument("--out", default=str(ROOT / "data/metrics/compare"))
  ap.add_argument("--shared", action="store_true", help="score only trajectories every predictor covers")
  args = ap.parse_args()
  lab = pd.read_parquet(Path(args.index) / "episodes.parquet")
  lab = lab[(lab.source == "cuadebug") & lab.root_error_step.notna()].set_index(["agent", "task_id"])
  dev = {(r["agent"], r["task_id"]) for r in json.loads((Path(args.index) / "h50.json").read_text())["rows"]}
  keys = list(lab.index)
  gold = lab.root_error_step.astype(float)
  split = pd.Series(["dev" if k in dev else "test" for k in keys], index=lab.index)
  specs = [args.ref] + [s for s in args.specs if s != args.ref]
  hits, rows, per = {}, [], []
  if args.shared:  # restrict to the intersection of every predictor's coverage
    have = pd.Series(True, index=lab.index)
    for s in specs:
      if not s.startswith("const:"):
        d = load_pred(s)
        have &= pd.Series([k in d for k in keys], index=lab.index)
    lab, keys = lab[have], [k for k, h in zip(keys, have) if h]
    gold = lab.root_error_step.astype(float)
    split = split[have]
  for s in specs:
    if s.startswith("const:"):
      k = float(s.split(":")[1])
      p = pd.Series([min(k, n) for n in lab.n_steps], index=lab.index, dtype=float)
    else:
      d = load_pred(s)
      p = pd.Series([d.get(k) for k in keys], index=lab.index, dtype=float)
    err = (p - gold).abs()
    hits[s] = (err <= 2, p.notna())
    row = {"predictor": s, "have": int(p.notna().sum()), "exact": (err == 0).mean(), "pm1": (err <= 1).mean(),
           "pm2": (err <= 2).mean(), "pm5": (err <= 5).mean()}
    for sp in ("dev", "test"):
      row[f"pm2_{sp}"] = (err[split == sp] <= 2).mean()
    for a in sorted(lab.index.get_level_values(0).unique()):
      m = lab.index.get_level_values(0) == a
      row[f"pm2_{a.split('-')[0]}"] = (err[m] <= 2).mean()
    rows.append(row)
  rh, rhave = hits[args.ref]
  for row in rows:
    h, have = hits[row["predictor"]]
    if row["predictor"] == args.ref:
      row["vs_ref"] = "ref"
      continue
    both = have & rhave
    a, b = int((h & ~rh & both).sum()), int((rh & ~h & both).sum())
    row["vs_ref"] = f"+{a}/-{b} p={binomtest(a, a + b).pvalue:.2f}" if a + b else "same"
  t = pd.DataFrame(rows)
  num = [c for c in t.columns if c not in {"predictor", "have", "vs_ref"}]
  t[num] = t[num].round(3)
  Path(args.out).mkdir(parents=True, exist_ok=True)
  t.to_csv(Path(args.out) / "comparison.csv", index=False)
  pd.set_option("display.width", 250)
  print(f"{len(keys)} labelled failures (dev {int((split == 'dev').sum())}, held-out {int((split == 'test').sum())}); "
        f"sign test on +-2 vs {args.ref}, on trajectories both predicted; uncorrected")
  print(t.to_string(index=False))


if __name__ == "__main__":
  main()
