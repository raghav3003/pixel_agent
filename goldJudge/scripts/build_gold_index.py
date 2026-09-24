#!/usr/bin/env python3
"""Build the gold-trajectory index: for every OSWorld task, which reference paths exist.

Reads ``data/index/episodes.parquet`` (from build_index.py) and the OSWorld sources.
Writes ``data/index/gold.parquet`` with one row per (task, condition, candidate):

  task_id, domain, cond (G1|G2), agent, package, run, n_steps, dir, rank

G1 = OSWorld-Human text steps (always present; dir = human json path, n_steps = #single actions).
G2 = evaluator-verified successful AI trajectory (result >= 1.0), ranked by
     agent preference then fewest steps (shortest successful path first).

Also prints coverage over the CUADebug failure set.

Usage:
  python scripts/build_gold_index.py --index data/index
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from goldjudge.loaders.osworld_meta import OSWorldMeta

# Preference order for G2 gold sources (stronger / cleaner agents first).
AGENT_PREF = ["claude-opus-4-7", "claude-sonnet-4-6", "kimi-k2.6", "claude-sonnet-4-5", "opencua-32b", "opencua-7b", "opencua-a3b"]


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--index", type=Path, default=Path("data/index"))
  ap.add_argument("--osworld-src", type=Path, default=None)
  args = ap.parse_args()

  eps = pd.read_parquet(args.index / "episodes.parquet")
  meta = OSWorldMeta(args.osworld_src) if args.osworld_src else OSWorldMeta()
  tasks = [(d, t) for d, ids in meta.task_list("test_nogdrive").items() for t in ids]

  rows: list[dict] = []
  for domain, task_id in tasks:
    human = meta.load_human(domain, task_id)
    if human is not None:
      rows.append({"task_id": task_id, "domain": domain, "cond": "G1", "agent": "osworld-human", "package": "osworld-human",
                   "run": "", "n_steps": len(human.single_action), "dir": str(meta.human_path(domain, task_id)), "rank": 0})
    succ = eps[(eps.task_id == task_id) & (eps.success) & (eps.source == "osworld_verified")].copy()
    if len(succ):
      succ["pref"] = succ.agent.map(lambda a: AGENT_PREF.index(a) if a in AGENT_PREF else len(AGENT_PREF))
      succ = succ.sort_values(["pref", "n_steps"]).reset_index(drop=True)
      for rank, r in succ.iterrows():
        rows.append({"task_id": task_id, "domain": domain, "cond": "G2", "agent": r.agent, "package": r.package,
                     "run": r.run, "n_steps": int(r.n_steps), "dir": r.dir, "rank": int(rank)})

  gold = pd.DataFrame(rows)
  gold.to_parquet(args.index / "gold.parquet", index=False)
  gold.to_csv(args.index / "gold.csv", index=False)

  g2 = gold[gold.cond == "G2"]
  print(f"G1 tasks: {gold[gold.cond == 'G1'].task_id.nunique()} | G2 tasks: {g2.task_id.nunique()} / {len(tasks)}")
  print("G2 best-agent distribution:", g2[g2["rank"] == 0].agent.value_counts().to_dict())

  failed = eps[(eps.source == "cuadebug")]
  for m, grp in failed.groupby("agent"):
    ids = set(grp.task_id)
    print(f"CUADebug {m}: {len(ids)} tasks | with G2 gold: {len(ids & set(g2.task_id))}")


if __name__ == "__main__":
  main()
