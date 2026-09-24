#!/usr/bin/env python3
"""Select a stratified subset of labelled CUADebug failures for a judge experiment.

Default (``long15``): long trajectories where on-demand screenshots can matter.
Per agent, ``--per-agent`` trajectories with G2 gold and ``--min-steps``..``--max-steps``
steps; at least ``--min-root-missing`` of them have the human root step *outside* the
fixed 16-image sample (the case on-demand screenshots are meant to fix). Excludes tasks
in ``--exclude`` subsets (e.g. pilot15). Deterministic for a given ``--seed``.
``--include`` forces the rows of existing subsets in (nested subsets, e.g. long15 ⊂ h50);
``--agent-counts`` overrides ``--per-agent`` per agent.

Writes the ``{task_ids, rows}`` JSON that ``run_judge.py --pilot`` reads.

Usage:
  python scripts/select_subset.py --out data/index/long15.json --exclude data/index/pilot15.json
  python scripts/select_subset.py --out data/index/h50.json --exclude data/index/pilot15.json \
      --include data/index/long15.json --min-steps 1 --max-steps 100 \
      --agent-counts claude-sonnet-4-5=20 gemini-2.5-pro=15 qwen3.5-vl=15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from goldjudge.judge.images import select_steps_for_images


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--index", type=Path, default=Path("data/index"))
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--exclude", type=Path, nargs="*", default=[])
  ap.add_argument("--include", type=Path, nargs="*", default=[], help="subsets whose rows are always kept")
  ap.add_argument("--agent-counts", nargs="*", default=[], help="per-agent sizes, e.g. qwen3.5-vl=15")
  ap.add_argument("--per-agent", type=int, default=5)
  ap.add_argument("--min-steps", type=int, default=17)
  ap.add_argument("--max-steps", type=int, default=60)
  ap.add_argument("--min-root-missing", type=int, default=2)
  ap.add_argument("--fixed-images", type=int, default=16, help="agent image budget of the fixed harness")
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()

  eps = pd.read_parquet(args.index / "episodes.parquet")
  gold = pd.read_parquet(args.index / "gold.parquet")
  g2 = gold[(gold.cond == "G2") & (gold["rank"] == 0)].set_index("task_id")
  excluded = {t for p in args.exclude for t in json.loads(p.read_text())["task_ids"]}
  forced = {(r["agent"], r["task_id"]) for p in args.include for r in json.loads(p.read_text())["rows"]}
  counts = {k: int(v) for k, v in (a.split("=", 1) for a in args.agent_counts)}

  c = eps[(eps.source == "cuadebug") & (~eps.success) & eps.has_annotation].copy()
  c["forced"] = [(a, t) in forced for a, t in zip(c.agent, c.task_id)]
  c = c[c.task_id.isin(g2.index) & ~c.task_id.isin(excluded) & (c.forced | c.n_steps.between(args.min_steps, args.max_steps))]
  missing = forced - set(zip(c.agent, c.task_id))
  if missing:
    raise SystemExit(f"--include rows not eligible (excluded or no G2 gold): {sorted(missing)}")
  c["root_error_step"] = c.root_error_step.astype(int)
  c["root_in_fixed"] = [r in set(select_steps_for_images(n, args.fixed_images)) for r, n in zip(c.root_error_step, c.n_steps)]
  c["gold_steps"] = c.task_id.map(g2.n_steps).astype(int)
  c["gold_agent"] = c.task_id.map(g2.agent)

  picked = []
  for agent, grp in c.groupby("agent"):
    want = counts.get(agent, args.per_agent)
    grp = grp.sample(frac=1.0, random_state=args.seed).drop_duplicates("task_id")
    keep = grp[grp.forced]
    free = grp[~grp.forced]
    n_miss = max(0, args.min_root_missing - int((~keep.root_in_fixed).sum()))
    miss = free[~free.root_in_fixed].head(n_miss)
    rest = free.drop(miss.index).head(max(0, want - len(keep) - len(miss)))
    sel = pd.concat([keep, miss, rest])
    if len(sel) < want:
      print(f"WARNING: {agent}: only {len(sel)} eligible trajectories")
    picked.append(sel)
  sub = pd.concat(picked).sort_values(["agent", "domain", "task_id"])

  cols = ["agent", "domain", "task_id", "n_steps", "taxonomy_tag", "taxonomy_family", "root_error_step",
          "root_in_fixed", "gold_agent", "gold_steps"]
  rows = [{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()} for r in sub[cols].to_dict("records")]
  out = {
    "selection": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k not in ("exclude", "include")}
    | {"exclude": [str(p) for p in args.exclude], "include": [str(p) for p in args.include], "eligible_pool": int(len(c))},
    "task_ids": sorted(set(sub.task_id)),
    "rows": rows,
  }
  args.out.write_text(json.dumps(out, indent=1))
  pd.set_option("display.width", 200)
  print(sub[cols].to_string(index=False))
  print(f"\n{len(sub)} trajectories from a pool of {len(c)}; root step outside fixed-{args.fixed_images} sample: "
        f"{int((~sub.root_in_fixed).sum())}; wrote {args.out}")


if __name__ == "__main__":
  main()
