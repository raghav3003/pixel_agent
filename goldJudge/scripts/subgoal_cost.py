#!/usr/bin/env python3
"""Token usage and cost per stage of a subgoal-pipeline version (from the usage logged in its outputs).

  python scripts/subgoal_cost.py gpt6luna-sg-v0.2 --price-in 0.10 --price-out 0.50
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE_DIRS = {"captions": "caption", "extract": "extract", "sets": "consolidate", "tracks/gold": "verify (gold tracks)",
              "tracks/fail": "track", "localize": "localize", "agent_goals": "agent goals"}  # sets_verified copies sets' usage: not counted


def usages(obj):
  if isinstance(obj, dict):
    if "prompt_tokens" in obj:
      yield obj
    else:
      for v in obj.values():
        yield from usages(v)
  elif isinstance(obj, list):
    for v in obj:
      yield from usages(v)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("version")
  ap.add_argument("--out", default=str(ROOT / "data/subgoals"))
  ap.add_argument("--price-in", type=float, default=0.0, help="$ per M input tokens (cached at 0.1x)")
  ap.add_argument("--price-out", type=float, default=0.0)
  args = ap.parse_args()
  d = Path(args.out) / args.version
  tot = defaultdict(lambda: defaultdict(float))
  for sub, stage in STAGE_DIRS.items():
    for f in (d / sub).rglob("*.json*"):
      lines = f.read_text(encoding="utf-8").splitlines() if f.suffix == ".jsonl" else [f.read_text(encoding="utf-8")]
      for line in lines:
        if not line.strip():
          continue
        for u in usages(json.loads(line)):
          t = tot[stage]
          t["calls"] += 1
          t["in"] += u.get("prompt_tokens") or 0
          t["cached"] += u.get("cached_tokens") or 0
          t["out"] += u.get("completion_tokens") or 0
  grand = 0.0
  print(f"{'stage':22s} {'calls':>7s} {'in_tok':>12s} {'out_tok':>10s} {'$':>8s}")
  for stage, t in tot.items():
    cost = ((t["in"] - t["cached"]) * args.price_in + t["cached"] * args.price_in * 0.1 + t["out"] * args.price_out) / 1e6
    grand += cost
    print(f"{stage:22s} {int(t['calls']):7d} {int(t['in']):12d} {int(t['out']):10d} {cost:8.2f}")
  print(f"{'total':22s} {'':7s} {'':12s} {'':10s} {grand:8.2f}")


if __name__ == "__main__":
  main()
