#!/usr/bin/env python3
"""Cost and harness stats per (judge_version, condition) from logged token usage.

Prices are $ per million tokens (base input / output). Anthropic: cache writes billed at 1.25x
input, cache reads at 0.1x. OpenAI-compatible (incl. vLLM): cached prompt tokens at 0.1x.

Usage:
  python scripts/cost_report.py --price-in 1 --price-out 5 haiku45-h50-fixed-v0.3 haiku45-h50-od-v0.3
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("versions", nargs="+")
  ap.add_argument("--price-in", type=float, required=True)
  ap.add_argument("--price-out", type=float, required=True)
  ap.add_argument("--judge-dir", default="data/judge")
  args = ap.parse_args()
  pin, pout = args.price_in, args.price_out
  grand = 0.0
  for v in args.versions:
    agg: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for f in sorted(glob.glob(f"{args.judge_dir}/{v}/G*/*.jsonl")):
      for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        u, a = r["usage"], agg[r["condition"]]
        a["n"] += 1
        if "input_tokens" in u:  # Anthropic usage
          a["usd"] += (u["input_tokens"] * pin + u["cache_creation_input_tokens"] * pin * 1.25
                       + u["cache_read_input_tokens"] * pin * 0.1 + u["completion_tokens"] * pout) / 1e6
        else:  # OpenAI-compatible usage: prompt_tokens includes cached_tokens (billed at 0.1x)
          cached = u.get("cached_tokens", 0)
          a["usd"] += ((u["prompt_tokens"] - cached) * pin + cached * pin * 0.1 + u["completion_tokens"] * pout) / 1e6
        a["turns"] += r.get("n_turns", 1)
        reqs = r.get("screenshot_requests") or []
        a["extra_imgs"] += sum(len(q["granted"]) for q in reqs)
        a["wasted_reqs"] += sum(1 for q in reqs if not q["granted"])
        a["root_seen"] += int(r["human"]["root_error_step"]) in set(r.get("viewed_agent_steps") or [])
        a["parse_ok"] += r["verdict"]["parse_ok"]
        a["truncated"] += u.get("stop_reason") == "max_tokens"
        a["latency"] += r["latency_s"]
    tot = 0.0
    for c, a in sorted(agg.items()):
      n = a["n"]
      tot += a["usd"]
      print(f"{v} {c}: n={n:.0f} ${a['usd']:.3f} (${a['usd'] / n:.3f}/traj) turns={a['turns'] / n:.1f} "
            f"extra_imgs={a['extra_imgs'] / n:.1f} wasted_reqs={a['wasted_reqs']:.0f} root_seen={a['root_seen']:.0f}/{n:.0f} "
            f"parse_ok={a['parse_ok']:.0f}/{n:.0f} truncated={a['truncated']:.0f} latency={a['latency'] / n:.0f}s")
    print(f"{v} TOTAL ${tot:.3f}")
    grand += tot
  print(f"GRAND TOTAL ${grand:.3f}")


if __name__ == "__main__":
  main()
