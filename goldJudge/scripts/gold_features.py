#!/usr/bin/env python
"""Per-trajectory properties of each labelled failure and of the gold it was judged with.

One row per CUADebug failure (agent, task_id) → ``data/analysis/gold_features.parquet``.
Input to ``scripts/gold_properties.py`` (which gold properties predict a gold effect).

Columns
  failure : n_steps, human_step, root_rel (= human_step / n_steps), human_tag, domain, agent
  G1      : h_steps (OSWorld-Human step groups), h_ratio (= h_steps / n_steps),
            h_overlap (share of human-step word types that also occur in the failed run's actions/intents)
  G2      : gold_key, gold_agent, g_steps, g_ratio, same_agent (gold agent == failing agent),
            vis_d (per failed step: min dHash distance to any gold screenshot, in bits of 256),
            vis_d0 (distance of the two first screenshots — same initial state expected)

The visual distances are stored raw so the divergence threshold can be chosen in the
analysis step. Screenshots are only read, never copied.

  python scripts/gold_features.py            # ~10 min on 1 CPU (hashes ~7k screenshots)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_judge import _load_gold  # noqa: E402 — same gold choice as the judge runs

from goldjudge.loaders import cuadebug  # noqa: E402
from goldjudge.loaders.osworld_meta import OSWorldMeta  # noqa: E402

HASH = 16  # dHash grid → 256 bits
_WORD = re.compile(r"[a-z][a-z0-9_.-]{2,}")
_STOP = set("the and for with then into from that this click press open select type enter text using on".split())


def dhash(path: str) -> np.ndarray | None:
  try:
    with Image.open(path) as im:
      g = im.convert("L").resize((HASH + 1, HASH), Image.BILINEAR)
  except Exception:  # noqa: BLE001 — a missing/corrupt screenshot is just a missing feature
    return None
  a = np.asarray(g, dtype=np.int16)
  return (a[:, 1:] > a[:, :-1]).ravel()


def hashes(traj) -> list[np.ndarray | None]:
  return [dhash(s.screenshot_path) if s.screenshot_path else None for s in traj.steps]


def words(text: str) -> set[str]:
  return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--index", type=Path, default=Path("data/index"))
  ap.add_argument("--out", type=Path, default=Path("data/analysis/gold_features.parquet"))
  args = ap.parse_args()

  gold_df = pd.read_parquet(args.index / "gold.parquet")
  osw = OSWorldMeta()
  rows = []
  trajs = cuadebug.load_all()
  for i, t in enumerate(trajs, 1):
    ann = t.annotation
    row = {"agent": t.agent, "task_id": t.task_id, "domain": t.domain, "n_steps": len(t.steps),
           "human_step": ann.root_error_step if ann else None, "human_tag": ann.taxonomy_tag if ann else None}
    row["root_rel"] = row["human_step"] / row["n_steps"] if row["human_step"] else None
    agent_words = words(" ".join(f"{s.action_raw} {s.intent}" for s in t.steps))

    human = osw.load_human(t.domain, t.task_id)
    if human:
      groups = human.grouped_action or [[a] for a in human.single_action]
      hw = words(" ".join(" ".join(g) for g in groups))
      row.update(h_steps=len(groups), h_ratio=len(groups) / len(t.steps),
                 h_overlap=len(hw & agent_words) / len(hw) if hw else None)

    gold = _load_gold(gold_df, t.task_id, "G2")
    if gold:
      fh, gh = hashes(t), [h for h in hashes(gold) if h is not None]
      vis = [int(min(np.count_nonzero(h != g) for g in gh)) if (h is not None and gh) else None for h in fh]
      row.update(gold_key=f"osworld_verified/{gold.package}/{gold.domain}/{gold.task_id}", gold_agent=gold.agent,
                 g_steps=len(gold.steps), g_ratio=len(gold.steps) / len(t.steps), same_agent=gold.agent == t.agent,
                 vis_d=vis, vis_d0=vis[0] if vis else None)
    rows.append(row)
    if i % 20 == 0:
      print(f"{i}/{len(trajs)}", flush=True)

  df = pd.DataFrame(rows)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  df.to_parquet(args.out, index=False)
  print(f"wrote {args.out}: {len(df)} rows, G1 {df.h_steps.notna().sum()}, G2 {df.g_steps.notna().sum()}")


if __name__ == "__main__":
  main()
