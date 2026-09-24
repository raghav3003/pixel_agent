#!/usr/bin/env python3
"""Index every episode across sources into one table.

Writes ``<out>/episodes.parquet`` (+ ``.csv``) with one row per episode:
source, package, run, agent, domain, task_id, result, success, n_steps, horizon,
has_annotation, root_error_step, taxonomy_tag, dir.

Usage:
  python scripts/build_index.py --out data/index
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from goldjudge.loaders import cuadebug, osworld_verified


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", type=Path, default=Path("data/index"))
  ap.add_argument("--cuadebug-root", type=Path, default=cuadebug.DEFAULT_ROOT)
  ap.add_argument("--oswv-root", type=Path, default=osworld_verified.DEFAULT_ROOT)
  args = ap.parse_args()

  rows: list[dict] = []
  for archive, task_dir in tqdm(list(cuadebug.iter_task_dirs(args.cuadebug_root)), desc="cuadebug"):
    t = cuadebug.load_trajectory(task_dir, archive)
    ann = t.annotation
    rows.append(
      {
        "source": t.source, "package": t.package, "run": "", "agent": t.agent,
        "domain": t.domain, "task_id": t.task_id, "result": t.result, "success": t.success,
        "n_steps": t.n_steps, "horizon": t.horizon, "has_annotation": ann is not None,
        "root_error_step": ann.root_error_step if ann else None,
        "taxonomy_tag": ann.taxonomy_tag if ann else None,
        "taxonomy_family": ann.taxonomy_family if ann else None, "dir": t.dir,
      }
    )
  if args.oswv_root.exists():
    for ep in tqdm(list(osworld_verified.iter_episode_dirs(args.oswv_root)), desc="osworld_verified"):
      t = osworld_verified.load_trajectory(
        ep["task_dir"], package=ep["package"], run=ep["run"], agent=ep["agent"], horizon=ep["horizon"]
      )
      rows.append(
        {
          "source": t.source, "package": t.package, "run": ep["run"], "agent": t.agent,
          "domain": t.domain, "task_id": t.task_id, "result": t.result, "success": t.success,
          "n_steps": t.n_steps, "horizon": t.horizon, "has_annotation": False,
          "root_error_step": None, "taxonomy_tag": None, "taxonomy_family": None, "dir": t.dir,
        }
      )

  df = pd.DataFrame(rows)
  args.out.mkdir(parents=True, exist_ok=True)
  df.to_parquet(args.out / "episodes.parquet", index=False)
  df.to_csv(args.out / "episodes.csv", index=False)
  print(f"wrote {len(df)} episodes -> {args.out}")
  print(df.groupby(["source", "agent", "run"]).agg(n=("task_id", "size"), succ=("success", "sum"), steps=("n_steps", "mean")).to_string())


if __name__ == "__main__":
  main()
