"""Extract replayable text action logs for every indexed episode and every OSWorld-Human reference.

Deterministic (no model calls). One JSON per trajectory:

  data/actions/<agent>/<run-slug>/<domain>/<task_id>.json     # agent runs (CUADebug + OSWorld-Verified)
  data/actions/osworld-human/<domain>/<task_id>.json          # human reference steps

plus ``data/actions/coverage.csv`` (ops per agent, how many fell back to CODE / OTHER / warnings).

  python scripts/extract_actions.py                       # everything (~2.4k logs)
  python scripts/extract_actions.py --pilot data/index/h50.json --with-gold
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from goldjudge.actions import ActionLog, action_log, human_action_log  # noqa: E402
from goldjudge.loaders import cuadebug, osworld_verified  # noqa: E402
from goldjudge.loaders.osworld_meta import OSWorldMeta  # noqa: E402
from goldjudge.schema import Trajectory  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_episode(r) -> Trajectory:
  if r.source == "cuadebug":
    return cuadebug.load_trajectory(Path(r.dir), r.package)
  return osworld_verified.load_trajectory(Path(r.dir), package=r.package, run=r.run, agent=r.agent, horizon=r.horizon)


def slug(s: str) -> str:
  return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")[:80] or "_"


def log_path(out: Path, r) -> Path:
  run = slug(f"{r.source}-{r.package}-{r.run}" if r.run else f"{r.source}-{r.package}")
  return out / r.agent / run / r.domain / f"{r.task_id}.json"


def human_path(out: Path, domain: str, task_id: str) -> Path:
  return out / "osworld-human" / domain / f"{task_id}.json"


def write(path: Path, log: ActionLog) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(log.model_dump_json(indent=1), encoding="utf-8")


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--index", default=str(ROOT / "data/index"))
  ap.add_argument("--out", default=str(ROOT / "data/actions"))
  ap.add_argument("--pilot", help="pilot JSON with rows [{agent, task_id}] (e.g. data/index/h50.json)")
  ap.add_argument("--with-gold", action="store_true", help="with --pilot: also every episode of the pilot's tasks")
  ap.add_argument("--force", action="store_true", help="rewrite logs that already exist")
  args = ap.parse_args()

  out = Path(args.out)
  eps = pd.read_parquet(Path(args.index) / "episodes.parquet")
  if args.pilot:
    keys = {(r["agent"], r["task_id"]) for r in json.loads(Path(args.pilot).read_text())["rows"]}
    if args.with_gold:
      eps = eps[eps.task_id.isin({t for _, t in keys})]
    else:
      eps = eps[(eps.source == "cuadebug") & [(a, t) in keys for a, t in zip(eps.agent, eps.task_id)]]

  cov: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
  for r in eps.itertuples():
    p = log_path(out, r)
    if p.exists() and not args.force:
      log = ActionLog.model_validate_json(p.read_text(encoding="utf-8"))
    else:
      log = action_log(load_episode(r))
      log.extra = {"package": r.package, "run": r.run, "dir": r.dir}
      write(p, log)
    c = cov[r.agent]
    c["trajectories"] += 1
    for s in log.steps:
      c["steps"] += 1
      c["empty_steps"] += not s.ops
      for o in s.ops:
        c["ops"] += 1
        c[f"op_{o.op}"] += 1
        c["warn"] += bool(o.warn)

  osw = OSWorldMeta()
  for domain, ids in osw.task_list("test_all").items():
    for tid in ids:
      if args.pilot and tid not in set(eps.task_id):
        continue
      h = osw.load_human(domain, tid)
      if h is None:
        continue
      p = human_path(out, domain, tid)
      log = human_action_log(h)
      if args.force or not p.exists():
        write(p, log)
      c = cov["osworld-human"]
      c["trajectories"] += 1
      for s in log.steps:
        c["steps"] += 1
        for o in s.ops:
          c["ops"] += 1
          c[f"op_{o.op}"] += 1

  df = pd.DataFrame(cov).T.fillna(0).astype(int)
  cols = ["trajectories", "steps", "empty_steps", "ops", "op_CODE", "op_OTHER", "warn"]
  df = df[[c for c in cols if c in df] + sorted(c for c in df if c not in cols)]
  out.mkdir(parents=True, exist_ok=True)
  df.to_csv(out / "coverage.csv")
  print(df[[c for c in cols if c in df]].to_string())


if __name__ == "__main__":
  main()
