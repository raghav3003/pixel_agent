"""Subgoal-based error analysis: caption -> extract -> consolidate -> verify -> track -> localize -> score.

See ``docs/subgoals.md`` for the method. Every stage is resumable (outputs that exist are skipped) and
versioned (``--version``; never overwrite another version's outputs):

  data/subgoals/<V>/captions/<agent>/<run>/<domain>/<task>.json   per-step target/effect (+ captioned action log)
  data/subgoals/<V>/extract/<task>/<view>.json                     candidate subgoals per evidence view
  data/subgoals/<V>/sets/<task>.json                               consolidated SubgoalSet
  data/subgoals/<V>/tracks/{gold,fail}/<agent>/<run>/<domain>/<task>.json
  data/subgoals/<V>/sets_verified/<task>.json                      after counterexample checks on successes
  data/subgoals/<V>/localize/<agent>.jsonl                         rule window + LLM root step per failure
  data/subgoals/<V>/metrics/                                       scores vs CUADebug human root steps

  python scripts/subgoals.py all --version gpt6luna-sg-v0.2 --env-file config/openai.env \
      --reasoning-effort none --think-effort medium --workers 8 --pilot data/index/smoke5.json
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_actions import human_path, load_episode, log_path, slug  # noqa: E402
from goldjudge.actions import ActionLog, action_log, human_action_log  # noqa: E402
from goldjudge.judge.client import JudgeClient, JudgeClientConfig, load_env_file  # noqa: E402
from goldjudge.loaders.osworld_meta import OSWorldMeta  # noqa: E402
from goldjudge.subgoals import core  # noqa: E402
from goldjudge.subgoals.prompts import (CAPTION_SYSTEM, CHECK_SYSTEM, CONSOLIDATE_SYSTEM, EXTRACT_SYSTEM,  # noqa: E402
                                        LOCALIZE_SYSTEM, SUBGOAL_PROMPT_VERSION)
from goldjudge.subgoals.schema import StepCaption, SubgoalSet, Track  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STAGES = ["caption", "extract", "consolidate", "verify", "track", "localize", "score"]
VIEW_NAMES = {"task": "instruction + initial setup + evaluator only", "human": "+ OSWorld-Human written steps",
              "ai": "+ one successful AI run (captioned actions)"}
_lock = threading.Lock()


# --- plumbing ----------------------------------------------------------------------------------


class Ctx:
  def __init__(self, args: argparse.Namespace):
    self.args = args
    self.out = Path(args.out) / args.version
    self.osw = OSWorldMeta()
    eps = pd.read_parquet(Path(args.index) / "episodes.parquet")
    fails = eps[(eps.source == "cuadebug") & (~eps.success)]
    if args.pilot:
      keys = {(r["agent"], r["task_id"]) for r in json.loads(Path(args.pilot).read_text())["rows"]}
      fails = fails[[(a, t) in keys for a, t in zip(fails.agent, fails.task_id)]]
    if args.limit:
      fails = fails.sort_values(["agent", "task_id"]).head(args.limit)
    self.fails = fails.sort_values(["agent", "domain", "task_id"])
    self.tasks = sorted(set(self.fails.task_id))
    golds = eps[(eps.source == "osworld_verified") & (eps.success) & (eps.task_id.isin(self.tasks))]
    self.golds = golds.sort_values(["task_id", "agent", "run"])
    self.gold_rank = pd.read_parquet(Path(args.index) / "gold.parquet")
    self.domain = dict(zip(self.fails.task_id, self.fails.domain))
    self.fast = self.think = None
    if not args.dry_run:
      load_env_file(args.env_file)
      base = dict(max_tokens=args.max_tokens)
      fcfg = JudgeClientConfig(reasoning_effort=args.reasoning_effort, **base)
      tcfg = JudgeClientConfig(reasoning_effort=args.think_effort or args.reasoning_effort,
                               **{**base, "max_tokens": args.think_max_tokens if args.think_effort else args.max_tokens})
      for c in (fcfg, tcfg):
        if args.provider:
          c.provider = args.provider
        if args.model:
          c.model = args.model
      self.fast, self.think = JudgeClient(fcfg), JudgeClient(tcfg)
    self.meta_cache: dict[str, Any] = {}

  def meta(self, task_id: str):
    if task_id not in self.meta_cache:
      self.meta_cache[task_id] = self.osw.load_task(self.domain.get(task_id) or self.osw.domain_of(task_id), task_id)
    return self.meta_cache[task_id]

  def ep_path(self, kind: str, r) -> Path:
    return log_path(self.out / kind, r)


def call(client: JudgeClient, system: str, content: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
  resp = client.chat(system, content)
  return resp.text, {**(resp.usage or {}), "latency_s": round(resp.latency_s, 2), "model": resp.model}


def write_json(path: Path, obj: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_suffix(".tmp")
  tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
  tmp.replace(path)


def run_units(name: str, units: list[Any], fn: Callable[[Any], str | None], workers: int) -> int:
  """Run fn over units in a thread pool; fn returns None when skipped. Returns the failure count."""
  t0, done, skipped, failed = time.time(), 0, 0, 0
  print(f"[{name}] {len(units)} units, workers={workers}", flush=True)
  with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
    futs = {ex.submit(fn, u): u for u in units}
    for f in as_completed(futs):
      try:
        r = f.result()
        if r is None:
          skipped += 1
        else:
          done += 1
      except Exception as exc:  # noqa: BLE001 — keep going; resumable
        failed += 1
        status = getattr(exc, "status_code", None)
        print(f"FAILED [{name}] {_unit_name(futs[f])}: {type(exc).__name__}: {exc}", flush=True)
        if status in (401, 403):  # bad/expired key: every other unit would fail the same way
          for g in futs:
            g.cancel()
          raise SystemExit(f"[{name}] authentication error ({status}) — fix the API key and rerun") from exc
        traceback.print_exc(limit=3)
      n = done + skipped + failed
      if n % 20 == 0 or n == len(units):
        print(f"[{name}] {n}/{len(units)} (done {done}, skipped {skipped}, failed {failed}) {time.time() - t0:.0f}s", flush=True)
  return failed


def _unit_name(u: Any) -> str:
  if hasattr(u, "agent") and hasattr(u, "task_id"):
    return f"{u.agent}/{u.domain}/{u.task_id}"
  return str(u)


def load_log(r) -> ActionLog:
  p = log_path(ROOT / "data/actions", r)
  if p.exists():
    return ActionLog.model_validate_json(p.read_text(encoding="utf-8"))
  return action_log(load_episode(r))


def captioned_log(ctx: Ctx, r) -> ActionLog:
  p = ctx.ep_path("captions", r)
  log = load_log(r)
  if p.exists():
    caps = {c["step"]: StepCaption(**c) for c in json.loads(p.read_text(encoding="utf-8"))["captions"]}
    log = core.apply_captions(log, caps)
  return log


# --- stages ------------------------------------------------------------------------------------


def stage_caption(ctx: Ctx) -> int:
  rows = list(ctx.fails.itertuples()) + list(ctx.golds.itertuples())

  def one(r) -> str | None:
    p = ctx.ep_path("captions", r)
    if p.exists():
      return None
    traj, log, meta = load_episode(r), load_log(r), ctx.meta(r.task_id)
    caps, usage, raw = [], [], []
    for win in core.caption_windows(traj.n_steps, ctx.args.caption_window):
      content = core.caption_content(meta, traj, log, win, max_width=ctx.args.max_width)
      text, u = call(ctx.fast, CAPTION_SYSTEM, content)
      got = core.parse_captions(text, win)
      caps += got
      usage.append(u)
      if len(got) < len(win):
        raw.append({"window": win, "text": text[:4000]})
    write_json(p, {"key": traj.key, "version": ctx.args.version, "prompt_version": SUBGOAL_PROMPT_VERSION,
                   "captions": [c.model_dump() for c in caps], "n_steps": traj.n_steps, "usage": usage, "unparsed": raw,
                   "action_log": core.apply_captions(log, {c.step: c for c in caps}).model_dump()})
    return "ok"

  return run_units("caption", rows, one, ctx.args.workers)


def _ai_gold_row(ctx: Ctx, task_id: str):
  g = ctx.gold_rank[(ctx.gold_rank.task_id == task_id) & (ctx.gold_rank.cond == "G2")].sort_values("rank")
  for gr in g.itertuples():
    m = ctx.golds[(ctx.golds.task_id == task_id) & (ctx.golds.agent == gr.agent) & (ctx.golds.run == gr.run)]
    if not m.empty:
      return m.iloc[0]
  return ctx.golds[ctx.golds.task_id == task_id].iloc[0] if (ctx.golds.task_id == task_id).any() else None


def stage_extract(ctx: Ctx) -> int:
  units = []
  for t in ctx.tasks:
    meta = ctx.meta(t)
    views = ["task"]
    if meta.human is not None:
      views.append("human")
    if _ai_gold_row(ctx, t) is not None:
      views.append("ai")
    units += [(t, v) for v in views]

  def one(u) -> str | None:
    t, view = u
    p = ctx.out / "extract" / t / f"{view}.json"
    if p.exists():
      return None
    meta = ctx.meta(t)
    human = human_action_log(meta.human) if view == "human" else None
    gold_row = _ai_gold_row(ctx, t) if view == "ai" else None
    ai = captioned_log(ctx, gold_row) if gold_row is not None else None
    content = core.extract_content(core.task_text(meta, ctx.osw), view, human_log=human, ai_log=ai)
    text, usage = call(ctx.think, EXTRACT_SYSTEM, content)
    sgs, notes = core.parse_subgoals(text)
    write_json(p, {"task_id": t, "view": view, "version": ctx.args.version, "prompt_version": SUBGOAL_PROMPT_VERSION,
                   "gold_key": ai.key if ai else None, "subgoals": [s.model_dump() for s in sgs], "notes": notes,
                   "usage": usage, "raw": text if not sgs else text[:6000]})
    if not sgs:
      raise RuntimeError(f"no subgoals parsed for {t}/{view}")
    return "ok"

  return run_units("extract", units, one, ctx.args.workers)


def stage_consolidate(ctx: Ctx) -> int:
  def one(t) -> str | None:
    p = ctx.out / "sets" / f"{t}.json"
    if p.exists():
      return None
    files = sorted((ctx.out / "extract" / t).glob("*.json"), key=lambda f: list(VIEW_NAMES).index(f.stem))
    if not files:
      raise RuntimeError(f"no extraction for {t}")
    meta = ctx.meta(t)
    cands, names, seen = {}, {}, []
    from goldjudge.subgoals.schema import Subgoal
    for letter, f in zip("ABCDEFG", files):
      d = json.loads(f.read_text(encoding="utf-8"))
      cands[letter] = [Subgoal(**s) for s in d["subgoals"]]
      names[letter] = f"{d['view']} ({VIEW_NAMES[d['view']]})"
      if d.get("gold_key"):
        seen.append(d["gold_key"])
    content = core.consolidate_content(core.task_text(meta, ctx.osw), cands, names)
    text, usage = call(ctx.think, CONSOLIDATE_SYSTEM, content)
    sgs, notes = core.parse_subgoals(text)
    if not sgs:
      raise RuntimeError(f"consolidation parse failed for {t}: {text[:300]}")
    sset = SubgoalSet(task_id=t, domain=meta.domain, instruction=meta.instruction, version=ctx.args.version,
                      views=[json.loads(f.read_text())["view"] for f in files],
                      subgoals=core.finalize_consolidated(sgs, len(files)), notes=notes,
                      extra={"prompt_version": SUBGOAL_PROMPT_VERSION, "usage": usage, "letters": names,
                             "gold_seen_in_extraction": seen})
    write_json(p, sset.model_dump())
    return "ok"

  return run_units("consolidate", ctx.tasks, one, ctx.args.workers)


def load_set(ctx: Ctx, t: str, verified: bool) -> SubgoalSet | None:
  p = ctx.out / ("sets_verified" if verified else "sets") / f"{t}.json"
  return SubgoalSet(**json.loads(p.read_text(encoding="utf-8"))) if p.exists() else None


def track_one(ctx: Ctx, r, sset: SubgoalSet, kind: str) -> Track:
  p = ctx.ep_path(f"tracks/{kind}", r)
  if p.exists():
    return Track(**json.loads(p.read_text(encoding="utf-8")))
  traj, meta = load_episode(r), ctx.meta(r.task_id)
  log = captioned_log(ctx, r)
  sgs = sset.subgoals
  valid = {s.id for s in sgs}
  statuses, usage, prev = [], [], None
  for win in core.track_windows(core.screenshot_steps(traj), ctx.args.check_window):
    content = core.check_content(meta, sgs, traj, log, win, prev, max_width=ctx.args.max_width)
    text, u = call(ctx.fast, CHECK_SYSTEM, content)
    got = core.parse_states(text, win, valid)
    usage.append(u)
    if not got:
      raise RuntimeError(f"no states parsed for {traj.key} steps {win[0]}-{win[-1]}: {text[:300]}")
    statuses += got
    prev = got[-1]
  tr = Track(key=traj.key, task_id=r.task_id, agent=r.agent, result=traj.result, n_steps=traj.n_steps,
             version=ctx.args.version, statuses=statuses, usage={"calls": usage})
  write_json(p, tr.model_dump())
  return tr


def stage_verify(ctx: Ctx) -> int:
  def one(t) -> str | None:
    p = ctx.out / "sets_verified" / f"{t}.json"
    if p.exists():
      return None
    sset = load_set(ctx, t, verified=False)
    if sset is None:
      raise RuntimeError(f"no consolidated set for {t}")
    tracks = [track_one(ctx, r, sset, "gold") for r in ctx.golds[ctx.golds.task_id == t].itertuples()]
    write_json(p, core.verify(sset, tracks, seen_keys=set(sset.extra.get("gold_seen_in_extraction") or [])).model_dump())
    return "ok"

  return run_units("verify", ctx.tasks, one, ctx.args.workers)


def stage_track(ctx: Ctx) -> int:
  def one(r) -> str | None:
    if ctx.ep_path("tracks/fail", r).exists():
      return None
    sset = load_set(ctx, r.task_id, verified=True) or load_set(ctx, r.task_id, verified=False)
    track_one(ctx, r, sset, "fail")
    return "ok"

  return run_units("track", list(ctx.fails.itertuples()), one, ctx.args.workers)


def stage_localize(ctx: Ctx) -> int:
  out_dir = ctx.out / "localize"
  out_dir.mkdir(parents=True, exist_ok=True)
  done: set[str] = set()
  for f in out_dir.glob("*.jsonl"):
    done |= {json.loads(line)["key"] for line in f.read_text(encoding="utf-8").splitlines() if line.strip()}

  def one(r) -> str | None:
    tp = ctx.ep_path("tracks/fail", r)
    if not tp.exists():
      raise RuntimeError("no track")
    track = Track(**json.loads(tp.read_text(encoding="utf-8")))
    if track.key in done:
      return None
    sset = load_set(ctx, r.task_id, verified=True) or load_set(ctx, r.task_id, verified=False)
    traj, meta, log = load_episode(r), ctx.meta(r.task_id), captioned_log(ctx, r)
    row: dict[str, Any] = {"key": track.key, "agent": r.agent, "task_id": r.task_id, "domain": r.domain,
                           "n_steps": track.n_steps, "version": ctx.args.version,
                           "human_root": None if pd.isna(r.root_error_step) else int(r.root_error_step),
                           "human_tag": r.taxonomy_tag}
    for variant, use_ver in (("verified", True), ("unverified", False)):
      row[f"rule_{variant}"] = core.locate_failure(sset, track, use_verification=use_ver)
    loc = row["rule_verified"]
    content = core.localize_content(meta, sset, track, traj, log, loc, image_budget=ctx.args.localize_images,
                                    max_width=ctx.args.max_width)
    lo, hi = loc["lo"], loc["hi"]
    text, usage = call(ctx.think, LOCALIZE_SYSTEM, content)
    row["llm"] = core.parse_localization(text, lo, hi)
    row["llm"]["window"] = [lo, hi]
    row["usage"] = usage
    if not row["llm"]["parse_ok"]:
      row["raw"] = text[:3000]
    with _lock, (out_dir / f"{r.agent}.jsonl").open("a", encoding="utf-8") as fh:
      fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return "ok"

  return run_units("localize", list(ctx.fails.itertuples()), one, ctx.args.workers)


def stage_score(ctx: Ctx) -> int:
  import subprocess
  cmd = [sys.executable, str(ROOT / "scripts/score_subgoals.py"), "--version", ctx.args.version, "--out", str(ctx.args.out)]
  if ctx.args.pilot:
    cmd += ["--pilot", ctx.args.pilot]
  return subprocess.call(cmd)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("stage", choices=STAGES + ["all"])
  ap.add_argument("--version", required=True)
  ap.add_argument("--index", default=str(ROOT / "data/index"))
  ap.add_argument("--out", default=str(ROOT / "data/subgoals"))
  ap.add_argument("--pilot", help="pilot JSON with rows [{agent, task_id}] (e.g. data/index/smoke5.json)")
  ap.add_argument("--limit", type=int, default=None, help="first N failures only")
  ap.add_argument("--env-file", default=str(ROOT / "config/openai.env"))
  ap.add_argument("--provider", choices=["openai", "anthropic"], default=None)
  ap.add_argument("--model", default=None)
  ap.add_argument("--reasoning-effort", default=None, help="caption/track calls (OpenAI reasoning models; omit for vLLM)")
  ap.add_argument("--think-effort", default=None, help="extract/consolidate/localize calls; default = --reasoning-effort")
  ap.add_argument("--max-tokens", type=int, default=3000)
  ap.add_argument("--think-max-tokens", type=int, default=16000, help="with --think-effort (reasoning tokens count)")
  ap.add_argument("--workers", type=int, default=4)
  ap.add_argument("--max-width", type=int, default=1280)
  ap.add_argument("--caption-window", type=int, default=6, help="steps per caption call (+1 after-screenshot)")
  ap.add_argument("--check-window", type=int, default=8, help="screenshots per tracking call")
  ap.add_argument("--localize-images", type=int, default=12)
  ap.add_argument("--dry-run", action="store_true", help="print the population and exit")
  args = ap.parse_args()

  ctx = Ctx(args)
  print(f"version={args.version} failures={len(ctx.fails)} tasks={len(ctx.tasks)} gold successes={len(ctx.golds)}", flush=True)
  if args.dry_run:
    return
  ctx.out.mkdir(parents=True, exist_ok=True)
  write_json(ctx.out / f"run_{time.strftime('%Y%m%d_%H%M%S')}.json",
             {k: v for k, v in vars(args).items()} | {"prompt_version": SUBGOAL_PROMPT_VERSION,
                                                      "fast_model": ctx.fast.cfg.model, "think_model": ctx.think.cfg.model})
  stages = STAGES if args.stage == "all" else [args.stage]
  fns = {"caption": stage_caption, "extract": stage_extract, "consolidate": stage_consolidate, "verify": stage_verify,
         "track": stage_track, "localize": stage_localize, "score": stage_score}
  total = 0
  for s in stages:  # failed units are skipped downstream; rerun the same command to retry them
    failed = fns[s](ctx)
    total += failed
    if failed:
      print(f"stage {s}: {failed} failures (rerun to resume)", flush=True)
  sys.exit(1 if total else 0)


if __name__ == "__main__":
  main()
