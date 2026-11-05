#!/usr/bin/env python3
"""Subgoals from text alone for the agent and the gold; divergence against the human root step.

  python scripts/text_subgoals.py run    --version V --env-file config/openai.env --think-effort medium --workers 8 \
      --pilot data/index/h50.json
  python scripts/text_subgoals.py screen --version V ... --pilot data/index/h50.json   # claims + searches vs screen
  python scripts/text_subgoals.py score  --version V --pilot data/index/h50.json
  python scripts/text_subgoals.py review --version V --pilot data/index/h50.json   # lists side by side, for reading

Outputs under data/textgoals/<V>/ (resumable, never overwritten):
  gold_human/<task>.r<N>.json        subgoals from the OSWorld-Human steps
  agent/<agent>/<task>.r<N>.json     subgoals from the failed run's reasoning and actions
  align/<agent>/<task>.r<N>.json     agent vs gold, plus the rule's label and step
  gold_ai/<task>.json                second gold, where a successful run has real reasoning
  align_gold/<task>.json             human-derived gold vs that second gold

Repeat N runs the whole chain independently (test-retest). Human labels are used only to score.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from score_subgoals import load_judge, load_localize  # noqa: E402
from subgoal_cost import usages  # noqa: E402
from subgoals import Ctx, ROOT, call, load_episode, load_log, run_units, write_json  # noqa: E402
from goldjudge.actions import human_action_log  # noqa: E402
from goldjudge.subgoals import textgoals as tg  # noqa: E402
from goldjudge.subgoals.textgoals import TextGoal  # noqa: E402

# Agents whose successful runs log real reasoning (median chars per step: opencua ~1600, sonnet ~180; opus ~21).
REASONING_GOLD_AGENTS = ("opencua-7b", "claude-sonnet-4-5")
BASELINES = ["gpt6luna-fixed-v0.3/G0", "gpt6luna-fixed-v0.3/G1", "gpt6luna-think-fixed-v0.3/G0"]
SUBGOAL_V02 = ROOT / "data/subgoals/gpt6luna-sg-v0.2"


def _goals(p: Path) -> list[TextGoal]:
  return [TextGoal(**g) for g in json.loads(p.read_text(encoding="utf-8"))["subgoals"]]


def _extract(ctx: Ctx, p: Path, content, n_steps: int, *, is_agent: bool, meta: dict) -> str | None:
  if p.exists():
    return None
  text, usage = call(ctx.think, tg.EXTRACT_SYSTEM, content)
  goals = tg.parse_goals(text, n_steps, is_agent=is_agent)
  if not goals:
    raise RuntimeError(f"no subgoals parsed: {text[:300]}")
  write_json(p, {**meta, "prompt_version": tg.TEXTGOAL_PROMPT_VERSION, "n_steps": n_steps,
                 "subgoals": [g.model_dump() for g in goals], "usage": usage})
  return "ok"


def run(ctx: Ctx) -> int:
  repeats = range(1, ctx.args.repeats + 1)
  out = ctx.out

  def gold_one(u) -> str | None:
    t, n = u
    meta = ctx.meta(t)
    log = human_action_log(meta.human)
    return _extract(ctx, out / "gold_human" / f"{t}.r{n}.json", tg.gold_content(meta.instruction, log), len(log.steps),
                    is_agent=False, meta={"task_id": t, "repeat": n, "source": "osworld-human"})

  def agent_one(u) -> str | None:
    r, n = u
    traj, log = load_episode(r), load_log(r)
    return _extract(ctx, out / "agent" / r.agent / f"{r.task_id}.r{n}.json",
                    tg.agent_content(ctx.meta(r.task_id).instruction, traj, log), traj.n_steps, is_agent=True,
                    meta={"task_id": r.task_id, "agent": r.agent, "repeat": n})

  def align_one(u) -> str | None:
    r, n = u
    p = out / "align" / r.agent / f"{r.task_id}.r{n}.json"
    if p.exists():
      return None
    gold = _goals(out / "gold_human" / f"{r.task_id}.r{n}.json")
    agent = _goals(out / "agent" / r.agent / f"{r.task_id}.r{n}.json")
    text, usage = call(ctx.think, tg.ALIGN_SYSTEM, tg.align_content(ctx.meta(r.task_id).instruction, gold, agent))
    parsed = tg.parse_alignment(text, gold, agent)
    if parsed is None:
      raise RuntimeError(f"alignment parse failed: {text[:300]}")
    gm, am = parsed
    write_json(p, {"task_id": r.task_id, "agent": r.agent, "repeat": n, "prompt_version": tg.TEXTGOAL_PROMPT_VERSION,
                   "human_root": None if pd.isna(r.root_error_step) else int(r.root_error_step),
                   "human_tag": r.taxonomy_tag, "human_family": r.taxonomy_family, "n_steps": int(r.n_steps),
                   "gold": [m.model_dump() for m in gm], "agent_rel": [m.model_dump() for m in am],
                   "summary": tg.classify(agent, gm, am, int(r.n_steps)), "usage": usage})
    return "ok"

  # a second gold wherever a successful run has real reasoning (first such run per task)
  rich = ctx.golds[ctx.golds.agent.isin(REASONING_GOLD_AGENTS)].drop_duplicates("task_id")

  def gold_ai_one(r) -> str | None:
    traj, log = load_episode(r), load_log(r)
    return _extract(ctx, out / "gold_ai" / f"{r.task_id}.json", tg.agent_content(ctx.meta(r.task_id).instruction, traj, log),
                    traj.n_steps, is_agent=False, meta={"task_id": r.task_id, "agent": r.agent, "key": traj.key})

  def align_gold_one(r) -> str | None:
    p = out / "align_gold" / f"{r.task_id}.json"
    if p.exists():
      return None
    human, ai = _goals(out / "gold_human" / f"{r.task_id}.r1.json"), _goals(out / "gold_ai" / f"{r.task_id}.json")
    text, usage = call(ctx.think, tg.ALIGN_SYSTEM, tg.align_content(ctx.meta(r.task_id).instruction, human, ai))
    parsed = tg.parse_alignment(text, human, ai)
    if parsed is None:
      raise RuntimeError(f"alignment parse failed: {text[:300]}")
    write_json(p, {"task_id": r.task_id, "agent": r.agent, "gold": [m.model_dump() for m in parsed[0]],
                   "agent_rel": [m.model_dump() for m in parsed[1]], "usage": usage})
    return "ok"

  rows = list(ctx.fails.itertuples())
  if ctx.args.cmd == "screen":
    return run_units("screen", rows, lambda r: screen_one(ctx, r), ctx.args.workers)
  failed = 0
  stages = set(ctx.args.stages)
  if "gold" in stages:
    failed += run_units("gold", [(t, n) for t in ctx.tasks for n in repeats], gold_one, ctx.args.workers)
  if stages == {"gold"}:
    return failed
  failed += run_units("agent", [(r, n) for r in rows for n in repeats], agent_one, ctx.args.workers)
  failed += run_units("align", [(r, n) for r in rows for n in repeats], align_one, ctx.args.workers)
  failed += run_units("gold_ai", list(rich.itertuples()), gold_ai_one, ctx.args.workers)
  failed += run_units("align_gold", list(rich.itertuples()), align_gold_one, ctx.args.workers)
  return failed


def screen_one(ctx: Ctx, r) -> str | None:
  """Repeat 1 only: check the agent's claims and its long searches against the screen."""
  out = ctx.out / ctx.args.screen_dir / r.agent / f"{r.task_id}.json"
  if out.exists():
    return None
  ap = ctx.out / "align" / r.agent / f"{r.task_id}.r1.json"
  if not ap.exists():
    raise RuntimeError("no alignment")
  a = json.loads(ap.read_text(encoding="utf-8"))
  agent = _goals(ctx.out / "agent" / r.agent / f"{r.task_id}.r1.json")
  gold = {g.id: g for g in _goals(ctx.out / "gold_human" / f"{r.task_id}.r1.json")}
  gm, am = [tg.GoldMatch(**m) for m in a["gold"]], [tg.AgentMatch(**m) for m in a["agent_rel"]]
  traj, log, meta = load_episode(r), load_log(r), ctx.meta(r.task_id)
  human = human_action_log(meta.human)
  usage, claims, searches = [], [], []
  goals = tg.claims_to_check(agent, am, ctx.args.claim_images)
  if goals:
    content, shots = tg.claims_content(meta.instruction, traj, log, goals, max_width=ctx.args.max_width)
    text, u = call(ctx.think, tg.CLAIMS_SYSTEM, content)
    usage.append(u)
    claims = tg.parse_claims(text, goals)
    for c in claims:
      c.screenshot_step = shots.get(c.id)
  serves = {m.id: m.gold for m in am}
  for g in tg.searches_to_check(agent, am, min_steps=ctx.args.min_search, max_n=ctx.args.max_searches):
    target = next((gold[i] for i in serves.get(g.id, []) if i in gold), None)
    hsteps = [s.render() for s in human.steps if target and target.first_step <= s.step <= target.last_step]
    content, shown = tg.search_content(meta.instruction, traj, log, g, target.purpose if target else "?", hsteps,
                                       image_budget=ctx.args.search_images, max_width=ctx.args.max_width)
    text, u = call(ctx.think, tg.SEARCH_SYSTEM, content)
    usage.append(u)
    searches.append(tg.parse_search(text, g, shown))
  summary = tg.classify_with_screen(a["summary"], agent, claims, searches)
  write_json(out, {"task_id": r.task_id, "agent": r.agent, "prompt_version": tg.SCREEN_PROMPT_VERSION,
                   "human_root": a["human_root"], "human_tag": a["human_tag"], "human_family": a["human_family"],
                   "n_steps": a["n_steps"], "claims": [c.model_dump() for c in claims],
                   "searches": [s.model_dump() for s in searches], "summary": summary, "usage": usage})
  return "ok"


def load_screened(d: Path, pilot: str | None, screen_dir: str = "screen") -> pd.DataFrame:
  rows = []
  for f in (d / screen_dir).rglob("*.json"):
    r = json.loads(f.read_text(encoding="utf-8"))
    rows.append({**{k: r.get(k) for k in ("task_id", "agent", "human_root", "human_tag", "human_family", "n_steps")}, **r["summary"]})
  df = pd.DataFrame(rows)
  if pilot and not df.empty:
    keys = {(r["agent"], r["task_id"]) for r in json.loads(Path(pilot).read_text())["rows"]}
    df = df[[(a, t) in keys for a, t in zip(df.agent, df.task_id)]]
  return df


def load_aligned(d: Path, pilot: str | None) -> pd.DataFrame:
  rows = []
  for f in (d / "align").rglob("*.json"):
    r = json.loads(f.read_text(encoding="utf-8"))
    rows.append({**{k: r.get(k) for k in ("task_id", "agent", "repeat", "human_root", "human_tag", "human_family", "n_steps")}, **r["summary"]})
  df = pd.DataFrame(rows)
  if pilot and not df.empty:
    keys = {(r["agent"], r["task_id"]) for r in json.loads(Path(pilot).read_text())["rows"]}
    df = df[[(a, t) in keys for a, t in zip(df.agent, df.task_id)]]
  return df


def _scores(pred: pd.Series, gold: pd.Series) -> dict:
  ok = pred.notna()
  d = (pred[ok].astype(float) - gold[ok].astype(float)).abs()
  n = len(pred)
  return {"n": n, "predicted": int(ok.sum()), "exact_all": round((d == 0).sum() / n, 3), "pm2_all": round((d <= 2).sum() / n, 3),
          "exact_when_predicted": round((d == 0).mean(), 3) if len(d) else None, "pm2_when_predicted": round((d <= 2).mean(), 3) if len(d) else None}


def score(args: argparse.Namespace) -> None:
  d = Path(args.out) / args.version
  df = load_aligned(d, args.pilot)
  if df.empty:
    print("no aligned rows")
    return
  r1 = df[(df["repeat"] == 1) & df.human_root.notna()].set_index(["agent", "task_id"]).sort_index()
  gold = r1.human_root.astype(float)

  # 1. localization, every predictor on the same trajectories
  table = {"text divergence (this)": r1.step, "const3": pd.Series(3.0, index=r1.index).clip(upper=r1.n_steps.astype(float))}
  scr = load_screened(d, args.pilot, args.screen_dir)
  if not scr.empty:
    scr = scr.set_index(["agent", "task_id"])
    table["text + screen check"] = pd.Series([scr.step.get(k) for k in r1.index], index=r1.index, dtype=float)
  for b in BASELINES:
    j = load_judge(b)
    table[f"judge {b}"] = pd.Series([j.get(k) for k in r1.index], index=r1.index, dtype=float)
  if (SUBGOAL_V02 / "localize").exists():
    loc = load_localize(SUBGOAL_V02).set_index(["agent", "task_id"])
    table["subgoals v0.2 + LLM"] = pd.Series([loc.llm.get(k, {}).get("root_step") if k in loc.index else None for k in r1.index], index=r1.index, dtype=float)
  loc_df = pd.DataFrame([{"predictor": k, **_scores(v, gold)} for k, v in table.items()])

  # paired against the plain judge, counting no prediction as a miss
  base = table[f"judge {BASELINES[0]}"].astype(float)
  mine = r1.step.astype(float)
  a, b = (mine - gold).abs() <= 2, (base - gold).abs() <= 2
  only_a, only_b = int((a & ~b).sum()), int((b & ~a).sum())
  p = binomtest(only_a, only_a + only_b).pvalue if only_a + only_b else None

  # 2. labels (after the screen check when it ran)
  lab = scr.reindex(r1.index) if not scr.empty else r1
  lab = lab.assign(human_root=r1.human_root, human_family=r1.human_family)
  by_label = pd.DataFrame([{"label": k, **_scores(g.step, g.human_root), **({"span_hit": round(((g.step <= g.human_root) & (g.human_root <= g.hi)).mean(), 3)})}
                           for k, g in lab.reset_index().groupby("label")])
  cross = pd.crosstab(lab.label, lab.human_family.fillna("?"), margins=True)
  if not scr.empty:
    moved = pd.crosstab(r1.label, lab.label, rownames=["text"], colnames=["text+screen"])
    print(f"== label change from the screen check ==\n{moved.to_string()}\n"
          f"claims checked {int(scr.n_claims.sum())}, false {int(scr.n_false_claims.sum())}, unclear {int(scr.n_unclear_claims.sum())}; "
          f"searches {int(scr.n_searches.sum())}, target visible {int(scr.n_visible_searches.sum())}, "
          f"visible but not used {int(scr.get('n_missed_searches', pd.Series(dtype=float)).sum())}\n")

  # 3. test-retest: the whole chain run independently
  retest = None
  wide = df.pivot_table(index=["agent", "task_id"], columns="repeat", values=["label", "step"], aggfunc="first")
  if 2 in df["repeat"].values:
    l1, l2 = wide["label"][1], wide["label"][2]
    s1, s2 = pd.to_numeric(wide["step"][1], errors="coerce"), pd.to_numeric(wide["step"][2], errors="coerce")
    both = s1.notna() & s2.notna()
    retest = {"n": len(wide), "same_label": round((l1 == l2).mean(), 3), "both_predict_a_step": int(both.sum()),
              "same_step": round(((s1 - s2)[both] == 0).mean(), 3), "within_2": round(((s1 - s2)[both].abs() <= 2).mean(), 3)}

  # 4. second gold: does a successful run with real reasoning contain the human-derived subgoals?
  stats = defaultdict(int)
  for f in (d / "align_gold").glob("*.json"):
    r = json.loads(f.read_text(encoding="utf-8"))
    stats["tasks"] += 1
    for m in r["gold"]:
      stats["human_subgoals"] += 1
      stats["matched" if m["status"] != "missing" else "unmatched"] += 1
    for m in r["agent_rel"]:
      stats["second_gold_subgoals"] += 1
      stats[f"second_gold_{m['relation']}"] += 1

  # 5. cost
  cost = defaultdict(lambda: [0, 0, 0])
  for sub in ("gold_human", "agent", "align", "gold_ai", "align_gold", args.screen_dir):
    for f in (d / sub).rglob("*.json"):
      for u in usages(json.loads(f.read_text(encoding="utf-8"))):
        c = cost[sub]
        c[0] += 1
        c[1] += u.get("prompt_tokens") or 0
        c[2] += u.get("completion_tokens") or 0
  dollars = sum((c[1] * args.price_in + c[2] * args.price_out) / 1e6 for c in cost.values())

  mdir = d / "metrics" / (Path(args.pilot).stem if args.pilot else "all")
  mdir.mkdir(parents=True, exist_ok=True)
  loc_df.to_csv(mdir / "localization.csv", index=False)
  by_label.to_csv(mdir / "by_label.csv", index=False)
  cross.to_csv(mdir / "label_vs_family.csv")
  print(f"== localization, first repeat, {len(r1)} labelled failures ==\n{loc_df.to_string(index=False)}")
  print(f"paired +-2 vs {BASELINES[0]}: this-only {only_a}, judge-only {only_b}, sign test p={p} (uncorrected)")
  print(f"\n== by label ==\n{by_label.to_string(index=False)}")
  print(f"\n== label vs human family ==\n{cross.to_string()}")
  print(f"\n== test-retest ==\n{retest}")
  print(f"\n== second gold ==\n{dict(stats)}")
  print(f"\n== sizes ==\n{r1[['n_gold', 'n_agent', 'n_find', 'n_finished', 'n_unfinished', 'n_missing', 'n_conflicts', 'n_extra']].mean().round(2).to_dict()}")
  print(f"\n== cost == {dict(cost)} total ${dollars:.2f}\nwritten to {mdir}")


def review(args: argparse.Namespace) -> None:
  d = Path(args.out) / args.version
  df = load_aligned(d, args.pilot)
  df = df[df["repeat"] == 1].sort_values(["agent", "task_id"])
  lines = [f"# Text subgoals, {args.version}", ""]
  for r in df.itertuples():
    ap = d / "align" / r.agent / f"{r.task_id}.r1.json"
    a = json.loads(ap.read_text(encoding="utf-8"))
    gold = {g.id: g for g in _goals(d / "gold_human" / f"{r.task_id}.r1.json")}
    agent = _goals(d / "agent" / r.agent / f"{r.task_id}.r1.json")
    rel = {m["id"]: m for m in a["agent_rel"]}
    lines += [f"## {r.agent} / {r.task_id}", f"Predicted: **{r.label}**, step {r.step}. Human: step {r.human_root}, tag {r.human_tag}. {r.n_steps} steps.", "", "Gold subgoals:"]
    for m in a["gold"]:
      lines.append(f"- {m['id']} [{m['status']}] {gold[m['id']].purpose} (agent: {', '.join(m['agent_goals']) or 'none'})")
    lines += ["", "Agent subgoals:"]
    for g in agent:
      lines.append(f"- {g.id} [{g.kind}, steps {g.first_step}-{g.last_step}, {g.ended}, {rel.get(g.id, {}).get('relation', '?')}] {g.purpose}")
    lines.append("")
  out = d / "review.md"
  out.write_text("\n".join(lines), encoding="utf-8")
  print(f"wrote {out} ({len(df)} trajectories)")


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("cmd", choices=["run", "screen", "score", "review"])
  ap.add_argument("--version", required=True)
  ap.add_argument("--index", default=str(ROOT / "data/index"))
  ap.add_argument("--out", default=str(ROOT / "data/textgoals"))
  ap.add_argument("--pilot")
  ap.add_argument("--limit", type=int, default=None)
  ap.add_argument("--repeats", type=int, default=2)
  ap.add_argument("--stages", nargs="*", default=["gold", "agent", "align", "gold_ai", "align_gold"],
                  help="run: 'gold' alone extracts only the gold subgoals (all that the checkpoint localizer needs)")
  ap.add_argument("--env-file", default=str(ROOT / "config/openai.env"))
  ap.add_argument("--provider", default=None)
  ap.add_argument("--model", default=None)
  ap.add_argument("--reasoning-effort", default=None)
  ap.add_argument("--think-effort", default=None)
  ap.add_argument("--max-tokens", type=int, default=4000)
  ap.add_argument("--think-max-tokens", type=int, default=16000)
  ap.add_argument("--workers", type=int, default=4)
  ap.add_argument("--price-in", type=float, default=0.10)
  ap.add_argument("--price-out", type=float, default=0.50)
  ap.add_argument("--claim-images", type=int, default=12, help="screen: max claims checked per trajectory")
  ap.add_argument("--min-search", type=int, default=4, help="screen: a find subgoal this many steps long is checked")
  ap.add_argument("--max-searches", type=int, default=3, help="screen: longest searches checked per trajectory")
  ap.add_argument("--search-images", type=int, default=6, help="screen: screenshots per search")
  ap.add_argument("--max-width", type=int, default=1280)
  ap.add_argument("--screen-dir", default="screen_v0.2", help="screen output subdir (screen = first version, kept)")
  ap.add_argument("--dry-run", action="store_true")
  args = ap.parse_args()
  if args.cmd == "score":
    score(args)
    return
  if args.cmd == "review":
    review(args)
    return
  ctx = Ctx(args)
  print(f"version={args.version} failures={len(ctx.fails)} tasks={len(ctx.tasks)} repeats={args.repeats}", flush=True)
  if args.dry_run:
    return
  ctx.out.mkdir(parents=True, exist_ok=True)
  write_json(ctx.out / f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json", vars(args) | {"prompt_version": tg.TEXTGOAL_PROMPT_VERSION, "model": ctx.think.cfg.model})
  sys.exit(1 if run(ctx) else 0)


if __name__ == "__main__":
  main()
