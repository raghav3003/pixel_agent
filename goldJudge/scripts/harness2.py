#!/usr/bin/env python3
"""Cross-domain harnesses (docs/research/2026-11-06-failure-analysis-harness-survey.md), on all labelled failures.

  python scripts/harness2.py run --method prefix|evtrace|adjudicate --version V --env-file config/openai.env
  python scripts/harness2.py score --version V

``adjudicate`` needs the top-k judge (run_judge.py --topk 5, --candidates-from). ``score`` decodes with 5-fold
cross-validation grouped by task: the position-prior fusion (idea 1) and the prefix-scan threshold (idea 3) are fit on
the training folds only. Outputs data/harness2/<V>/<method>/<agent>/<task>.json. Labels are used only to score.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from score_subgoals import load_judge  # noqa: E402
from subgoal_cost import usages  # noqa: E402
from subgoals import ROOT, Ctx, call, load_episode, load_log, run_units, write_json  # noqa: E402
from goldjudge.subgoals import core, harness2 as H  # noqa: E402
from goldjudge.subgoals.core import parse_json  # noqa: E402
from goldjudge.subgoals.localizers import parse_root  # noqa: E402

TOPK_DEFAULT = "gpt6luna-fixed-after-top5-v0.3/G1"


def load_topk(spec: str) -> dict[tuple[str, str], dict]:
  """(agent, task) -> {"root": int, "cands": [(step, conf)], "symptom": int|None} from a --topk judge run."""
  v, cond = spec.split("/")
  out = {}
  for f in (ROOT / "data/judge" / v / cond).glob("*.jsonl"):
    for line in f.read_text(encoding="utf-8").splitlines():
      if not line.strip():
        continue
      r = json.loads(line)
      root = (r.get("verdict") or {}).get("root_step")
      obj = parse_json(r.get("raw") or "", "candidate_steps") or {}
      cands = []
      for c in obj.get("candidate_steps") or []:
        try:
          cands.append((int(c["step"]), float(c.get("confidence") or 0)))
        except (TypeError, ValueError, KeyError):
          continue
      if root is not None and all(s != root for s, _ in cands):
        cands.insert(0, (int(root), max([c for _, c in cands] + [0.5])))
      try:
        symptom = int(obj["symptom_step"]) if obj.get("symptom_step") is not None else None
      except (TypeError, ValueError):
        symptom = None
      out[(r["agent"], r["task_id"])] = {"root": root, "cands": cands, "symptom": symptom}
  return out


def run(ctx: Ctx) -> int:
  a = ctx.args
  m = a.method
  base = ctx.out / m

  def task_spec(t: str) -> str:
    return core.task_text(ctx.meta(t), ctx.osw)

  if m == "prefix":
    def one(r) -> str | None:
      p = base / r.agent / f"{r.task_id}.json"
      if p.exists():
        return None
      traj, log = load_episode(r), load_log(r)
      wins, usage = [], []
      for w in H.prefix_windows(traj.n_steps, a.window):
        text, u = call(ctx.fast, H.PREFIX_SYSTEM, H.prefix_content(task_spec(r.task_id), traj, log, w, max_width=a.max_width))
        wins.append({"window": [w[0], w[-1]], **H.parse_prefix(text, w)})
        usage.append(u)
      write_json(p, {"agent": r.agent, "task_id": r.task_id, "n_steps": traj.n_steps, "prompt_version": H.HARNESS2_PROMPT_VERSION,
                     "windows": wins, "usage": usage})
      return "ok"
    return run_units("prefix", list(ctx.fails.itertuples()), one, a.workers)

  if m == "evtrace":
    def checks_one(t) -> str | None:
      p = base / "checks" / f"{t}.json"
      if p.exists():
        return None
      text, u = call(ctx.fast, H.CHECKS_SYSTEM, H.checks_content(task_spec(t)))
      checks = (parse_json(text, "checks") or {}).get("checks") or []
      write_json(p, {"task_id": t, "checks": checks, "usage": u})
      return "ok"

    def one(r) -> str | None:
      p = base / r.agent / f"{r.task_id}.json"
      if p.exists():
        return None
      checks = json.loads((base / "checks" / f"{r.task_id}.json").read_text(encoding="utf-8"))["checks"]
      traj, log = load_episode(r), load_log(r)
      t1, u1 = call(ctx.fast, H.TOUCH_SYSTEM, H.touch_content(task_spec(r.task_id), checks, traj, log))
      touches, done = H.parse_touches(t1, traj.n_steps)
      t2, u2 = call(ctx.fast, H.TRACE_SYSTEM, H.trace_content(task_spec(r.task_id), checks, traj, log, touches, done,
                                                               image_budget=a.images, max_width=a.max_width))
      out = parse_root(t2, traj.n_steps)
      if not out["parse_ok"]:
        raise RuntimeError(f"no root_step parsed: {t2[:300]}")
      write_json(p, {"agent": r.agent, "task_id": r.task_id, "n_steps": traj.n_steps, "touches": touches, "done_step": done,
                     "out": out, "usage": [u1, u2]})
      return "ok"
    failed = run_units("evtrace-checks", ctx.tasks, checks_one, a.workers)
    return failed + run_units("evtrace", list(ctx.fails.itertuples()), one, a.workers)

  if m == "adjudicate":
    topk = load_topk(a.candidates_from)

    def one(r) -> str | None:
      p = base / r.agent / f"{r.task_id}.json"
      if p.exists():
        return None
      j = topk.get((r.agent, r.task_id))
      if not j:
        raise RuntimeError("no top-k judge output")
      traj, log = load_episode(r), load_log(r)
      cands = sorted({s for s, _ in j["cands"] if 1 <= s <= traj.n_steps})[: a.max_candidates]
      labels, usage = {}, []
      for c in cands:
        text, u = call(ctx.fast, H.ADJUDICATE_SYSTEM, H.adjudicate_content(task_spec(r.task_id), traj, log, c, max_width=a.max_width))
        labels[c] = H.parse_adjudication(text)
        usage.append(u)
      step, how = H.decide_adjudication(cands, labels, j["root"])
      write_json(p, {"agent": r.agent, "task_id": r.task_id, "n_steps": traj.n_steps, "candidates": cands,
                     "labels": {str(k): v for k, v in labels.items()}, "out": {"root_step": step, "how": how}, "usage": usage})
      return "ok"
    return run_units("adjudicate", list(ctx.fails.itertuples()), one, a.workers)
  raise ValueError(m)


# --- scoring with cross-validation -------------------------------------------------------------


def folds_by_task(tasks: list[str], k: int = 5, seed: int = 0) -> dict[str, int]:
  uniq = sorted(set(tasks))
  rng = np.random.default_rng(seed)
  rng.shuffle(uniq)
  return {t: i % k for i, t in enumerate(uniq)}


def prior_table(train: pd.DataFrame, bw: float = 2.0, max_n: int = 101) -> dict[str, np.ndarray]:
  """Per-agent smoothed distribution of the absolute root step (index = step)."""
  out = {}
  for agent, g in list(train.groupby("agent")) + [("__all__", train)]:
    xs = np.arange(max_n)
    dens = sum(np.exp(-0.5 * ((xs - s) / bw) ** 2) for s in g.g.values) + 1e-3
    dens[0] = 0
    out[agent] = dens / dens.sum()
  return out


def fuse(cands: list[tuple[int, float]], n: int, prior: np.ndarray, lam: float, eps: float, sigma: float = 1.0) -> int:
  xs = np.arange(1, n + 1)
  pj = np.full(n, eps / n)
  tot = sum(c for _, c in cands) or 1.0
  for s, c in cands:
    pj += (c / tot) * np.exp(-0.5 * ((xs - s) / sigma) ** 2)
  pr = prior[1:n + 1] / prior[1:n + 1].sum()
  return int(xs[np.argmax(np.log(pj) + lam * np.log(pr))])


def score(args: argparse.Namespace) -> None:
  d = Path(args.out) / args.version
  lab = pd.read_parquet(Path(args.index) / "episodes.parquet")
  lab = lab[(lab.source == "cuadebug") & lab.root_error_step.notna()].reset_index(drop=True)
  dev = {(r["agent"], r["task_id"]) for r in json.loads((Path(args.index) / "h50.json").read_text())["rows"]}
  df = pd.DataFrame({"agent": lab.agent, "task_id": lab.task_id, "n": lab.n_steps.astype(int), "g": lab.root_error_step.astype(float)})
  df["split"] = ["dev" if (a, t) in dev else "test" for a, t in zip(df.agent, df.task_id)]
  df["fold"] = df.task_id.map(folds_by_task(list(df.task_id), args.folds))
  key = list(zip(df.agent, df.task_id))
  preds: dict[str, list] = {}

  for name, spec in [("orig judge G0", "gpt6luna-fixed-v0.3/G0"), ("orig judge G1", "gpt6luna-fixed-v0.3/G1"),
                     ("after-label judge G1", "gpt6luna-fixed-after-v0.3/G1")]:
    j = load_judge(spec)
    preds[name] = [j.get(k) for k in key]
  topk = load_topk(args.candidates_from)
  preds["top5 judge (its root)"] = [(topk.get(k) or {}).get("root") for k in key]
  preds["const 3"] = [min(3, n) for n in df.n]

  def load_dir(m: str) -> dict:
    out = {}
    for f in (d / m).rglob("*.json"):
      r = json.loads(f.read_text(encoding="utf-8"))
      if "agent" in r:
        out[(r["agent"], r["task_id"])] = r
    return out

  adj, pre, evt = load_dir("adjudicate"), load_dir("prefix"), load_dir("evtrace")
  if adj:
    preds["2 adjudicate (earliest ROOT)"] = [(adj.get(k) or {}).get("out", {}).get("root_step") for k in key]
  if evt:
    preds["4 evaluator trace"] = [(evt.get(k) or {}).get("out", {}).get("root_step") for k in key]

  # cross-validated decoders
  fus, fus_prior_only, pre_pred = [None] * len(df), [None] * len(df), [None] * len(df)
  chosen = []
  for f in range(args.folds):
    tr, te = df[df.fold != f], df[df.fold == f]
    pri = prior_table(tr)
    # idea 1: grid on training folds
    best, best_acc = (1.0, 0.05), -1
    for lam in (0.0, 0.25, 0.5, 1.0, 2.0):
      for eps in (0.01, 0.05, 0.2, 0.5):
        hits = []
        for i in tr.index:
          j = topk.get(key[i])
          if j and j["cands"]:
            p = fuse(j["cands"], df.n[i], pri.get(df.agent[i], pri["__all__"]), lam, eps)
            hits.append(abs(p - df.g[i]) <= 2)
        acc = np.mean(hits) if hits else 0
        if acc > best_acc:
          best, best_acc = (lam, eps), acc
    chosen.append({"fold": f, "lambda": best[0], "eps": best[1], "train_pm2": round(best_acc, 3)})
    for i in te.index:
      j = topk.get(key[i])
      pr = pri.get(df.agent[i], pri["__all__"])
      if j and j["cands"]:
        fus[i] = fuse(j["cands"], df.n[i], pr, *best)
      fus_prior_only[i] = int(np.argmax(pr[1:df.n[i] + 1]) + 1)
    # idea 3: threshold on training folds, fallback = top5 judge root
    if pre:
      bt, bacc = 50, -1
      for theta in range(0, 101, 5):
        hits = [abs(H.decode_prefix(pre[key[i]]["windows"], theta, (topk.get(key[i]) or {}).get("root")) - df.g[i]) <= 2
                for i in tr.index if key[i] in pre and (topk.get(key[i]) or {}).get("root") is not None]
        acc = np.mean(hits) if hits else 0
        if acc > bacc:
          bt, bacc = theta, acc
      chosen[-1] |= {"theta": bt, "theta_train_pm2": round(bacc, 3)}
      for i in te.index:
        if key[i] in pre:
          pre_pred[i] = H.decode_prefix(pre[key[i]]["windows"], bt, (topk.get(key[i]) or {}).get("root"))
  preds["1 prior fusion (CV)"] = fus
  preds["prior only (CV)"] = fus_prior_only
  if pre:
    preds["3 prefix scan (CV threshold)"] = pre_pred
    preds["3 prefix scan, first flag any conf"] = [H.decode_prefix(pre[k]["windows"], 1, None) if k in pre else None for k in key]

  ref = np.array([(p is not None and abs(p - g) <= 2) for p, g in zip(preds[args.reference], df.g)])
  rows = []
  for name, p in preds.items():
    p = pd.Series(p, dtype=float)
    err = (p - df.g).abs()
    hit = (err <= 2).values
    row = {"predictor": name, "have": int(p.notna().sum()), "exact": (err == 0).mean(), "pm1": (err <= 1).mean(),
           "pm2": hit.mean(), "pm5": (err <= 5).mean(),
           "pm2_dev": hit[df.split == "dev"].mean(), "pm2_test": hit[df.split == "test"].mean()}
    for a in sorted(df.agent.unique()):
      row[f"pm2_{a.split('-')[0]}"] = hit[df.agent == a].mean()
    if name != args.reference:
      a_, b_ = int((hit & ~ref).sum()), int((ref & ~hit).sum())
      row["vs_ref"] = f"+{a_}/-{b_} p={binomtest(a_, a_ + b_).pvalue:.2f}" if a_ + b_ else "same"
    else:
      row["vs_ref"] = "ref"
    rows.append(row)
  t = pd.DataFrame(rows)
  num = [c for c in t.columns if c not in {"predictor", "have", "vs_ref"}]
  t[num] = t[num].round(3)
  if topk:
    hit3 = np.mean([any(abs(s - g) <= 2 for s, _ in (topk.get(k) or {"cands": []})["cands"][:3]) for k, g in zip(key, df.g)])
    hit5 = np.mean([any(abs(s - g) <= 2 for s, _ in (topk.get(k) or {"cands": []})["cands"]) for k, g in zip(key, df.g)])
  cost = 0.0
  for m in ("prefix", "evtrace", "adjudicate"):
    for f in (d / m).rglob("*.json"):
      for u in usages(json.loads(f.read_text(encoding="utf-8"))):
        cost += ((u.get("prompt_tokens") or 0) * 0.1 + (u.get("completion_tokens") or 0) * 0.5) / 1e6
  out = d / "metrics"
  out.mkdir(parents=True, exist_ok=True)
  t.to_csv(out / "comparison.csv", index=False)
  pd.DataFrame(chosen).to_csv(out / "cv_choices.csv", index=False)
  pd.set_option("display.width", 250)
  print(f"{len(df)} labelled failures, {args.folds}-fold CV grouped by task; paired sign test on +-2 vs '{args.reference}' (uncorrected)")
  print(t.to_string(index=False))
  if topk:
    print(f"top-k judge: root within +-2 of one of its top-3 candidates {hit3:.3f}, of its top-5 {hit5:.3f}")
  print(pd.DataFrame(chosen).to_string(index=False))
  print(f"harness2 call cost ${cost:.2f} -> {out}")


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("cmd", choices=["run", "score"])
  ap.add_argument("--method", choices=["prefix", "evtrace", "adjudicate"])
  ap.add_argument("--version", required=True)
  ap.add_argument("--candidates-from", default=TOPK_DEFAULT, help="top-k judge run <judge_version>/<cond>")
  ap.add_argument("--reference", default="orig judge G1")
  ap.add_argument("--index", default=str(ROOT / "data/index"))
  ap.add_argument("--out", default=str(ROOT / "data/harness2"))
  ap.add_argument("--pilot")
  ap.add_argument("--limit", type=int, default=None)
  ap.add_argument("--folds", type=int, default=5)
  ap.add_argument("--window", type=int, default=8, help="prefix: steps per window")
  ap.add_argument("--images", type=int, default=12, help="evtrace: screenshots of touching steps")
  ap.add_argument("--max-candidates", type=int, default=5)
  ap.add_argument("--env-file", default=str(ROOT / "config/openai.env"))
  ap.add_argument("--provider", default=None)
  ap.add_argument("--model", default=None)
  ap.add_argument("--reasoning-effort", default="none")
  ap.add_argument("--think-effort", default=None)
  ap.add_argument("--max-tokens", type=int, default=3000)
  ap.add_argument("--think-max-tokens", type=int, default=16000)
  ap.add_argument("--workers", type=int, default=6)
  ap.add_argument("--max-width", type=int, default=1280)
  ap.add_argument("--dry-run", action="store_true")
  args = ap.parse_args()
  if args.cmd == "score":
    score(args)
    return
  if not args.method:
    ap.error("--method is required for run")
  ctx = Ctx(args)
  print(f"method={args.method} version={args.version} failures={len(ctx.fails)}", flush=True)
  if args.dry_run:
    return
  ctx.out.mkdir(parents=True, exist_ok=True)
  sys.exit(1 if run(ctx) else 0)


if __name__ == "__main__":
  main()
