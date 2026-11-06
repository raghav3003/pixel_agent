#!/usr/bin/env python3
"""Round-3 methods (docs/research/2026-11-07-round3-research-and-audit.md): build per-trajectory extras for
run_judge.py --extra-file, and score everything with one 5-fold split grouped by task.

  python scripts/round3.py induce      # N1: convention guide per fold (LLM, training folds only) -> extras_n1.json
  python scripts/round3.py exemplars   # N2: retrieved training-fold exemplars (no LLM)            -> extras_n2.json
  python scripts/round3.py segments    # N4: subgoal table + screenshots of the pointed-at subgoals -> extras_n4.json
  python scripts/round3.py candidates  # N5: pooled candidates with before/after screens         -> extras_n5.json
  python scripts/round3.py score

Outputs under data/round3/. Human labels and evidence are only ever read for training-fold cases when building
the extras of a test-fold case, and for scoring.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_actions import load_episode, log_path  # noqa: E402
from harness2 import folds_by_task  # noqa: E402
from score_subgoals import load_judge  # noqa: E402
from goldjudge.actions import ActionLog  # noqa: E402
from goldjudge.judge.client import JudgeClient, JudgeClientConfig, load_env_file  # noqa: E402
from goldjudge.loaders.osworld_meta import OSWorldMeta  # noqa: E402
from goldjudge.parsing import run_date  # noqa: E402
from goldjudge.subgoals.core import trunc  # noqa: E402
from goldjudge.subgoals.localizers import candidate_steps  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/round3"
REPAIRED = "gpt6luna-repaired-v0.3-s{k}"
TEXTGOALS = ROOT / "data/textgoals/gpt6luna-text-v0.3/agent"


def labelled() -> pd.DataFrame:
  lab = pd.read_parquet(ROOT / "data/index/episodes.parquet")
  lab = lab[(lab.source == "cuadebug") & lab.root_error_step.notna()].reset_index(drop=True)
  lab["key"] = lab.agent + "|" + lab.task_id
  lab["fold"] = lab.task_id.map(folds_by_task(list(lab.task_id), 5))
  return lab


def evidence(r) -> dict:
  return json.loads((Path(r.dir) / "human_annotation.json").read_text(encoding="utf-8"))


def brief(traj, log: ActionLog, lo: int, hi: int, chars: int = 150) -> str:
  ls = {s.step: s for s in log.steps}
  return "\n".join(f"  step {s.step}: {trunc(s.thought or s.intent, chars) or '-'} || {trunc(ls[s.step].render(), 80) if s.step in ls else '?'}"
                   for s in traj.steps if lo <= s.step <= hi)


def load_log(r) -> ActionLog:
  return ActionLog.model_validate_json(log_path(ROOT / "data/actions", r).read_text(encoding="utf-8"))


# --- N1: convention induction ----------------------------------------------------------------

INDUCE_SYSTEM = """You study how expert annotators label the ROOT-CAUSE STEP of failed computer-use agent runs, so that \
an automatic judge can follow the same conventions. The annotators' rule: the earliest step whose executed action \
introduces a new causal error that leads to the failure (not a step that merely inherits an earlier mistake).

You get labelled cases: the task, the agent, run length, the step the annotators chose, their tag and evidence, and \
where an automatic judge put the root when it disagreed. Infer the CONVENTIONS behind the annotators' choices that the \
judge does not follow: which step of an error episode they label (decision vs execution, first attempt vs later \
retries, a premature DONE, a wrong strategy in a run that hits the step limit, an omission, an infeasible task, an \
environment or evaluator problem, an error the agent later repaired), and any systematic direction of the judge's \
misses. Write them as a short guide of concrete rules (at most 400 words), each rule with the situation and where to put \
the root step. Do not mention specific tasks."""


def induce(args) -> None:
  lab = labelled()
  load_env_file(args.env_file)
  client = JudgeClient(JudgeClientConfig(reasoning_effort=args.think_effort, max_tokens=12000))
  judge = load_judge(args.judge_for_induction)
  osw = OSWorldMeta()
  guides = {}
  OUT.mkdir(parents=True, exist_ok=True)
  for f in range(5):
    p = OUT / "guides" / f"fold{f}.txt"
    if p.exists():
      guides[f] = p.read_text(encoding="utf-8")
      continue
    lines = []
    for r in lab[lab.fold != f].itertuples():
      ev = evidence(r)
      instr = osw.load_task(r.domain, r.task_id, with_human=False).instruction
      j = judge.get((r.agent, r.task_id))
      miss = f" | judge said step {j}" if j is not None and abs(j - r.root_error_step) > 2 else ""
      lines.append(f"- [{r.agent}, {int(r.n_steps)} steps] task: {trunc(instr, 160)} | ROOT step {int(r.root_error_step)} "
                   f"[{r.taxonomy_tag}]{miss} | evidence: {trunc(ev.get('evidence', ''), 380)}")
    resp = client.chat(INDUCE_SYSTEM, [{"type": "text", "text": "# LABELLED CASES\n" + "\n".join(lines)},
                                       {"type": "text", "text": "Write the convention guide."}])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(resp.text.strip(), encoding="utf-8")
    guides[f] = resp.text.strip()
    print(f"fold {f}: {len(lines)} training cases, guide {len(resp.text)} chars, usage {resp.usage}", flush=True)
  extras = {r.key: {"system": "ANNOTATION CONVENTIONS (learned from expert-labelled runs; follow them when choosing "
                    "root_step):\n" + guides[r.fold]} for r in lab.itertuples()}
  (OUT / "extras_n1.json").write_text(json.dumps(extras), encoding="utf-8")
  print(f"wrote {OUT / 'extras_n1.json'}")


# --- N2: retrieved exemplars -----------------------------------------------------------------

_W = re.compile(r"[a-z0-9]+")


def _vec(text: str) -> collections.Counter:
  return collections.Counter(w for w in _W.findall(text.lower()) if len(w) > 2)


def _cos(a: collections.Counter, b: collections.Counter) -> float:
  num = sum(a[w] * b[w] for w in a if w in b)
  return num / (math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values())) or 1)


def exemplars(args) -> None:
  lab = labelled()
  osw = OSWorldMeta()
  instr = {r.key: osw.load_task(r.domain, r.task_id, with_human=False).instruction for r in lab.itertuples()}
  vec = {k: _vec(v) for k, v in instr.items()}
  views = {}
  for r in lab.itertuples():  # condensed view of each case around its root (used only as a training-fold exemplar)
    traj, log = load_episode(r), load_log(r)
    g = int(r.root_error_step)
    views[r.key] = (f"Task: {trunc(instr[r.key], 200)}\nAgent {r.agent}, {int(r.n_steps)} steps. Steps around the root:\n"
                    f"{brief(traj, log, g - 2, g + 1)}\nAnnotated ROOT step: {g} [{r.taxonomy_tag}]. "
                    f"Why: {trunc(evidence(r).get('evidence', ''), 350)}")
  extras = {}
  for r in lab.itertuples():
    pool = lab[lab.fold != r.fold]
    scored = sorted(((2.0 * (p.domain == r.domain) + _cos(vec[r.key], vec[p.key]), p.key) for p in pool.itertuples()),
                    reverse=True)[: args.k]
    ex = "\n\n".join(f"## Example {i}\n{views[k]}" for i, (_, k) in enumerate(scored, 1))
    extras[r.key] = {"context": "# LABELLED EXAMPLES (other runs, labelled by expert annotators; they show where the "
                                "annotators put the root step. Different tasks: do not copy step numbers.)\n" + ex}
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / "extras_n2.json").write_text(json.dumps(extras), encoding="utf-8")
  print(f"wrote {OUT / 'extras_n2.json'} ({len(extras)} cases, k={args.k})")


# --- N4 / N5: focus on what independent runs point at ----------------------------------------

def run_predictions(lab: pd.DataFrame, samples: int) -> dict[str, list[int]]:
  preds: dict[str, list[int]] = collections.defaultdict(list)
  for k in range(1, samples + 1):
    for cond in ("G1", "G0"):
      j = load_judge(f"{REPAIRED.format(k=k)}/{cond}")
      for (a, t), s in j.items():
        if s is not None and s > 0:
          preds[f"{a}|{t}"].append(int(s))
  return preds


def segments(args) -> None:
  lab = labelled()
  preds = run_predictions(lab, args.samples)
  extras = {}
  for r in lab.itertuples():
    p = TEXTGOALS / r.agent / f"{r.task_id}.r1.json"
    if not p.exists() or not preds.get(r.key):
      continue
    sgs = json.loads(p.read_text(encoding="utf-8"))["subgoals"]
    votes = collections.Counter()
    for s in preds[r.key]:
      for i, g in enumerate(sgs):
        if g["first_step"] <= s <= g["last_step"]:
          votes[i] += 1
    top = [i for i, _ in votes.most_common(2)]
    must = sorted({t for i in top for t in range(sgs[i]["first_step"], sgs[i]["last_step"] + 1)} | {max(1, sgs[top[0]]["first_step"] - 1)} if top else set())
    table = "\n".join(f"- subgoal {i + 1}: steps {g['first_step']}-{g['last_step']} [{g['kind']}]: {g['purpose']}" for i, g in enumerate(sgs))
    extras[r.key] = {"context": "# THE AGENT'S SUBGOALS (read from its reasoning; the screenshots below concentrate on the "
                                f"subgoals {[i + 1 for i in top]}, where earlier analyses placed the root, but the root may be elsewhere)\n" + table,
                     "must_steps": must[: args.max_must], "image_budget": args.budget}
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / "extras_n4.json").write_text(json.dumps(extras), encoding="utf-8")
  print(f"wrote {OUT / 'extras_n4.json'} ({len(extras)} cases)")


def candidates(args) -> None:
  lab = labelled()
  preds = run_predictions(lab, args.samples)
  extras = {}
  for r in lab.itertuples():
    cands = candidate_steps(preds.get(r.key, []), max_n=args.max_candidates)
    if not cands:
      continue
    must = sorted({t for c in cands for t in (c - 1, c) if 1 <= t <= r.n_steps})
    extras[r.key] = {"context": f"# CANDIDATE ROOT STEPS proposed by independent analyses of this run: {cands}\n"
                                "The screenshots below include the screen before and after each candidate. Decide which "
                                "candidate is the root-cause step under the definition (or another step if none is). "
                                "In your evidence, quote the agent's action at the chosen step and name the task "
                                "requirement it violates.",
                     "must_steps": must, "image_budget": max(args.budget, len(must) + 4)}
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / "extras_n5.json").write_text(json.dumps(extras), encoding="utf-8")
  print(f"wrote {OUT / 'extras_n5.json'} ({len(extras)} cases)")


# --- scoring ---------------------------------------------------------------------------------

def vote(ps: list[float]) -> float | None:
  ps = [p for p in ps if p == p and p is not None and p > 0]
  if not ps:
    return None
  sc = {c: sum(abs(p - c) <= 2 for p in ps) for c in range(int(min(ps)), int(max(ps)) + 1)}
  best = max(sc.values())
  centers = [c for c, v in sc.items() if v == best]
  # inside the densest +-2 window(s), answer the most frequent single prediction (earliest on ties)
  inside = [p for p in ps if any(abs(p - c) <= 2 for c in centers)]
  cnt = collections.Counter(inside)
  top = max(cnt.values())
  return float(min(p for p, v in cnt.items() if v == top))


def holm(pvals: list[float]) -> list[float]:
  order = np.argsort(pvals)
  adj, run = [0.0] * len(pvals), 0.0
  for rank, i in enumerate(order):
    run = max(run, min(1.0, (len(pvals) - rank) * pvals[i]))
    adj[i] = run
  return adj


def score(args) -> None:
  lab = labelled()
  keys = list(zip(lab.agent, lab.task_id))
  g = lab.root_error_step.astype(float).values
  seeded = np.array([bool(evidence(r).get("chosen_debugger")) for r in lab.itertuples()])
  conf = np.array([evidence(r).get("confidence", "") for r in lab.itertuples()])
  P: dict[str, np.ndarray] = {}

  def judge(spec: str) -> np.ndarray:
    j = load_judge(spec)
    return np.array([j[k] if j.get(k) not in (None, -1) else np.nan for k in keys], float)

  P["orig judge G1"] = judge("gpt6luna-fixed-v0.3/G1")
  P["orig judge G0"] = judge("gpt6luna-fixed-v0.3/G0")
  samples = {c: [judge(f"{REPAIRED.format(k=k)}/{c}") for k in range(1, args.samples + 1)] for c in ("G1", "G0")}
  P["R0 repaired G1 (s1)"] = samples["G1"][0]
  P["R0 repaired G0 (s1)"] = samples["G0"][0]
  for c in ("G1", "G0"):
    good = [s for s in samples[c] if np.isfinite(s).sum() > 150]
    if len(good) > 1:
      P[f"R0 repaired {c}: mean of {len(good)} runs"] = None  # filled below (accuracy averaged over runs)
      P[f"N3 window vote {c} x{len(good)}"] = np.array([vote(list(col)) or np.nan for col in np.array(good).T], float)
  allr = [s for c in ("G1", "G0") for s in samples[c] if np.isfinite(s).sum() > 150]
  if len(allr) > 2:
    P[f"N3 window vote G1+G0 x{len(allr)}"] = np.array([vote(list(col)) or np.nan for col in np.array(allr).T], float)
  n2 = [judge(f"gpt6luna-r3-n2{sfx}-v0.3/G1") for sfx in ("", "-s2", "-s3", "-s4") if (ROOT / f"data/judge/gpt6luna-r3-n2{sfx}-v0.3/G1").exists()]
  if len(n2) > 1:
    P[f"N2 exemplars: mean of {len(n2)} runs"] = ("runs", n2)
    P[f"N2+N3 window vote x{len(n2)}"] = np.array([vote(list(col)) or np.nan for col in np.array(n2).T], float)
  for name, v in [("N1 conventions", "gpt6luna-r3-n1-v0.3"), ("N2 exemplars", "gpt6luna-r3-n2-v0.3"),
                  ("N4 segment focus", "gpt6luna-r3-n4-v0.3"), ("N5 candidate selection", "gpt6luna-r3-n5-v0.3"),
                  ("N1+N2", "gpt6luna-r3-n12-v0.3"), ("checkpoints on repaired frame (G0)", "gpt6luna-r3-ckpt-v0.3")]:
    cond = "G0" if "(G0)" in name else "G1"
    if (ROOT / "data/judge" / v / cond).exists():
      P[name] = judge(f"{v}/{cond}")
  P["const 3"] = np.minimum(3, lab.n_steps.values).astype(float)

  ref = P["R0 repaired G1 (s1)"]
  rh = np.abs(ref - g) <= 2
  rows, pv = [], []
  long_ = lab.n_steps.values > 10
  for name, p in P.items():
    if p is None or isinstance(p, tuple):  # average accuracy over independent runs
      runs = p[1] if isinstance(p, tuple) else [x for x in samples["G1" if "G1" in name else "G0"] if np.isfinite(x).sum() > 150]
      accs = [np.mean(np.abs(s - g) <= 2) for s in runs]
      exs = [np.mean(np.abs(s - g) == 0) for s in runs]
      rows.append({"method": name, "pm2": np.mean(accs), "exact": np.mean(exs), "pm2_sd": np.std(accs, ddof=1)})
      continue
    e = np.abs(p - g)
    hit = e <= 2
    row = {"method": name, "have": int(np.isfinite(p).sum()), "exact": np.mean(e == 0), "pm1": np.mean(e <= 1), "pm2": hit.mean(),
           "pm5": np.mean(e <= 5), "mae": np.nanmean(e), "signed": np.nanmean(p - g),
           "pm2_len>10": hit[long_].mean(), "pm2_claude": hit[lab.agent.values == "claude-sonnet-4-5"].mean(),
           "pm2_seeded": hit[seeded].mean(), "pm2_unseeded": hit[~seeded].mean(), "pm2_highconf": hit[conf == "high"].mean()}
    if name != "R0 repaired G1 (s1)":
      a, b = int((hit & ~rh).sum()), int((rh & ~hit).sum())
      p_ = binomtest(a, a + b).pvalue if a + b else 1.0
      row["vs_R0"] = f"+{a}/-{b}"
      row["p"] = p_
      if name.startswith("N"):
        pv.append((len(rows), p_))
    rows.append(row)
  for (i, _), q in zip(pv, holm([p for _, p in pv])):
    rows[i]["p_holm_N"] = q
  t = pd.DataFrame(rows)
  num = [c for c in t.columns if c not in {"method", "have", "vs_R0"}]
  t[num] = t[num].astype(float).round(3)
  pd.set_option("display.width", 300)
  print(f"{len(lab)} labelled failures; ref = R0 repaired G1 (s1); Holm over the N* comparisons")
  print(t.to_string(index=False))
  # N6: selective prediction by agreement among repaired runs
  if len(allr) > 2:
    M = np.array(allr)
    v = P[f"N3 window vote G1+G0 x{len(allr)}"]
    agree = np.array([np.sum(np.abs(M[:, i] - v[i]) <= 2) for i in range(len(g))])
    print("\nN6 selective prediction (window vote over all repaired runs; answer only if >= m runs agree within +-2):")
    for m in range(len(allr), 0, -1):
      sel = agree >= m
      print(f"  m>={m:2d}: coverage {sel.mean():.2f}  +-2 {np.mean(np.abs(v[sel] - g[sel]) <= 2):.3f}  exact {np.mean(v[sel] == g[sel]):.3f}")
  OUT.mkdir(parents=True, exist_ok=True)
  t.to_csv(OUT / "comparison.csv", index=False)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("cmd", choices=["induce", "exemplars", "segments", "candidates", "score"])
  ap.add_argument("--env-file", default=str(ROOT / "config/openai.env"))
  ap.add_argument("--think-effort", default="medium")
  ap.add_argument("--judge-for-induction", default="gpt6luna-fixed-v0.3/G1")
  ap.add_argument("--k", type=int, default=3)
  ap.add_argument("--samples", type=int, default=5)
  ap.add_argument("--budget", type=int, default=20)
  ap.add_argument("--max-must", type=int, default=14)
  ap.add_argument("--max-candidates", type=int, default=4)
  args = ap.parse_args()
  {"induce": induce, "exemplars": exemplars, "segments": segments, "candidates": candidates, "score": score}[args.cmd](args)


if __name__ == "__main__":
  main()
