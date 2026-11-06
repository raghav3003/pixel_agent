#!/usr/bin/env python3
"""Every number in the paper's harness, audit and selective-prediction tables, from the saved outputs.

  python scripts/paper_tables.py --out /ocean/projects/cis260247p/rgupta19/paper_cua/build

Writes numbers.json (all predictors, all metrics, paired tests) and LaTeX table bodies. Populations are reported per
row; paired sign tests are on the rows both predictors cover. No model calls.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness2 import folds_by_task, fuse, load_topk, prior_table  # noqa: E402
from round3 import vote  # noqa: E402
from score_subgoals import load_judge, load_localize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"


def labels() -> pd.DataFrame:
  lab = pd.read_parquet(D / "index/episodes.parquet")
  lab = lab[(lab.source == "cuadebug") & lab.root_error_step.notna()].reset_index(drop=True)
  ann = [json.loads((Path(r.dir) / "human_annotation.json").read_text(encoding="utf-8")) for r in lab.itertuples()]
  lab["seeded"] = [bool(a.get("chosen_debugger")) for a in ann]
  lab["conf"] = [a.get("confidence", "") for a in ann]
  dev = {(r["agent"], r["task_id"]) for r in json.loads((D / "index/h50.json").read_text())["rows"]}
  lab["dev"] = [(a, t) in dev for a, t in zip(lab.agent, lab.task_id)]
  return lab


def from_dir(pattern: str) -> dict:
  out = {}
  for f in glob.glob(str(D / pattern), recursive=True):
    r = json.loads(Path(f).read_text(encoding="utf-8"))
    if "agent" in r and isinstance(r.get("out"), dict) and r["out"].get("root_step") is not None:
      out[(r["agent"], r["task_id"])] = r["out"]["root_step"]
  return out


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default="/ocean/projects/cis260247p/rgupta19/paper_cua/build")
  args = ap.parse_args()
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  lab = labels()
  keys = list(zip(lab.agent, lab.task_id))
  g = lab.root_error_step.astype(float).values
  n = lab.n_steps.values.astype(float)

  def arr(d: dict) -> np.ndarray:
    return np.array([d[k] if d.get(k) not in (None, -1) else np.nan for k in keys], float)

  J = lambda v: arr(load_judge(v))  # noqa: E731
  P: dict[str, dict] = {}

  def add(name, family, p=None, runs=None, note=""):
    P[name] = {"family": family, "p": p, "runs": runs, "note": note}

  # references and baselines
  add("Judge, no reference (G0)", "judge", J("gpt6luna-fixed-v0.3/G0"))
  add("Judge + human steps (G1)", "judge", J("gpt6luna-fixed-v0.3/G1"))
  add("Judge + successful AI run (G2)", "judge", J("gpt6luna-fixed-v0.3/G2"))
  add("Judge G0, medium reasoning", "judge", J("gpt6luna-think-fixed-v0.3/G0"))
  add("Judge G1, medium reasoning", "judge", J("gpt6luna-think-fixed-v0.3/G1"))
  add("Judge G0, corrected screenshot labels", "audit", J("gpt6luna-fixed-after-v0.3/G0"))
  add("Judge G1, corrected screenshot labels", "audit", J("gpt6luna-fixed-after-v0.3/G1"))
  add("Judge G1, repaired inputs (5 runs)", "audit", runs=[J(f"gpt6luna-repaired-v0.3-s{k}/G1") for k in range(1, 6)])
  add("Judge G0, repaired inputs (5 runs)", "audit", runs=[J(f"gpt6luna-repaired-v0.3-s{k}/G0") for k in range(1, 6)])
  add("Constant step 3", "baseline", np.minimum(3, n))
  add("Constant step 1", "baseline", np.ones_like(n))
  # CUADebug reproductions
  add("CUADebug prompt-only (reproduced)", "reproduction", arr(from_dir("cuadebugger/gpt6luna-cuad-v0.1/prompt/*/*.json")))
  add("CUADebugger tool loop (reproduced)", "reproduction", arr(from_dir("cuadebugger/gpt6luna-cuad-v0.1/tool/*/*.json")))
  # decomposition
  loc = load_localize(D / "subgoals/gpt6luna-sg-v0.2")
  if not loc.empty:
    d = {(a, t): x["rule_step"] for a, t, x in zip(loc.agent, loc.task_id, loc.rule_verified)}
    add("Required states, tracked on screen: rule", "decomposition", arr(d))
    d = {(a, t): x.get("root_step") for a, t, x in zip(loc.agent, loc.task_id, loc.llm)}
    add("Required states, tracked on screen: + LLM in window", "decomposition", arr(d))
  tg = {}
  for f in glob.glob(str(D / "textgoals/gpt6luna-text-v0.3/align/*/*.r1.json")):
    r = json.loads(Path(f).read_text())
    tg[(r["agent"], r["task_id"])] = r["summary"].get("step")
  add("Text subgoals, first divergence from gold", "decomposition", arr(tg), note="dev only")
  sc = {}
  for f in glob.glob(str(D / "textgoals/gpt6luna-text-v0.3/screen_v0.2/*/*.json")):
    r = json.loads(Path(f).read_text())
    sc[(r["agent"], r["task_id"])] = r["summary"].get("step")
  add("Text subgoals + screen check", "decomposition", arr(sc), note="dev only")
  # step / state checks
  st = {}
  for f in glob.glob(str(D / "stepcheck/gpt6luna-step-v0.1/*/*.json")):
    r = json.loads(Path(f).read_text())
    if "verdicts" in r:
      m = [x["step"] for x in r["verdicts"] if x["match"] == "no"]
      if m:
        st[(r["agent"], r["task_id"])] = m[0]
  add("Per-step intent vs outcome, first mismatch", "checks", arr(st))
  add("Method constraints from evaluator", "checks", arr(from_dir("constraints/gpt6luna-cons-v0.1/check/*/*.json")))
  add("Evaluator-anchored backward trace", "checks", arr(from_dir("harness2/gpt6luna-h2-v0.1/evtrace/*-*/*.json")))
  # candidates and selection
  add("Candidates + local ROOT/INHERITED adjudication", "candidates", arr(from_dir("harness2/gpt6luna-h2-v0.1/adjudicate/*/*.json")))
  add("Candidate selection, pooled runs (N5)", "candidates", J("gpt6luna-r3-n5-v0.3/G1"))
  add("Segment focus (N4)", "candidates", J("gpt6luna-r3-n4-v0.3/G1"))
  # aggregation
  topk = load_topk("gpt6luna-fixed-after-top5-v0.3/G1")
  lab_f = lab.assign(g=g, fold=lab.task_id.map(folds_by_task(list(lab.task_id), 5)))
  fus = np.full(len(lab), np.nan)
  for f in range(5):
    tr, te = lab_f[lab_f.fold != f], lab_f[lab_f.fold == f]
    pri = prior_table(tr.rename(columns={"root_error_step": "g"}) if "g" not in tr else tr)
    best, bacc = (1.0, .05), -1
    for lam in (0.0, .25, .5, 1.0, 2.0):
      for eps in (.01, .05, .2, .5):
        h = [abs(fuse(topk[keys[i]]["cands"], int(n[i]), pri.get(lab.agent[i], pri["__all__"]), lam, eps) - g[i]) <= 2
             for i in tr.index if topk.get(keys[i], {}).get("cands")]
        if h and np.mean(h) > bacc:
          best, bacc = (lam, eps), np.mean(h)
    for i in te.index:
      if topk.get(keys[i], {}).get("cands"):
        fus[i] = fuse(topk[keys[i]]["cands"], int(n[i]), pri.get(lab.agent[i], pri["__all__"]), *best)
  add("Judge top-k x position prior (CV)", "aggregation", fus)
  rep = [J(f"gpt6luna-repaired-v0.3-s{k}/{c}") for k in range(1, 6) for c in ("G1", "G0")]
  add("Window vote over 10 judge runs", "aggregation", np.array([vote(list(c)) or np.nan for c in np.array(rep).T], float))
  # context
  for tag, nm in [("S", "subgoals"), ("F", "step flags"), ("SF", "subgoals + step flags"), ("C", "constraints"), ("SFC", "all three")]:
    add(f"Judge G0 + {nm} as context", "context", J(f"gpt6luna-fixed-after-ctx{tag}-v0.3/G0"))
  add("Judge G0 + checkpoints as context", "context", J("gpt6luna-r3-ckpt-v0.3/G0"))
  # learning from labels
  add("Induced labelling conventions (N1, CV)", "labels", J("gpt6luna-r3-n1-v0.3/G1"))
  add("Retrieved labelled exemplars (N2, CV; 4 runs)", "labels",
      runs=[J(f"gpt6luna-r3-n2{s}-v0.3/G1") for s in ("", "-s2", "-s3", "-s4")])
  add("Conventions + exemplars", "labels", J("gpt6luna-r3-n12-v0.3/G1"))

  ref = P["Judge + human steps (G1)"]["p"]
  rh = np.abs(ref - g) <= 2
  rows = []
  for name, v in P.items():
    runs = v["runs"] if v["runs"] is not None else [v["p"]]
    mets = []
    for p in runs:
      e = np.abs(p - g)
      # population: dev-only methods on the 50 dev failures, G2 on the failures that have a gold run, otherwise all
      # 204; a missing prediction inside the population counts as a miss (e is NaN, so every comparison is False)
      cov = lab.dev.values if v["note"] == "dev only" else (np.isfinite(p) if "G2" in name else np.ones(len(g), bool))
      mets.append({"exact": np.mean(e[cov] == 0), "pm1": np.mean(e[cov] <= 1), "pm2": np.mean(e[cov] <= 2),
                   "pm5": np.mean(e[cov] <= 5), "mae": np.nanmean(e), "signed": np.nanmean(p - g),
                   "late": np.mean((p - g)[cov] > 2), "early": np.mean((p - g)[cov] < -2), "n": int(cov.sum()),
                   "pm2_claude": np.mean(e[cov & (lab.agent.values == "claude-sonnet-4-5")] <= 2),
                   "pm2_len10": np.mean(e[cov & (n > 10)] <= 2)})
    row = {"name": name, "family": v["family"], "note": v["note"], "runs": len(runs),
           "predicted": int(np.mean([np.isfinite(p).sum() for p in runs])),
           **{k: float(np.mean([m[k] for m in mets])) for k in mets[0]}}
    if len(runs) > 1:
      row["pm2_sd"] = float(np.std([m["pm2"] for m in mets], ddof=1))
      row["exact_sd"] = float(np.std([m["exact"] for m in mets], ddof=1))
    p = runs[0]
    both = np.isfinite(p) & np.isfinite(ref)
    hit = np.abs(p - g) <= 2
    a, b = int((hit & ~rh & both).sum()), int((rh & ~hit & both).sum())
    row["vs_G1"] = f"+{a}/-{b}"
    row["p_vs_G1"] = float(binomtest(a, a + b).pvalue) if a + b else 1.0
    rows.append(row)
  T = pd.DataFrame(rows)
  (out / "numbers.json").write_text(T.to_json(orient="records", indent=1), encoding="utf-8")

  # selective prediction (all 10 repaired runs)
  M = np.array(rep)
  v = np.array([vote(list(c)) or np.nan for c in M.T], float)
  agree = np.array([np.sum(np.abs(M[:, i] - v[i]) <= 2) for i in range(len(g))])
  sel = [{"min_agree": m, "coverage": float(np.mean(agree >= m)), "pm2": float(np.mean(np.abs(v[agree >= m] - g[agree >= m]) <= 2)),
          "exact": float(np.mean(v[agree >= m] == g[agree >= m]))} for m in (10, 9, 8, 7, 6, 5)]
  (out / "selective.json").write_text(json.dumps(sel, indent=1), encoding="utf-8")

  # strata for the best judge
  e = np.abs(ref - g)
  strata = {"length<=10": float(np.mean(e[n <= 10] <= 2)), "length>10": float(np.mean(e[n > 10] <= 2)),
            "length>35": float(np.mean(e[n > 35] <= 2)), "const3_length>10": float(np.mean(np.abs(np.minimum(3, n) - g)[n > 10] <= 2)),
            "seeded": float(np.mean(e[lab.seeded.values] <= 2)), "unseeded": float(np.mean(e[~lab.seeded.values] <= 2)),
            "conf_high": float(np.mean(e[lab.conf.values == "high"] <= 2)), "conf_other": float(np.mean(e[lab.conf.values != "high"] <= 2)),
            "n_seeded": int(lab.seeded.sum()), "n_high": int((lab.conf == "high").sum()), "n_len10": int((n <= 10).sum())}
  for fam in sorted(lab.taxonomy_tag.str[:1].unique()):
    m = lab.taxonomy_tag.str[:1].values == fam
    strata[f"family_{fam}"] = [int(m.sum()), float(np.mean(e[m] <= 2))]
  (out / "strata.json").write_text(json.dumps(strata, indent=1), encoding="utf-8")

  pd.set_option("display.width", 250)
  cols = ["name", "family", "runs", "n", "exact", "pm1", "pm2", "pm5", "mae", "signed", "late", "pm2_len10", "vs_G1", "p_vs_G1"]
  print(T[cols].round(3).to_string(index=False))
  print(json.dumps(sel, indent=1))
  print(json.dumps(strata, indent=1))


if __name__ == "__main__":
  main()
