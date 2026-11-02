#!/usr/bin/env python
"""Which gold / trajectory properties predict whether gold helps the judge? (direction 4)

Unit of analysis: one labelled failure. For every judge configuration (judge_version) the
paired gold effect is  d = outcome(Gk) − outcome(G0)  on the same trajectory; d is then
averaged over configurations (less judge noise) and related to the properties from
``scripts/gold_features.py``. Outcomes:
  exact  root step exactly right            (higher = better)
  err    log(1 + |pred − human| steps)      (lower  = better; d < 0 means gold helped)
  leaf   primary leaf correct               (higher = better)

Tests per (cond, outcome, property): Spearman on the averaged d, plus OLS on the
un-averaged d with configuration fixed effects and trajectory-clustered SEs. p-values are
BH-corrected across all tests of the run (q). Binned tables show the effect by tercile.

Divergence features (G2): failed step i "matches" the gold if some gold screenshot is within
``--thresh`` dHash bits; div_step = first unmatched failed step; div_gap = div_step − human
root step. The pull test asks whether, when gold changes the judge's answer, the new answer
moves toward div_step or toward the human root step.

  python scripts/gold_properties.py                      # Luna configs, full 204
  python scripts/gold_properties.py --versions qwen3vl32b-fixed-v0.3 --split all
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

DEFAULT_VERSIONS = ["gpt6luna-fixed-v0.3", "gpt6luna-od-v0.3", "gpt6luna-think-fixed-v0.3", "gpt6luna-think-od-v0.3"]
OUTCOMES = {"exact": +1, "err": -1, "leaf": +1}  # sign: +1 → higher is better
PROPS = {
  "G1": ["n_steps", "human_step", "root_rel", "h_steps", "h_ratio", "h_overlap"],
  "G2": ["n_steps", "human_step", "root_rel", "g_steps", "g_ratio", "match_frac", "div_step", "div_gap", "abs_div_gap"],
}
CATS = {"G1": ["agent", "family"], "G2": ["agent", "family", "gold_agent", "same_agent"]}


def load_scores(versions: list[str], split: str, metrics: Path) -> pd.DataFrame:
  parts = []
  for v in versions:
    f = metrics / v / split / "scored_rows.csv"
    if not f.exists():
      f = metrics / v / "scored_rows.csv"
    parts.append(pd.read_csv(f))
  d = pd.concat(parts, ignore_index=True)
  d = d[d.parse_ok.astype(bool)].copy()
  d["exact"] = d.step_exact.astype(float)
  d["err"] = np.log1p((d.pred_step - d.human_step).abs())
  d["leaf"] = d.leaf_acc.astype(float)
  return d


def paired(d: pd.DataFrame, cond: str) -> pd.DataFrame:
  key = ["judge_version", "agent", "task_id"]
  base = d[d.condition == "G0"].set_index(key)
  gk = d[d.condition == cond].set_index(key)
  j = gk.join(base, rsuffix="_g0", how="inner")
  out = pd.DataFrame(index=j.index)
  for o in OUTCOMES:
    out[o] = j[o] - j[f"{o}_g0"]
  out["pred_g0"], out["pred_gk"], out["human_step"] = j.pred_step_g0, j.pred_step, j.human_step
  return out.reset_index()


def add_divergence(f: pd.DataFrame, thresh: int) -> pd.DataFrame:
  f = f.copy()
  div, frac = [], []
  for v in f.vis_d:
    if v is None or (isinstance(v, float) and np.isnan(v)) or len(v) == 0:
      div.append(np.nan); frac.append(np.nan); continue
    vv = np.array([np.nan if x is None else x for x in v], dtype=float)
    unmatched = np.where(vv > thresh)[0]
    div.append(float(unmatched[0] + 1) if len(unmatched) else float(len(vv) + 1))
    frac.append(float(np.nanmean(vv <= thresh)))
  f["div_step"], f["match_frac"] = div, frac
  f["div_gap"] = f.div_step - f.human_step
  f["abs_div_gap"] = f.div_gap.abs()
  return f


def spearman_rows(per: pd.DataFrame, long: pd.DataFrame, cond: str) -> list[dict]:
  rows = []
  for p in PROPS[cond]:
    for o, sign in OUTCOMES.items():
      x = per[[p, o]].dropna()
      if len(x) < 15 or x[p].nunique() < 3:
        continue
      rho, pv = stats.spearmanr(x[p], x[o])
      ll = long[[p, o, "judge_version", "key"]].dropna()
      ll = ll.assign(z=(ll[p] - ll[p].mean()) / ll[p].std())
      m = smf.ols(f"{o} ~ z + C(judge_version)", data=ll).fit(cov_type="cluster", cov_kwds={"groups": ll.key.astype("category").cat.codes})
      rows.append({"cond": cond, "property": p, "outcome": o, "n": len(x), "rho": rho, "p_spearman": pv,
                   # helps_per_sd: change in the gold effect per +1 SD of the property, signed so + = gold helps more
                   "helps_per_sd": sign * m.params["z"], "p_ols": m.pvalues["z"]})
  return rows


def terciles(per: pd.DataFrame, cond: str) -> pd.DataFrame:
  out = []
  for p in PROPS[cond]:
    x = per.dropna(subset=[p])
    if x[p].nunique() < 3:
      continue
    try:
      bins = pd.qcut(x[p], 3, duplicates="drop")
    except ValueError:
      continue
    g = x.groupby(bins, observed=True)
    t = g[list(OUTCOMES)].mean()
    t.insert(0, "n", g.size())
    t.insert(0, "bin", [str(b) for b in t.index])
    t.insert(0, "property", p)
    out.append(t.reset_index(drop=True))
  return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def categorical(per: pd.DataFrame, cond: str) -> pd.DataFrame:
  out = []
  for c in CATS[cond]:
    if c not in per or per[c].nunique() < 2:
      continue
    g = per.groupby(c)
    t = g[list(OUTCOMES)].mean()
    t.insert(0, "n", g.size())
    kw = [stats.kruskal(*[grp[o].dropna() for _, grp in g if len(grp) >= 5]).pvalue if sum(len(grp) >= 5 for _, grp in g) >= 2 else np.nan
          for o in OUTCOMES]
    t = t.reset_index().rename(columns={c: "level"})
    t.insert(0, "property", c)
    t["p_kruskal(exact,err,leaf)"] = ", ".join(f"{p:.3f}" for p in kw)
    out.append(t)
  return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def pull_test(long: pd.DataFrame) -> pd.DataFrame:
  """Among G2 answers that differ from G0, does the new answer move toward the gold divergence step?"""
  x = long.dropna(subset=["div_step", "pred_g0", "pred_gk"])
  x = x[(x.pred_g0 != x.pred_gk) & (x.div_step != x.human_step)]
  to_div = (x.pred_gk - x.div_step).abs() < (x.pred_g0 - x.div_step).abs()
  to_root = (x.pred_gk - x.human_step).abs() < (x.pred_g0 - x.human_step).abs()
  rows = []
  for name, m in [("all", np.ones(len(x), bool)), ("div before root", (x.div_step < x.human_step).values),
                  ("div after root", (x.div_step > x.human_step).values)]:
    n = int(m.sum())
    if n == 0:
      continue
    rows.append({"subset": name, "changed_answers": n, "moved_toward_div": to_div[m].mean(),
                 "moved_toward_root": to_root[m].mean(),
                 "p_binom(toward_div vs toward_root)": stats.binomtest(int((to_div & ~to_root)[m].sum()),
                                                                     int((to_div ^ to_root)[m].sum()) or 1).pvalue})
  return pd.DataFrame(rows)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--versions", nargs="+", default=DEFAULT_VERSIONS)
  ap.add_argument("--split", default="all")
  ap.add_argument("--metrics", type=Path, default=Path("data/metrics"))
  ap.add_argument("--features", type=Path, default=Path("data/analysis/gold_features.parquet"))
  ap.add_argument("--thresh", type=int, default=20, help="dHash bits (of 256) within which a failed step matches a gold screenshot")
  ap.add_argument("--out", type=Path, default=Path("data/analysis/gold_properties"))
  args = ap.parse_args()

  f = pd.read_parquet(args.features)
  f["vis_d"] = f.vis_d.apply(lambda v: list(v) if v is not None else None)
  thresh = args.thresh
  f = add_divergence(f, thresh)
  f["family"] = f.human_tag.str[0]
  d = load_scores(args.versions, args.split, args.metrics)

  args.out.mkdir(parents=True, exist_ok=True)
  lines = [f"# Gold-property analysis", "", f"versions: {', '.join(args.versions)} (split {args.split})",
           f"divergence threshold: {thresh} dHash bits of 256 (first-screenshot distance: median "
           f"{np.nanmedian(f.vis_d0.astype(float)):.0f}, p95 {np.nanpercentile(f.vis_d0.astype(float), 95):.0f})", ""]
  all_tests = []
  for cond in ["G1", "G2"]:
    long = paired(d, cond).merge(f.drop(columns=["human_step"]), on=["agent", "task_id"], how="left")
    long["key"] = long.agent + "/" + long.task_id
    per = long.groupby(["agent", "task_id"], as_index=False).agg(
      {**{o: "mean" for o in OUTCOMES}, **{p: "first" for p in PROPS[cond] + [c for c in CATS[cond] if c in long]}})
    lines += [f"## {cond} vs G0  (n = {len(per)} trajectories × {long.judge_version.nunique()} configs)", "",
              "Mean paired effect (+exact/+leaf = gold helped; −err = gold helped): " +
              ", ".join(f"{o} {per[o].mean():+.3f}" for o in OUTCOMES), ""]
    all_tests += spearman_rows(per, long, cond)
    t = terciles(per, cond)
    c = categorical(per, cond)
    t.to_csv(args.out / f"terciles_{cond}.csv", index=False)
    c.to_csv(args.out / f"categorical_{cond}.csv", index=False)
    lines += ["### By tercile (mean paired effect)", "", t.round(3).to_markdown(index=False), "",
              "### By category", "", c.round(3).to_markdown(index=False), ""]
    if cond == "G2":
      pt = pull_test(long)
      pt.to_csv(args.out / "pull_test_G2.csv", index=False)
      lines += ["### Pull test: when G2 changes the judge's root step, where does it move?", "",
                pt.round(3).to_markdown(index=False), ""]

  tests = pd.DataFrame(all_tests)
  tests["q_spearman"] = multipletests(tests.p_spearman, method="fdr_bh")[1]
  tests["q_ols"] = multipletests(tests.p_ols, method="fdr_bh")[1]
  tests = tests.sort_values("p_ols")
  tests.to_csv(args.out / "tests.csv", index=False)
  lines[5:5] = ["## All property tests (sorted by OLS p; q = BH over all tests)", "",
                "`helps_per_sd` > 0: gold helps more as the property increases (per 1 SD).", "",
                tests.round(3).to_markdown(index=False), ""]
  (args.out / "report.md").write_text("\n".join(lines))
  print("\n".join(lines))


if __name__ == "__main__":
  main()
