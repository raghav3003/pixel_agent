#!/usr/bin/env python3
"""Score judge outputs against CUADebug human annotations.

Metrics per (judge_version, condition, agent) and pooled:
  parse_rate      fraction of verdicts that parsed with a valid leaf and in-range root step
  step_exact      predicted root_step == human root_error_step
  step_pm2        |pred - human| <= 2
  family_acc      v2 family of predicted primary == v2 family of the mapped human tag
  leaf_acc        predicted primary ∈ acceptable v2 leaves for the human tag
  leaf_any_acc    leaf_acc OR any secondary leaf ∈ acceptable set
  joint           leaf_acc AND step_pm2
  cuadebug_family_acc  family agreement in CUADebug's own P/G/R/S/IF scheme (via reverse map)
  root_seen       the judge saw the screenshot of the human root step (seed or requested);
                  blank for rows written before this field existed

Usage:
  python scripts/score_judge.py --judge-version qwen3vl32b-v0.1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from goldjudge.taxonomy import load_taxonomy

V2_TO_CUADEBUG_FAMILY = {"P": "P", "G": "G", "R": "R", "E": "S"}


def load_rows(root: Path, judge_version: str | None) -> pd.DataFrame:
  rows = []
  for p in root.rglob("*.jsonl"):
    if judge_version and p.parts[-3] != judge_version:
      continue
    for line in p.open(encoding="utf-8"):
      try:
        rows.append(json.loads(line))
      except json.JSONDecodeError:
        continue
  return pd.DataFrame(rows)


def score(df: pd.DataFrame) -> pd.DataFrame:
  tax = load_taxonomy()
  # Leaf ids are only comparable within one taxonomy version (v2.1 renumbered the G family).
  tv = df["taxonomy_version"] if "taxonomy_version" in df else pd.Series(None, index=df.index)
  stale = sorted(df.loc[tv != tax.version, "judge_version"].unique())
  if stale:
    print(f"WARNING: judge versions {stale} were not labelled under taxonomy {tax.version!r}; leaf metrics may be wrong")
  recs = []
  for r in df.itertuples():
    v, h = r.verdict, r.human
    if not h:
      continue
    acceptable = tax.map_from("cuadebug", h["taxonomy_tag"])
    human_fams = {tax.family_of(l) for l in acceptable if tax.family_of(l)}
    pred = v["primary"]
    pred_fam = tax.family_of(pred)
    sec = [s["leaf"] for s in v.get("secondary") or []]
    step_ok = v["root_step"] >= 1
    d = abs(v["root_step"] - h["root_error_step"]) if step_ok else 10**6
    cd_fam_pred = V2_TO_CUADEBUG_FAMILY.get(pred_fam or "", None)
    cd_fam_human = "IF" if h["taxonomy_tag"].startswith("IF") or h["taxonomy_tag"].lower().startswith("infeasible") else h["taxonomy_tag"][:1]
    if pred == "R9":
      cd_fam_pred = "IF" if cd_fam_human == "IF" else cd_fam_pred
    recs.append(
      {
        "judge_version": r.judge_version, "condition": r.condition, "agent": r.agent, "task_id": r.task_id,
        "parse_ok": bool(v.get("parse_ok", False)),
        "step_exact": d == 0, "step_pm2": d <= 2,
        "family_acc": bool(pred_fam) and pred_fam in human_fams,
        "leaf_acc": pred in acceptable,
        "leaf_any_acc": pred in acceptable or any(s in acceptable for s in sec),
        "joint": (pred in acceptable) and d <= 2,
        "cuadebug_family_acc": cd_fam_pred == cd_fam_human,
        "root_seen": (int(h["root_error_step"]) in set(r.viewed_agent_steps)) if isinstance(getattr(r, "viewed_agent_steps", None), list) else None,
        "human_tag": h["taxonomy_tag"], "pred": pred, "human_step": h["root_error_step"], "pred_step": v["root_step"],
        "n_images": r.n_images, "prompt_tokens": (r.usage or {}).get("prompt_tokens"),
      }
    )
  return pd.DataFrame(recs)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--judge-dir", type=Path, default=Path("data/judge"))
  ap.add_argument("--judge-version", default=None)
  ap.add_argument("--out", type=Path, default=None)
  ap.add_argument("--pilot", type=Path, default=None, help="JSON with rows [{agent, task_id}]: score only these trajectories")
  ap.add_argument("--reparse", action="store_true", help="re-parse stored raw text with the current parser (outputs untouched)")
  args = ap.parse_args()

  df = load_rows(args.judge_dir, args.judge_version)
  if args.pilot:
    keys = {(r["agent"], r["task_id"]) for r in json.loads(args.pilot.read_text())["rows"]}
    df = df[[(a, t) in keys for a, t in zip(df.agent, df.task_id)]]
  if df.empty:
    raise SystemExit("no judge outputs found")
  if args.reparse:
    from goldjudge.judge.verdict import parse_verdict

    tax = load_taxonomy()
    df["verdict"] = [
      parse_verdict(r.raw, tax, n_steps=r.n_steps, truncated=(r.usage or {}).get("stop_reason") == "max_tokens").model_dump(exclude={"raw"})
      for r in df.itertuples()
    ]
  s = score(df)
  metrics = ["parse_ok", "step_exact", "step_pm2", "family_acc", "leaf_acc", "leaf_any_acc", "joint", "cuadebug_family_acc", "root_seen"]
  s["root_seen"] = s["root_seen"].astype("float")
  by = s.groupby(["judge_version", "condition", "agent"])[metrics].mean().round(3)
  by["n"] = s.groupby(["judge_version", "condition", "agent"]).size()
  pooled = s.groupby(["judge_version", "condition"])[metrics].mean().round(3)
  pooled["n"] = s.groupby(["judge_version", "condition"]).size()
  pd.set_option("display.width", 200)
  print("== per agent ==\n", by.to_string())
  print("\n== pooled ==\n", pooled.to_string())
  print("\n== confusion (human family -> predicted family) ==")
  print(pd.crosstab(s.human_tag.str[:1], s.pred.str[:1]).to_string())
  if args.out:
    args.out.mkdir(parents=True, exist_ok=True)
    s.to_csv(args.out / "scored_rows.csv", index=False)
    by.to_csv(args.out / "metrics_by_agent.csv")
    pooled.to_csv(args.out / "metrics_pooled.csv")


if __name__ == "__main__":
  main()
