#!/usr/bin/env python3
"""Run the judge over failed trajectories under one gold condition.

Outputs one JSONL per (judge_version, condition, agent) under ``--out``; existing
rows are skipped (resume) and never overwritten — bump ``--judge-version`` for a
new run.

Examples (vLLM endpoint on a compute node):
  export JUDGE_BASE_URL=http://<node>:8000/v1 JUDGE_MODEL=Qwen/Qwen3-VL-32B-Instruct
  python scripts/run_judge.py --cond G0 --agents gemini-2.5-pro --limit 5
  python scripts/run_judge.py --cond G2 --judge-version qwen3vl32b-v0.1

On-demand screenshots (Anthropic backend): the bundle carries a seed of screenshots and
the judge requests others with a tool, in every condition:
  python scripts/run_judge.py --cond G2 --harness on-demand --agent-images 8 --gold-images 4 \
      --view-budget 24 --thinking adaptive --max-tokens 16000 --judge-version sonnet5-long15-od-v0.3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from goldjudge.judge.bundle import BundleConfig, build_bundle
from goldjudge.judge.client import JudgeClient, JudgeClientConfig, load_env_file
from goldjudge.judge.context import context_sections
from goldjudge.judge.interactive import ScreenshotTool
from goldjudge.judge.prompts import PROMPT_VERSION, build_system_prompt, screenshot_rules
from goldjudge.judge.verdict import parse_verdict
from goldjudge.loaders import cuadebug, osworld_verified
from goldjudge.loaders.osworld_meta import OSWorldMeta
from goldjudge.schema import Trajectory
from goldjudge.taxonomy import load_taxonomy


def _load_gold(gold_df: pd.DataFrame, task_id: str, cond: str) -> Trajectory | None:
  rows = gold_df[(gold_df.task_id == task_id) & (gold_df.cond == cond)].sort_values("rank")
  if rows.empty:
    return None
  r = rows.iloc[0]
  pkg_root = Path(r.dir)
  # dir = <root>/<package>/.../<domain>/<task_id>
  package = r.package
  return osworld_verified.load_trajectory(pkg_root, package=package, run=r.run, agent=r.agent, horizon=None)


def estimate_prompt_tokens(system_chars: int, text_chars: int, n_images: int) -> int:
  """Rough first-turn prompt size: ~3.5 chars/token for text, ~1.2k tokens per 1280x720 image."""
  return round((system_chars + text_chars) / 3.5 + n_images * 1230)


def _read_eval_log(traj_dir: str) -> str | None:
  p = Path(traj_dir) / "eval.log"
  return p.read_text(encoding="utf-8", errors="replace") if p.exists() else None


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--cond", choices=["G0", "G1", "G2", "G3"], required=True)
  ap.add_argument("--judge-version", default="dev")
  ap.add_argument("--index", type=Path, default=Path("data/index"))
  ap.add_argument("--out", type=Path, default=Path("data/judge"))
  ap.add_argument("--source", default="cuadebug", help="cuadebug (labelled) or osworld_verified")
  ap.add_argument("--agents", nargs="*", default=None)
  ap.add_argument("--task-ids", nargs="*", default=None)
  ap.add_argument("--pilot", type=Path, default=None, help="JSON with rows [{agent, task_id}] to restrict to (from pilot selection)")
  ap.add_argument("--limit", type=int, default=None)
  ap.add_argument("--agent-images", type=int, default=16)
  ap.add_argument("--gold-images", type=int, default=8)
  ap.add_argument("--image-width", type=int, default=1280)
  ap.add_argument("--max-tokens", type=int, default=1500, help="includes thinking tokens; use >=16000 with thinking")
  ap.add_argument("--topk", type=int, default=0, help="also ask for top-k candidate steps with confidences + a symptom step")
  ap.add_argument("--context", nargs="*", default=[], choices=["subgoals", "stepflags", "constraints"],
                  help="analysis aids added to the input (goldjudge/judge/context.py); default none")
  ap.add_argument("--screen-labels", choices=["legacy", "after"], default="legacy",
                  help="after: label step_<t>.png as the screen after step t (what OSWorld saved); legacy = all runs before 2026-11-05")
  ap.add_argument("--harness", choices=["fixed", "on-demand"], default="fixed",
                  help="on-demand: --agent-images/--gold-images are the seed; the judge requests more via a tool")
  ap.add_argument("--view-budget", type=int, default=24, help="on-demand: extra screenshots per trajectory")
  ap.add_argument("--per-call", type=int, default=4, help="on-demand: max screenshots per tool call")
  ap.add_argument("--max-turns", type=int, default=10, help="on-demand: max model calls per trajectory")
  ap.add_argument("--thinking", choices=["default", "adaptive", "disabled"], default="default", help="Anthropic only")
  ap.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], default=None, help="Anthropic only")
  ap.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high", "xhigh", "max"], default=None,
                  help="OpenAI reasoning models only (e.g. gpt-6-luna; tool calls need 'none'); omit for vLLM")
  ap.add_argument("--provider", choices=["openai", "anthropic"], default=None, help="default: $JUDGE_PROVIDER or openai")
  ap.add_argument("--model", default=None, help="default: $JUDGE_MODEL")
  ap.add_argument("--env-file", type=Path, default=Path("config/judge.env"), help="KEY=VALUE file loaded into the environment (git-ignored)")
  ap.add_argument("--workers", type=int, default=1, help="concurrent requests (use >1 for vLLM; keep 1 for the Anthropic key)")
  ap.add_argument("--dry-run", action="store_true", help="build bundles, print sizes, do not call the model")
  args = ap.parse_args()

  tax = load_taxonomy()
  osw = OSWorldMeta()
  eps = pd.read_parquet(args.index / "episodes.parquet")
  gold_df = pd.read_parquet(args.index / "gold.parquet") if (args.index / "gold.parquet").exists() else pd.DataFrame()

  sel = eps[(eps.source == args.source) & (~eps.success)]
  if args.agents:
    sel = sel[sel.agent.isin(args.agents)]
  if args.task_ids:
    sel = sel[sel.task_id.isin(args.task_ids)]
  if args.pilot:
    keys = {(r["agent"], r["task_id"]) for r in json.loads(args.pilot.read_text())["rows"]}
    sel = sel[[(a, t) in keys for a, t in zip(sel.agent, sel.task_id)]]
  if args.cond in {"G2", "G3"} and not gold_df.empty:
    have = set(gold_df[gold_df.cond == args.cond].task_id)
    sel = sel[sel.task_id.isin(have)]
  sel = sel.sort_values(["agent", "domain", "task_id"])
  if args.limit:
    sel = sel.head(args.limit)

  on_demand = args.harness == "on-demand"
  if args.screen_labels == "after" and args.harness != "fixed":
    ap.error("--screen-labels after is implemented for the fixed harness only")
  cfg = BundleConfig(condition=args.cond, agent_image_budget=args.agent_images, gold_image_budget=args.gold_images, screen_labels=args.screen_labels,
                     image_max_width=args.image_width, on_demand=on_demand)
  load_env_file(args.env_file)
  ccfg = JudgeClientConfig(max_tokens=args.max_tokens, thinking=args.thinking, effort=args.effort,
                           reasoning_effort=args.reasoning_effort)
  if args.provider:
    ccfg.provider = args.provider
  if args.model:
    ccfg.model = args.model
  client = None if args.dry_run else JudgeClient(ccfg)
  has_gold_traj = args.cond in {"G2", "G3"}
  shots = screenshot_rules(budget=args.view_budget, per_call=args.per_call, has_gold=has_gold_traj) if on_demand else ""
  system_prompt = build_system_prompt(tax, has_reference=args.cond != "G0", screenshots=shots)
  if args.topk:  # extra fields for candidate-based harnesses; parsed from the saved raw reply
    system_prompt += (
      f"\n\nIn the same JSON object also include \"candidate_steps\": up to {args.topk} objects "
      "{\"step\": <int>, \"confidence\": <0-1>} for the steps most likely to be the root step, most likely first "
      "(the root_step you chose must be the first), and \"symptom_step\": <int>, the step where the failure first "
      "becomes visible.")

  out_dir = args.out / args.judge_version / args.cond
  if not args.dry_run:
    out_dir.mkdir(parents=True, exist_ok=True)

  import contextlib
  import threading
  from concurrent.futures import ThreadPoolExecutor, as_completed

  def judge_one(r) -> dict | None:
    if r.source == "cuadebug":
      traj = cuadebug.load_trajectory(Path(r.dir), r.package)
    else:
      traj = osworld_verified.load_trajectory(Path(r.dir), package=r.package, run=r.run, agent=r.agent, horizon=r.horizon)
    meta = osw.load_task(traj.domain, traj.task_id)
    gold = _load_gold(gold_df, traj.task_id, args.cond) if args.cond in {"G2", "G3"} else None
    bundle = build_bundle(traj, meta, osw, cfg, gold=gold, eval_log=_read_eval_log(traj.dir))
    context_have = None
    if args.context:  # analysis aids go right after the task section, before any reference and the trajectory
      ctx_text, context_have = context_sections(args.context, traj.agent, traj.task_id)
      bundle.content.insert(1, {"type": "text", "text": ctx_text})
    text_chars = sum(len(p["text"]) for p in bundle.content if p["type"] == "text")
    if args.dry_run:
      print(f"{r.agent} {traj.domain}/{traj.task_id}: steps={traj.n_steps} images={bundle.n_images} text_chars={text_chars} "
            f"est_prompt_tokens~{estimate_prompt_tokens(len(system_prompt), text_chars, bundle.n_images)} gold={bundle.meta['gold_key']}")
      return None
    t0 = time.time()
    tool = None
    if on_demand:
      tool = ScreenshotTool(agent=traj, gold=gold, budget=args.view_budget, per_call=args.per_call, max_width=args.image_width,
                            shown={"agent": bundle.agent_image_steps, "gold": bundle.gold_image_steps})
      resp = client.chat_interactive(system_prompt, bundle.content, [tool.definition()], tool, max_turns=args.max_turns)
    else:
      resp = client.chat(system_prompt, bundle.content)
    verdict = parse_verdict(resp.text, tax, n_steps=traj.n_steps, truncated=resp.usage.get("stop_reason") == "max_tokens")
    return {
      "judge_version": args.judge_version, "prompt_version": PROMPT_VERSION, "taxonomy_version": tax.version, "condition": args.cond,
      "harness": args.harness, "screen_labels": args.screen_labels, "context": args.context, "context_have": context_have, "topk": args.topk,
      "thinking": args.thinking, "effort": args.effort, "reasoning_effort": args.reasoning_effort,
      "source": traj.source, "package": traj.package, "agent": traj.agent, "domain": traj.domain,
      "task_id": traj.task_id, "n_steps": traj.n_steps, "n_images": bundle.n_images, "text_chars": text_chars,
      "gold_key": bundle.meta["gold_key"], "gold_steps": bundle.meta["gold_steps"],
      "agent_image_steps": bundle.agent_image_steps, "gold_image_steps": bundle.gold_image_steps,
      "viewed_agent_steps": tool.viewed("agent") if tool else bundle.agent_image_steps,
      "viewed_gold_steps": (tool.viewed("gold") if tool and gold else bundle.gold_image_steps),
      "screenshot_requests": tool.requests if tool else [], "n_turns": resp.n_turns,
      "model": resp.model, "usage": resp.usage, "latency_s": round(time.time() - t0, 2),
      "verdict": verdict.model_dump(exclude={"raw"}), "raw": resp.text,
      "human": traj.annotation.model_dump() if traj.annotation else None,
    }

  failed = 0
  for agent, grp in sel.groupby("agent"):
    out_path = out_dir / f"{agent}.jsonl"
    done: set[str] = set()
    if out_path.exists():
      for line in out_path.open(encoding="utf-8"):
        try:
          done.add(json.loads(line)["task_id"])
        except (json.JSONDecodeError, KeyError):
          continue
    todo = [r for r in grp.itertuples() if r.task_id not in done]
    lock = threading.Lock()
    with (contextlib.nullcontext() if args.dry_run else out_path.open("a", encoding="utf-8")) as fh, \
         ThreadPoolExecutor(max_workers=1 if args.dry_run else args.workers) as pool:
      futs = {pool.submit(judge_one, r): r.task_id for r in todo}
      for fut in tqdm(as_completed(futs), total=len(futs), desc=f"{args.cond}/{agent}"):
        try:
          row = fut.result()
        except Exception as exc:  # noqa: BLE001 — one bad trajectory must not kill the run; it stays missing and is retried on resume
          failed += 1
          print(f"FAILED {args.cond}/{agent}/{futs[fut]}: {type(exc).__name__}: {exc}", flush=True)
          continue
        if row is not None:
          with lock:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
  if failed:
    raise SystemExit(f"{failed} trajectories failed (not written; rerun the same command to retry them)")

if __name__ == "__main__":
  main()
