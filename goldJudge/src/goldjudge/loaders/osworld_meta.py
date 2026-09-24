"""OSWorld task metadata + OSWorld-Human references from pinned git checkouts.

Expects the layout produced by ``scripts/fetch_osworld_sources.sh``::

  <root>/OSWorld/evaluation_examples/{test_nogdrive,test_all,test_infeasible}.json
  <root>/OSWorld/evaluation_examples/examples/<domain>/<task_id>.json
  <root>/OSWorld/desktop_env/evaluators/metrics/*.py
  <root>/osworld-human/<domain>/<task_id>.json
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from goldjudge.schema import HumanReference, TaskMeta

DEFAULT_ROOT = Path(os.environ.get("OSWORLD_SRC", "/ocean/projects/cis260247p/rgupta19/data/osworld_src"))


class OSWorldMeta:
  def __init__(self, root: Path | str = DEFAULT_ROOT):
    self.root = Path(root)
    self.osworld = self.root / "OSWorld"
    self.human_root = self.root / "osworld-human"
    self.examples = self.osworld / "evaluation_examples"
    self._infeasible: set[str] | None = None

  # -- task lists ----------------------------------------------------------
  def task_list(self, split: str = "test_nogdrive") -> dict[str, list[str]]:
    return json.loads((self.examples / f"{split}.json").read_text(encoding="utf-8"))

  def domain_of(self, task_id: str) -> str | None:
    for domain, ids in self.task_list("test_all").items():
      if task_id in ids:
        return domain
    return None

  @property
  def infeasible_ids(self) -> set[str]:
    if self._infeasible is None:
      p = self.examples / "test_infeasible.json"
      self._infeasible = (
        {t for ids in json.loads(p.read_text(encoding="utf-8")).values() for t in ids}
        if p.exists()
        else set()
      )
    return self._infeasible

  # -- per-task ------------------------------------------------------------
  def task_path(self, domain: str, task_id: str) -> Path:
    return self.examples / "examples" / domain / f"{task_id}.json"

  def human_path(self, domain: str, task_id: str) -> Path:
    return self.human_root / domain / f"{task_id}.json"

  def load_human(self, domain: str, task_id: str) -> HumanReference | None:
    p = self.human_path(domain, task_id)
    if not p.exists():
      return None
    data = json.loads(p.read_text(encoding="utf-8"))
    hgt = data.get("human-ground-truth") or {}
    grouped: list[list[str]] = []
    for g in hgt.get("grouped-action") or []:
      grouped.append([str(x) for x in g] if isinstance(g, list) else [str(g)])
    return HumanReference(
      task_id=task_id,
      domain=domain,
      single_action=[str(x) for x in hgt.get("single-action") or []],
      grouped_action=grouped,
    )

  def load_task(self, domain: str, task_id: str, *, with_human: bool = True) -> TaskMeta:
    p = self.task_path(domain, task_id)
    data = json.loads(p.read_text(encoding="utf-8"))
    return TaskMeta(
      task_id=task_id,
      domain=domain,
      instruction=str(data.get("instruction", "")),
      evaluator=data.get("evaluator") or {},
      related_apps=list(data.get("related_apps") or []),
      config=list(data.get("config") or []),
      infeasible=task_id in self.infeasible_ids
      or str((data.get("evaluator") or {}).get("func", "")) == "infeasible",
      human=self.load_human(domain, task_id) if with_human else None,
      raw=data,
    )

  # -- evaluator source ----------------------------------------------------
  @lru_cache(maxsize=None)
  def metric_source(self, func_name: str) -> str | None:
    """Return the Python source of an evaluator metric function, if found."""
    import ast

    mdir = self.osworld / "desktop_env" / "evaluators" / "metrics"
    for py in sorted(mdir.glob("*.py")):
      src = py.read_text(encoding="utf-8", errors="replace")
      if f"def {func_name}(" not in src:
        continue
      tree = ast.parse(src)
      for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
          return ast.get_source_segment(src, node)
    return None

  def evaluator_funcs(self, meta: TaskMeta) -> list[str]:
    f = meta.evaluator.get("func")
    if isinstance(f, list):
      return [str(x) for x in f]
    return [str(f)] if f else []
