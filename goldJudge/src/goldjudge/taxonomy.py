"""Taxonomy v2 access: leaves, families, decision rules, and cross-scheme mappings."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "docs" / "taxonomy_v2.yaml"


@dataclass(frozen=True)
class Leaf:
  id: str
  name: str
  family: str
  definition: str


class Taxonomy:
  def __init__(self, data: dict[str, Any]):
    self.version: str = str(data.get("version", ""))
    self.question_order: list[str] = list(data.get("question_order") or [])
    self.families: dict[str, dict[str, str]] = dict(data.get("families") or {})
    self.leaves: dict[str, Leaf] = {
      lid: Leaf(id=lid, name=str(v["name"]), family=str(v["family"]), definition=str(v["definition"]))
      for lid, v in (data.get("leaves") or {}).items()
    }
    self.secondary_relations: list[str] = list(data.get("secondary_relations") or [])
    self.decision_rules: list[dict[str, Any]] = list(data.get("decision_rules") or [])
    self.mappings: dict[str, dict[str, list[str]]] = {
      scheme: {str(k): list(v or []) for k, v in m.items()}
      for scheme, m in (data.get("mappings") or {}).items()
    }

  # -- lookups -------------------------------------------------------------
  def family_of(self, leaf_id: str) -> str | None:
    leaf = self.leaves.get(leaf_id)
    return leaf.family if leaf else None

  def leaves_in(self, family: str) -> list[Leaf]:
    return [l for l in self.leaves.values() if l.family == family]

  def is_leaf(self, leaf_id: str) -> bool:
    return leaf_id in self.leaves

  def map_from(self, scheme: str, tag: str) -> list[str]:
    """All acceptable v2 leaves for a tag in another scheme (first = default)."""
    return list(self.mappings.get(scheme, {}).get(tag, []))

  # -- prompt rendering ----------------------------------------------------
  def render_for_prompt(self, *, with_rules: bool = True) -> str:
    lines: list[str] = []
    for fam in self.question_order:
      f = self.families[fam]
      lines.append(f"## {fam} — {f['name']}: {f['question']}")
      for leaf in self.leaves_in(fam):
        lines.append(f"- {leaf.id} {leaf.name}: {leaf.definition}")
      lines.append("")
    if with_rules and self.decision_rules:
      lines.append("## Decision rules for confusable leaves")
      for r in self.decision_rules:
        a, b = r["between"]
        lines.append(f"- {a} vs {b}: {r['rule']}")
    return "\n".join(lines).strip()


@lru_cache(maxsize=4)
def load_taxonomy(path: Path | str = DEFAULT_PATH) -> Taxonomy:
  return Taxonomy(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
