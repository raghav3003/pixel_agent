"""Judge output model + tolerant parser."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from goldjudge.taxonomy import Taxonomy


class SecondaryLabel(BaseModel):
  leaf: str
  step: int | None = None
  relation: str = "co-occurring"


class Flags(BaseModel):
  evaluator_doubt: bool = False
  gold_divergence_step: int | None = None


class Verdict(BaseModel):
  root_step: int
  primary: str
  secondary: list[SecondaryLabel] = Field(default_factory=list)
  evidence: str = ""
  correction: str = ""
  confidence: str = "mid"
  flags: Flags = Field(default_factory=Flags)
  raw: str = ""
  parse_ok: bool = True
  parse_error: str = ""


_JSON_RE = re.compile(r"\{.*\}", re.S)
_FENCED_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)
_VERDICT_KEYS = ("root_step", "primary")


def _fix_escapes(text: str) -> str:
  """Double backslashes that do not start a valid JSON escape (e.g. regex \\w in evidence)."""
  return re.sub(r"\\(.)", lambda m: m.group(0) if m.group(1) in '"\\/bfnrtu' else "\\\\" + m.group(1), text, flags=re.S)


def _is_verdict(obj: Any) -> bool:
  return isinstance(obj, dict) and any(k in obj for k in _VERDICT_KEYS)


def _extract_json(text: str) -> dict[str, Any] | None:
  text = text.strip()
  # strip ```json fences
  bare = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S | re.M).strip()
  try:
    return json.loads(bare)
  except json.JSONDecodeError:
    pass
  # Prose with quoted code/JSON before the verdict (common after tool turns): take the
  # LAST verdict-shaped object, fenced blocks first, then any decodable {...}.
  found = _last_verdict(text)
  if found is None:
    found = _last_verdict(_fix_escapes(text))
  if found is not None:
    return found
  m = _JSON_RE.search(bare)
  if not m:
    return None
  cand = m.group(0)
  # progressively trim trailing garbage
  for end in range(len(cand), max(len(cand) - 2000, 1), -1):
    try:
      return json.loads(cand[:end])
    except json.JSONDecodeError:
      continue
  return None


def _last_verdict(text: str) -> dict[str, Any] | None:
  for m in reversed(list(_FENCED_RE.finditer(text))):
    try:
      obj = json.loads(m.group(1))
    except json.JSONDecodeError:
      continue
    if _is_verdict(obj):
      return obj
  dec, last = json.JSONDecoder(), None
  for i, ch in enumerate(text):
    if ch == "{":
      try:
        obj, _ = dec.raw_decode(text, i)
      except json.JSONDecodeError:
        continue
      if _is_verdict(obj):
        last = obj
  return last


def parse_verdict(text: str, tax: Taxonomy | None = None, *, n_steps: int | None = None, truncated: bool = False) -> Verdict:
  if truncated:
    return Verdict(root_step=-1, primary="UNPARSED", raw=text, parse_ok=False, parse_error="truncated (max_tokens)")
  data = _extract_json(text)
  if data is None:
    return Verdict(root_step=-1, primary="UNPARSED", raw=text, parse_ok=False, parse_error="no json object found")
  try:
    v = Verdict(**{k: data.get(k) for k in Verdict.model_fields if k in data}, raw=text)
  except ValidationError as exc:
    # salvage the two key fields if possible
    try:
      rs = int(data.get("root_step", -1))
    except (TypeError, ValueError):
      rs = -1
    return Verdict(root_step=rs, primary=str(data.get("primary", "UNPARSED")), raw=text, parse_ok=False, parse_error=str(exc)[:300])
  v.primary = v.primary.strip().upper()
  for s in v.secondary:
    s.leaf = s.leaf.strip().upper()
  if tax is not None and not tax.is_leaf(v.primary):
    v.parse_ok = False
    v.parse_error = f"unknown primary leaf {v.primary!r}"
  if n_steps is not None and not (1 <= v.root_step <= n_steps):
    v.parse_ok = v.parse_ok and False
    v.parse_error = (v.parse_error + "; " if v.parse_error else "") + f"root_step {v.root_step} outside 1..{n_steps}"
  return v
