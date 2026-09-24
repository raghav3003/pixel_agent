"""Screenshot selection and encoding for multimodal judge prompts."""

from __future__ import annotations

import base64
import io
from functools import lru_cache
from pathlib import Path

from PIL import Image


@lru_cache(maxsize=4096)
def encode_screenshot(path: str, max_width: int = 1280, quality: int = 80) -> str:
  """Return a ``data:image/jpeg;base64,...`` URI, downscaled to ``max_width``."""
  img = Image.open(path).convert("RGB")
  if img.width > max_width:
    h = round(img.height * max_width / img.width)
    img = img.resize((max_width, h), Image.LANCZOS)
  buf = io.BytesIO()
  img.save(buf, format="JPEG", quality=quality)
  return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def select_steps_for_images(n_steps: int, budget: int, *, must: list[int] | None = None) -> list[int]:
  """Choose which 1-based steps get a screenshot under an image budget.

  Always includes the first and last step and any ``must`` steps, then fills the
  budget with evenly spaced steps.
  """
  if n_steps <= 0 or budget <= 0:
    return []
  chosen: set[int] = {1, n_steps}
  for m in must or []:
    if 1 <= m <= n_steps:
      chosen.add(m)
  if n_steps <= budget:
    return list(range(1, n_steps + 1))
  remaining = budget - len(chosen)
  if remaining > 0:
    stride = n_steps / (remaining + 1)
    for i in range(1, remaining + 1):
      chosen.add(max(1, min(n_steps, round(i * stride))))
  return sorted(chosen)[:budget] if len(chosen) > budget else sorted(chosen)


def image_path_ok(path: str | None) -> bool:
  return bool(path) and Path(path).exists()
