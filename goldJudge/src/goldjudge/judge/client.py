"""Judge model clients.

Two backends behind one ``chat(system, content)`` interface:

* ``openai``    — any OpenAI-compatible endpoint (vLLM on Bridges, OpenRouter, …)
* ``anthropic`` — the official Anthropic SDK (used with a cheap model for pipeline
  validation; see ``docs/runbook.md``)

The Anthropic backend also supports ``chat_interactive``: a tool-use loop in which the
judge can request more screenshots on demand (``goldjudge.judge.interactive``).

Bundles are built in OpenAI-style content parts (``text`` / ``image_url`` with a
``data:image/jpeg;base64,…`` URI); the Anthropic backend converts them.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

_DATA_URI_RE = re.compile(r"^data:(image/[a-z0-9.+-]+);base64,(.+)$", re.S)


def load_env_file(path: Path | str | None) -> None:
  """Load KEY=VALUE lines into os.environ (existing vars win). Silently skips a missing file."""
  if not path:
    return
  p = Path(path)
  if not p.exists():
    return
  for line in p.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
      continue
    k, v = line.split("=", 1)
    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _env(name: str, default: str) -> str:
  return os.environ.get(name, default)


@dataclass
class JudgeClientConfig:
  # Defaults resolve from the environment at *instantiation* (after load_env_file), not import.
  provider: str = field(default_factory=lambda: _env("JUDGE_PROVIDER", "openai"))  # openai | anthropic
  base_url: str = field(default_factory=lambda: _env("JUDGE_BASE_URL", "http://localhost:8000/v1"))
  api_key: str = field(default_factory=lambda: _env("JUDGE_API_KEY", "EMPTY"))
  model: str = field(default_factory=lambda: _env("JUDGE_MODEL", "judge"))
  temperature: float = 0.0
  max_tokens: int = 1500
  timeout_s: float = 600.0
  retries: int = 6
  # Anthropic only. "default" omits the param (model default: off on Haiku 4.5, adaptive on
  # Sonnet 5); "adaptive" / "disabled" set it explicitly. Thinking tokens count toward max_tokens.
  thinking: str = "default"
  effort: str | None = None  # output_config.effort; not supported on Haiku 4.5
  # OpenAI reasoning models (e.g. gpt-6-luna) only. When set, requests send reasoning_effort and
  # max_completion_tokens and omit temperature; None keeps the plain vLLM-style request.
  # gpt-6-luna accepts function calling on Chat Completions only with "none".
  reasoning_effort: str | None = None
  extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeResponse:
  text: str
  usage: dict[str, Any]
  latency_s: float
  model: str
  n_turns: int = 1


class _OpenAIBackend:
  def __init__(self, cfg: JudgeClientConfig):
    from openai import OpenAI  # lazy

    self.cfg = cfg
    self._client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=cfg.timeout_s, max_retries=0)

  def _kwargs(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
    if self.cfg.reasoning_effort:  # OpenAI reasoning model: no temperature, completion-token cap
      return dict(model=self.cfg.model, reasoning_effort=self.cfg.reasoning_effort,
                  max_completion_tokens=self.cfg.max_tokens, messages=messages)
    return dict(model=self.cfg.model, temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens, messages=messages)

  def _use_responses(self) -> bool:
    # gpt-6-luna rejects function tools with reasoning on /v1/chat/completions, so every reasoning-on
    # request (both harnesses) goes through /v1/responses: fixed and on-demand share one API surface.
    return bool(self.cfg.reasoning_effort) and self.cfg.reasoning_effort != "none"

  def _responses_kwargs(self, system: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    # Stateless (store=False): reasoning items come back encrypted and are passed back verbatim so
    # the model keeps its reasoning across tool turns.
    return dict(model=self.cfg.model, instructions=system, input=items, reasoning={"effort": self.cfg.reasoning_effort},
                max_output_tokens=self.cfg.max_tokens, store=False, include=["reasoning.encrypted_content"])

  def chat(self, system: str, user_content: list[dict[str, Any]]) -> JudgeResponse:
    t0 = time.time()
    if self._use_responses():
      resp = self._client.responses.create(
        **self._responses_kwargs(system, [{"role": "user", "content": openai_parts_to_responses(user_content)}]))
      return JudgeResponse(text=resp.output_text or "", usage=self._usage_responses([resp]), latency_s=time.time() - t0,
                           model=resp.model or self.cfg.model)
    resp = self._client.chat.completions.create(
      **self._kwargs([{"role": "system", "content": system}, {"role": "user", "content": user_content}]))
    text = resp.choices[0].message.content or ""
    return JudgeResponse(text=text, usage=self._usage([resp]), latency_s=time.time() - t0, model=resp.model or self.cfg.model)

  @staticmethod
  def _usage_responses(resps: list[Any]) -> dict[str, Any]:
    tot = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "reasoning_tokens": 0}
    for r in resps:
      if r.usage:
        tot["prompt_tokens"] += r.usage.input_tokens or 0
        tot["completion_tokens"] += r.usage.output_tokens or 0  # includes reasoning tokens
        itd, otd = getattr(r.usage, "input_tokens_details", None), getattr(r.usage, "output_tokens_details", None)
        tot["cached_tokens"] += (getattr(itd, "cached_tokens", 0) or 0) if itd else 0
        tot["reasoning_tokens"] += (getattr(otd, "reasoning_tokens", 0) or 0) if otd else 0
    last = resps[-1]
    inc = getattr(last, "incomplete_details", None)
    truncated = last.status == "incomplete" and getattr(inc, "reason", None) == "max_output_tokens"
    tot["stop_reason"] = "max_tokens" if truncated else ("stop" if last.status == "completed" else last.status)
    return tot

  @staticmethod
  def _tool_results(run_tool: Callable[[str, dict[str, Any]], tuple[list[dict[str, Any]], bool]],
                    calls: list[tuple[str, str, str]]) -> tuple[list[tuple[str, str]], list[dict[str, Any]]]:
    """Run ``(call_id, name, arguments_json)`` tool calls.

    Returns the text result per call id and the screenshots (each with its caption, as OpenAI
    chat parts) for one follow-up user message — tool results carry text only.
    """
    texts: list[tuple[str, str]] = []
    follow: list[dict[str, Any]] = []
    for cid, name, arguments in calls:
      try:
        blocks, _ = run_tool(name, json.loads(arguments or "{}"))
      except json.JSONDecodeError:
        blocks = [{"type": "text", "text": "arguments must be a JSON object"}]
      parts = anthropic_blocks_to_openai(blocks)
      notes: list[str] = []
      for i, p in enumerate(parts):
        nxt_is_image = i + 1 < len(parts) and parts[i + 1]["type"] == "image_url"
        if p["type"] == "image_url" or nxt_is_image:
          follow.append(p)
        else:
          notes.append(p["text"])
      n_img = sum(p["type"] == "image_url" for p in parts)
      shown = f"{n_img} screenshot(s) follow in the next message. " if n_img else ""
      texts.append((cid, shown + " ".join(notes)))
    return texts, follow

  @staticmethod
  def _usage(resps: list[Any]) -> dict[str, Any]:
    tot = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "reasoning_tokens": 0}
    for r in resps:
      if r.usage:
        tot["prompt_tokens"] += r.usage.prompt_tokens or 0
        tot["completion_tokens"] += r.usage.completion_tokens or 0
        ptd, ctd = getattr(r.usage, "prompt_tokens_details", None), getattr(r.usage, "completion_tokens_details", None)
        tot["cached_tokens"] += (getattr(ptd, "cached_tokens", 0) or 0) if ptd else 0
        tot["reasoning_tokens"] += (getattr(ctd, "reasoning_tokens", 0) or 0) if ctd else 0
    finish = resps[-1].choices[0].finish_reason
    tot["stop_reason"] = "max_tokens" if finish == "length" else finish  # same vocabulary as the Anthropic backend
    return tot

  def chat_interactive(
    self,
    system: str,
    user_content: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    run_tool: Callable[[str, dict[str, Any]], tuple[list[dict[str, Any]], bool]],
    *,
    max_turns: int,
    retry: Callable[[Callable[[], Any]], Any],
    keep_last_images: int | None = None,
  ) -> JudgeResponse:
    """Tool-use loop over an OpenAI-compatible endpoint (vLLM needs ``--enable-auto-tool-choice``).

    Tool messages carry text only, so the tool's text goes in the ``tool`` message and its
    screenshots follow in one ``user`` message. On the last allowed turn tools are not offered,
    so the model must answer. Reasoning-on requests use the Responses API (see ``_use_responses``).
    """
    if self._use_responses():
      return self._responses_interactive(system, user_content, tools, run_tool, max_turns=max_turns, retry=retry)
    t0 = time.time()
    oa_tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
                for t in tools]
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}, {"role": "user", "content": user_content}]
    resps: list[Any] = []
    for turn in range(max_turns):
      if keep_last_images is not None:  # as CUADebugger: drop older screenshots, keep the written observations
        _drop_old_images(messages, keep_last_images)
      kwargs = self._kwargs(list(messages))
      if turn < max_turns - 1:
        kwargs["tools"] = oa_tools
        kwargs["tool_choice"] = "auto"
      resp = retry(lambda: self._client.chat.completions.create(**kwargs))
      resps.append(resp)
      msg = resp.choices[0].message
      calls = msg.tool_calls or []
      if not calls:
        break
      messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
        {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls]})
      texts, follow = self._tool_results(run_tool, [(c.id, c.function.name, c.function.arguments) for c in calls])
      messages += [{"role": "tool", "tool_call_id": cid, "content": text} for cid, text in texts]
      if turn + 1 == max_turns - 1:
        follow.append({"type": "text", "text": _LAST_TURN_NOTE})
      if follow:
        messages.append({"role": "user", "content": follow})
    text = resps[-1].choices[0].message.content or ""
    return JudgeResponse(text=text, usage=self._usage(resps), latency_s=time.time() - t0,
                         model=resps[-1].model or self.cfg.model, n_turns=len(resps))

  def _responses_interactive(
    self,
    system: str,
    user_content: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    run_tool: Callable[[str, dict[str, Any]], tuple[list[dict[str, Any]], bool]],
    *,
    max_turns: int,
    retry: Callable[[Callable[[], Any]], Any],
  ) -> JudgeResponse:
    """The same tool loop over the Responses API (function tools with reasoning on).

    Each turn's output items (encrypted reasoning + function calls) are passed back verbatim.
    On the last allowed turn the tools stay declared (earlier calls reference them) but
    ``tool_choice`` is ``none``, so the model must answer.
    """
    t0 = time.time()
    rs_tools = [{"type": "function", "name": t["name"], "description": t["description"], "parameters": t["input_schema"]} for t in tools]
    items: list[dict[str, Any]] = [{"role": "user", "content": openai_parts_to_responses(user_content)}]
    resps: list[Any] = []
    for turn in range(max_turns):
      kwargs = self._responses_kwargs(system, list(items))
      kwargs["tools"] = rs_tools
      kwargs["tool_choice"] = "auto" if turn < max_turns - 1 else "none"
      resp = retry(lambda: self._client.responses.create(**kwargs))
      resps.append(resp)
      calls = [o for o in resp.output if o.type == "function_call"]
      if not calls:
        break
      items += [o.model_dump(exclude_none=True) for o in resp.output]
      texts, follow = self._tool_results(run_tool, [(c.call_id, c.name, c.arguments) for c in calls])
      items += [{"type": "function_call_output", "call_id": cid, "output": text} for cid, text in texts]
      if turn + 1 == max_turns - 1:
        follow.append({"type": "text", "text": _LAST_TURN_NOTE})
      if follow:
        items.append({"role": "user", "content": openai_parts_to_responses(follow)})
    return JudgeResponse(text=resps[-1].output_text or "", usage=self._usage_responses(resps), latency_s=time.time() - t0,
                         model=resps[-1].model or self.cfg.model, n_turns=len(resps))


_LAST_TURN_NOTE = "No more tool calls are available. Respond now with ONLY the JSON verdict."


def _drop_old_images(messages: list[dict[str, Any]], keep: int) -> None:
  """Replace all but the last ``keep`` images in follow-up user messages (not the first user message) by a note."""
  seen = 0
  for m in reversed(messages[2:]):
    if m.get("role") != "user" or not isinstance(m.get("content"), list):
      continue
    for i in range(len(m["content"]) - 1, -1, -1):
      if m["content"][i].get("type") == "image_url":
        seen += 1
        if seen > keep:
          m["content"][i] = {"type": "text", "text": "(older screenshot removed from context)"}


def openai_parts_to_responses(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Convert OpenAI chat content parts to Responses API input parts."""
  out: list[dict[str, Any]] = []
  for p in parts:
    if p["type"] == "text":
      out.append({"type": "input_text", "text": p["text"]})
    elif p["type"] == "image_url":
      img = {"type": "input_image", "image_url": p["image_url"]["url"]}
      if p["image_url"].get("detail"):
        img["detail"] = p["image_url"]["detail"]
      out.append(img)
  return out


def anthropic_blocks_to_openai(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Convert the screenshot tool's Anthropic-style blocks to OpenAI content parts."""
  out: list[dict[str, Any]] = []
  for b in blocks:
    if b["type"] == "text":
      out.append({"type": "text", "text": b["text"]})
    elif b["type"] == "image":
      src = b["source"]
      url = f"data:{src['media_type']};base64,{src['data']}" if src["type"] == "base64" else src["url"]
      out.append({"type": "image_url", "image_url": {"url": url}})
  return out


def openai_parts_to_anthropic(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Convert OpenAI-style content parts to Anthropic content blocks."""
  out: list[dict[str, Any]] = []
  for p in parts:
    if p.get("type") == "text":
      out.append({"type": "text", "text": p["text"]})
    elif p.get("type") == "image_url":
      url = p["image_url"]["url"]
      m = _DATA_URI_RE.match(url)
      if m:
        out.append({"type": "image", "source": {"type": "base64", "media_type": m.group(1), "data": m.group(2)}})
      else:
        out.append({"type": "image", "source": {"type": "url", "url": url}})
    else:
      raise ValueError(f"unsupported content part type: {p.get('type')}")
  return out


class _AnthropicBackend:
  def __init__(self, cfg: JudgeClientConfig):
    import anthropic  # lazy

    self.cfg = cfg
    self._anthropic = anthropic
    # Credentials resolve from ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / `ant auth login` profile.
    self._client = anthropic.Anthropic(timeout=cfg.timeout_s, max_retries=0)

  def _kwargs(self, system: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
    kwargs: dict[str, Any] = dict(
      model=self.cfg.model,
      max_tokens=self.cfg.max_tokens,
      system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
      messages=messages,
    )
    # Sampling params (temperature/top_p) are not part of the 1.x SDK Messages surface; omit.
    thinking = self.cfg.extra.get("thinking")
    if thinking is None and self.cfg.thinking != "default":
      thinking = {"type": self.cfg.thinking}
    if thinking:
      kwargs["thinking"] = thinking
    if self.cfg.effort:
      kwargs["output_config"] = {"effort": self.cfg.effort}
    return kwargs

  @staticmethod
  def _check(resp: Any) -> None:
    if resp.stop_reason == "refusal":
      details = getattr(resp, "stop_details", None)
      raise RuntimeError(f"judge request refused: {details}")

  @staticmethod
  def _usage(resps: list[Any]) -> dict[str, Any]:
    tot = {"input_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "completion_tokens": 0}
    for r in resps:
      u = r.usage
      tot["input_tokens"] += u.input_tokens or 0
      tot["cache_read_input_tokens"] += u.cache_read_input_tokens or 0
      tot["cache_creation_input_tokens"] += u.cache_creation_input_tokens or 0
      tot["completion_tokens"] += u.output_tokens or 0
    tot["prompt_tokens"] = tot["input_tokens"] + tot["cache_read_input_tokens"] + tot["cache_creation_input_tokens"]
    tot["stop_reason"] = resps[-1].stop_reason
    return tot

  def chat(self, system: str, user_content: list[dict[str, Any]]) -> JudgeResponse:
    t0 = time.time()
    resp = self._client.messages.create(**self._kwargs(system, [{"role": "user", "content": openai_parts_to_anthropic(user_content)}]))
    self._check(resp)
    text = "".join(b.text for b in resp.content if b.type == "text")
    return JudgeResponse(text=text, usage=self._usage([resp]), latency_s=time.time() - t0, model=resp.model)

  def chat_interactive(
    self,
    system: str,
    user_content: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    run_tool: Callable[[str, dict[str, Any]], tuple[list[dict[str, Any]], bool]],
    *,
    max_turns: int,
    retry: Callable[[Callable[[], Any]], Any],
  ) -> JudgeResponse:
    """Tool-use loop. ``run_tool(name, input) -> (content_blocks, is_error)``.

    On the last allowed turn tools are switched off (``tool_choice: none``) so the model
    must answer. Each API call is retried independently via ``retry``.
    """
    t0 = time.time()
    messages: list[dict[str, Any]] = [{"role": "user", "content": openai_parts_to_anthropic(user_content)}]
    resps: list[Any] = []
    for turn in range(max_turns):
      kwargs = self._kwargs(system, messages)
      kwargs["tools"] = tools
      kwargs["cache_control"] = {"type": "ephemeral"}  # cache the growing conversation prefix
      if turn == max_turns - 1:
        kwargs["tool_choice"] = {"type": "none"}
      resp = retry(lambda: self._client.messages.create(**kwargs))
      self._check(resp)
      resps.append(resp)
      if resp.stop_reason != "tool_use":
        break
      messages.append({"role": "assistant", "content": resp.content})
      results: list[dict[str, Any]] = []
      for b in resp.content:
        if b.type == "tool_use":
          content, is_error = run_tool(b.name, dict(b.input))
          results.append({"type": "tool_result", "tool_use_id": b.id, "content": content, "is_error": is_error})
      if turn + 1 == max_turns - 1:
        results.append({"type": "text", "text": "No more tool calls are available. Respond now with ONLY the JSON verdict."})
      messages.append({"role": "user", "content": results})
    text = "".join(b.text for b in resps[-1].content if b.type == "text")
    return JudgeResponse(text=text, usage=self._usage(resps), latency_s=time.time() - t0, model=resps[-1].model, n_turns=len(resps))


class JudgeClient:
  """Provider-agnostic judge client with simple retry."""

  def __init__(self, cfg: JudgeClientConfig):
    self.cfg = cfg
    if cfg.provider == "anthropic":
      self._backend: Any = _AnthropicBackend(cfg)
    elif cfg.provider == "openai":
      self._backend = _OpenAIBackend(cfg)
    else:
      raise ValueError(f"unknown provider {cfg.provider!r}")

  def chat(self, system: str, user_content: list[dict[str, Any]]) -> JudgeResponse:
    return self._retry(lambda: self._backend.chat(system, user_content))

  def chat_interactive(self, system: str, user_content: list[dict[str, Any]], tools: list[dict[str, Any]],
                       run_tool: Callable[[str, dict[str, Any]], tuple[list[dict[str, Any]], bool]], *, max_turns: int,
                       **kw: Any) -> JudgeResponse:
    if not hasattr(self._backend, "chat_interactive"):
      raise NotImplementedError(f"on-demand screenshots are not implemented for provider {self.cfg.provider!r}")
    return self._backend.chat_interactive(system, user_content, tools, run_tool, max_turns=max_turns, retry=self._retry, **kw)

  def _retry(self, fn: Callable[[], Any]) -> Any:
    last: Exception | None = None
    for attempt in range(self.cfg.retries):
      try:
        return fn()
      except (TypeError, ValueError, KeyError):
        raise  # programming errors: never retry
      except Exception as exc:  # noqa: BLE001 — retry transport/rate-limit errors
        last = exc
        # do not retry hard client errors
        status = getattr(exc, "status_code", None)
        if status is not None and 400 <= int(status) < 500 and int(status) not in (408, 409, 429):
          raise
        delay = 5.0 * (2 ** attempt)  # 5, 10, 20, 40, 80 s
        if status == 429:
          headers = getattr(getattr(exc, "response", None), "headers", None) or {}
          try:
            delay = max(delay, float(headers.get("retry-after", 0)))
          except (TypeError, ValueError):
            pass
        time.sleep(min(delay, 120.0))
    chain: list[str] = []
    cause: BaseException | None = last
    while cause is not None and len(chain) < 5:
      chain.append(f"{type(cause).__name__}: {cause}")
      cause = cause.__cause__ or cause.__context__
    raise RuntimeError(f"judge request failed after {self.cfg.retries} attempts: " + " <- ".join(chain))
