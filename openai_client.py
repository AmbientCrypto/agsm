from __future__ import annotations

from typing import Generator
from pathlib import Path
import os


class OpenAIClient:
    """
    Minimal wrapper around OpenAI's Python SDK to mirror the
    Ambient/Together interface: complete() and stream_complete().

    Uses Chat Completions for broad compatibility and simple streaming.
    """

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini", timeout: int = 180):
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError("OpenAI SDK not installed. Add 'openai' to requirements.") from e

        self.api_key = api_key or self._load_key()
        self.model = model or "gpt-4o-mini"
        self.timeout = timeout
        # Instantiate SDK client
        from openai import OpenAI  # type: ignore
        self._sdk = OpenAI(api_key=self.api_key, timeout=timeout)
        self.last_usage = None  # type: ignore
        self.last_error = None  # type: ignore
        self.last_reasoning = None  # type: ignore

    def _load_key(self) -> str:
        # Prefer environment variable
        k = os.getenv("OPENAI_API_KEY")
        if k and k.strip():
            return k.strip()
        # Common local files
        for cand in [Path.cwd() / "openai_api_key.txt", Path.home() / ".openai_api_key"]:
            if cand.exists():
                key = cand.read_text(encoding="utf-8").strip()
                if key:
                    return key
        raise RuntimeError("OPENAI_API_KEY not found. Create openai_api_key.txt or set env var.")

    def _is_responses_model(self) -> bool:
        m = (self.model or "").lower()
        # Use Chat for explicit chat variants; otherwise Responses for o3/o4/gpt-5 families
        if "chat" in m:
            return False
        return any(x in m for x in ("gpt-5", "o4", "o3", "gpt-4.1"))

    def _is_gpt5(self) -> bool:
        m = (self.model or "").lower()
        return "gpt-5" in m

    def complete(
        self,
        prompt: str,
        max_tokens: int = 600,
        temperature: float = 0.6,
        stream: bool = False,
        reasoning_effort: str | None = None,
        text_verbosity: str | None = None,
    ) -> str:
        self.last_error = None
        if stream:
            return "".join(self.stream_complete(prompt, max_tokens=max_tokens, temperature=temperature))
        if self._is_responses_model():
            # Use Responses API (gpt-5/o-series). Prefer max_output_tokens per latest SDKs.
            kwargs = {
                "model": self.model,
                "input": prompt,
                "max_output_tokens": max_tokens,
            }
            if not self._is_gpt5():
                kwargs["temperature"] = temperature
            # For gpt-5, default to high reasoning unless caller overrides
            if self._is_gpt5():
                kwargs["reasoning"] = {"effort": "high"}
            elif reasoning_effort:
                # For other Responses models, include when explicitly requested
                kwargs["reasoning"] = {"effort": reasoning_effort, "summary": "detailed"}
            # Configure text output formatting per latest docs
            tv = (text_verbosity or "low")
            kwargs["text"] = {"format": {"type": "text"}, "verbosity": tv}
            try:
                try:
                    resp = self._sdk.responses.create(**kwargs)
                except Exception as e:
                    # Fallback for older SDKs that expect max_completion_tokens
                    if "max_completion_tokens" in str(e) or "max_output_tokens" in str(e):
                        kwargs2 = dict(kwargs)
                        # ensure only one token field is present for retry
                        kwargs2.pop("max_output_tokens", None)
                        kwargs2["max_completion_tokens"] = max_tokens
                        resp = self._sdk.responses.create(**kwargs2)
                    else:
                        raise
                self.last_usage = getattr(resp, "usage", None)
                # Attempt to extract any reasoning summaries present
                try:
                    self.last_reasoning = self._extract_reasoning_summary(resp)
                except Exception:
                    self.last_reasoning = None
                # Best-effort provider-side error capture
                try:
                    if hasattr(resp, "to_dict"):
                        d = resp.to_dict()
                        if isinstance(d, dict) and d.get("error"):
                            self.last_error = str(d.get("error"))
                except Exception:
                    pass
                text = getattr(resp, "output_text", None)
                if not text:
                    out = []
                    output = getattr(resp, "output", None)
                    if output is None and hasattr(resp, "to_dict"):
                        output = resp.to_dict().get("output")
                    for item in (output or []):
                        try:
                            itype = getattr(item, "type", "") or (item.get("type") if isinstance(item, dict) else "")
                            if itype == "output_text":
                                out.append((getattr(item, "text", None) or (item.get("text") if isinstance(item, dict) else "") or ""))
                            if itype == "message":
                                content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
                                if isinstance(content, list):
                                    for c in content:
                                        ctype = getattr(c, "type", "") or (c.get("type") if isinstance(c, dict) else "")
                                        if ctype in ("output_text", "text"):
                                            out.append((getattr(c, "text", None) or (c.get("text") if isinstance(c, dict) else "") or ""))
                        except Exception:
                            continue
                    text = "".join(out).strip()
                if text:
                    return str(text).strip()
            except Exception as e:
                self.last_error = str(e)
            # Do not fallback to Chat for Responses-only models
            return ""
        # Default: Chat Completions API
        try:
            resp = self._sdk.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
            )
            try:
                self.last_usage = getattr(resp, "usage", None)
                if self.last_usage is None and hasattr(resp, "to_dict"):
                    self.last_usage = resp.to_dict().get("usage")  # type: ignore
            except Exception:
                self.last_usage = None
            # Chat Completions paths do not currently include reasoning summaries
            self.last_reasoning = None
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            self.last_error = str(e)
            return ""

    def stream_complete(self, prompt: str, max_tokens: int = 600, temperature: float = 0.6) -> Generator[str, None, None]:
        self.last_usage = None
        self.last_error = None
        emitted_any = False
        if self._is_responses_model():
            # Responses streaming API
            try:
                # Context manager yields events
                kwargs = {
                    "model": self.model,
                    "input": prompt,
                    # Prefer modern parameter; fall back if unsupported
                    "max_output_tokens": max_tokens,
                }
                if not self._is_gpt5():
                    kwargs["temperature"] = temperature
                if self._is_gpt5():
                    kwargs["reasoning"] = {"effort": "high"}
                try:
                    ctx = self._sdk.responses.stream(**kwargs)
                except Exception as e:
                    if "max_completion_tokens" in str(e) or "max_output_tokens" in str(e):
                        kwargs2 = dict(kwargs)
                        kwargs2.pop("max_output_tokens", None)
                        kwargs2["max_completion_tokens"] = max_tokens
                        ctx = self._sdk.responses.stream(**kwargs2)
                    else:
                        raise
                with ctx as stream:
                    for event in stream:
                        try:
                            et = getattr(event, "type", "")
                            if et == "response.output_text.delta":
                                delta = getattr(event, "delta", "")
                                if delta:
                                    emitted_any = True
                                    yield str(delta)
                            elif et == "response.completed":
                                resp = getattr(event, "response", None)
                                self.last_usage = getattr(resp, "usage", None)
                                try:
                                    self.last_reasoning = self._extract_reasoning_summary(resp)
                                except Exception:
                                    pass
                                # Capture any provider error embedded in the response
                                try:
                                    if hasattr(resp, "to_dict"):
                                        d = resp.to_dict()
                                        if isinstance(d, dict) and d.get("error"):
                                            self.last_error = str(d.get("error"))
                                except Exception:
                                    pass
                        except Exception:
                            continue
            except Exception as e:
                self.last_error = str(e)
                # Fallback to Chat Completions streaming regardless of model
                try:
                    stream = self._sdk.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=True,
                    )
                    for chunk in stream:  # type: ignore
                        try:
                            if hasattr(chunk, "choices") and chunk.choices:
                                delta = getattr(chunk.choices[0], "delta", None)
                                if delta is not None:
                                    content = getattr(delta, "content", None)
                                    if content:
                                        emitted_any = True
                                        yield str(content)
                        except Exception:
                            continue
                except Exception as e2:
                    self.last_error = str(e2)
                    # Final fallback below will try non-streaming
                    pass
            # If we streamed nothing (due to provider/model mismatch), try non-streaming once
        if not emitted_any:
            try:
                text = self.complete(prompt, max_tokens=max_tokens, temperature=temperature, stream=False)
                if text:
                    yield text
            except Exception:
                return
        return
        # Chat Completions streaming
        emitted_any = False
        stream = self._sdk.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:  # type: ignore
            try:
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = getattr(chunk.choices[0], "delta", None)
                    if delta is not None:
                        content = getattr(delta, "content", None)
                        if content:
                            emitted_any = True
                            yield str(content)
            except Exception as e:
                self.last_error = str(e)
                continue
        if not emitted_any:
            try:
                text = self.complete(prompt, max_tokens=max_tokens, temperature=temperature, stream=False)
                if text:
                    yield text
            except Exception:
                return

    # --- Helpers ---
    def _extract_reasoning_summary(self, response_obj) -> str | None:
        """
        Best-effort extraction of reasoning summaries from a Responses API object.
        - Prefers `response_obj.to_dict()` when available.
        - Recurses through nested lists/dicts to collect any items that look like
          reasoning summaries: e.g., nodes with type=='summary_text' and a 'text' field,
          or 'reasoning' objects with a 'summary' that is a list of such nodes.
        Returns a single string joined with blank lines, or None if none found.
        """
        data = None
        # Try direct dict representation
        try:
            if hasattr(response_obj, "to_dict"):
                data = response_obj.to_dict()
        except Exception:
            data = None
        if data is None:
            try:
                # Fallback: shallow attribute scrape
                data = getattr(response_obj, "__dict__", None)
            except Exception:
                data = None
        if not isinstance(data, (dict, list)):
            return None

        summaries: list[str] = []

        def _walk(node):
            if isinstance(node, dict):
                # direct summary list
                if "summary" in node and isinstance(node["summary"], list):
                    for s in node["summary"]:
                        if isinstance(s, dict) and s.get("type") in ("summary_text", "text") and isinstance(s.get("text"), str):
                            summaries.append(str(s["text"]).strip())
                # nested
                for v in node.values():
                    _walk(v)
            elif isinstance(node, list):
                for v in node:
                    _walk(v)

        _walk(data)
        if summaries:
            # Deduplicate while preserving order
            seen = set()
            ordered = []
            for s in summaries:
                if s and s not in seen:
                    seen.add(s)
                    ordered.append(s)
            return "\n\n".join(ordered).strip() if ordered else None
        return None
