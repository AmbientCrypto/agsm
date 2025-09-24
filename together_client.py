from __future__ import annotations

from typing import Generator, Dict, Any
from pathlib import Path
import os


class TogetherClient:
    """
    Minimal wrapper around Together's Python SDK to provide a similar
    interface to AmbientAPIClientV3: complete() and stream_complete().
    """

    def __init__(self, api_key: str | None = None, model: str = "deepseek-ai/DeepSeek-R1", timeout: int = 180):
        try:
            from together import Together  # type: ignore
        except Exception as e:
            raise RuntimeError("Together SDK not installed. Add 'together' to requirements.") from e

        self.api_key = api_key or self._load_key()
        self.model = self._normalize_model(model)
        self.timeout = timeout
        self._sdk = Together(api_key=self.api_key)
        self.last_usage = None  # type: ignore

    def _normalize_model(self, model: str | None) -> str:
        m = (model or "").strip()
        if not m:
            # Allow env override for test/control usage
            env_model = os.getenv("TOGETHER_MODEL") or os.getenv("TOGETHER_TEST_MODEL")
            if env_model and env_model.strip():
                return env_model.strip()
            return "deepseek-ai/DeepSeek-R1"
        # Fix common shorthand or truncated vendor forms
        if m.lower() in ("deepseek-ai", "deepseek-ai/"):
            return "deepseek-ai/DeepSeek-R1"
        return m

    def _load_key(self) -> str:
        # Try env first
        k = os.getenv("TOGETHER_API_KEY")
        if k and k.strip():
            return k.strip()
        # Try file in repo root
        for cand in [Path.cwd() / "togetherai_api_key.txt", Path.home() / ".together_api_key"]:
            if cand.exists():
                key = cand.read_text(encoding="utf-8").strip()
                if key:
                    return key
        raise RuntimeError("TOGETHER_API_KEY not found. Create togetherai_api_key.txt or set env var.")

    def complete(self, prompt: str, max_tokens: int = 600, temperature: float = 0.6, stream: bool = False) -> str:
        if stream:
            return "".join(self.stream_complete(prompt, max_tokens=max_tokens, temperature=temperature))
        resp = self._sdk.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )
        # The Together SDK returns an object with choices
        try:
            try:
                # Attempt to capture usage if provided by SDK
                self.last_usage = getattr(resp, "usage", None)
                if self.last_usage is None:
                    data = getattr(resp, "__dict__", {}) or {}
                    self.last_usage = data.get("usage")
            except Exception:
                self.last_usage = None
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            # Fall back to dict-like behavior if needed
            data: Dict[str, Any] = getattr(resp, "__dict__", {}) or {}  # best-effort
            choices = data.get("choices") or []
            if choices:
                message = choices[0].get("message", {})
                return (message.get("content") or "").strip()
            return ""

    def stream_complete(self, prompt: str, max_tokens: int = 600, temperature: float = 0.6) -> Generator[str, None, None]:
        self.last_usage = None
        provider_max = max(64, int(max_tokens or 0))  # ensure enough budget to reach final answer with thinky models
        resp = self._sdk.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=provider_max,
            temperature=temperature,
            stream=True,
        )
        for token in resp:  # type: ignore
            try:
                # SDK: token.choices[0].delta.content
                if hasattr(token, "choices") and token.choices:
                    delta = getattr(token.choices[0], "delta", None)
                    if delta is not None:
                        content = getattr(delta, "content", None)
                        if content:
                            yield str(content)
            except Exception:
                continue
