from __future__ import annotations

from typing import Optional, Generator, Any
from pathlib import Path
import os


class GrokClient:
    """
    Minimal xAI Grok client with SDK-or-HTTP fallback.

    - Prefers `xai-sdk` if installed; otherwise uses the HTTP OpenAI-compatible endpoint.
    - Streaming is not currently implemented; `stream_complete()` yields the non-stream result once.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "grok-4", timeout: int = 180):
        self.api_key = api_key or self._load_key()
        self.model = model or "grok-4"
        self.timeout = int(timeout)
        self._sdk = None  # lazily created xai-sdk client
        self.last_usage: Optional[dict] = None
        self.last_error: Optional[str] = None

    def _load_key(self) -> str:
        k = os.getenv("XAI_API_KEY")
        if k and k.strip():
            return k.strip()
        # local files for convenience
        for cand in [Path.cwd() / "xai_api_key.txt", Path.home() / ".xai_api_key"]:
            if cand.exists():
                t = cand.read_text(encoding="utf-8").strip()
                if t:
                    return t
        raise RuntimeError("XAI_API_KEY not found. Create xai_api_key.txt or set env var.")

    def _ensure_sdk(self) -> Optional[Any]:
        if self._sdk is not None:
            return self._sdk
        try:
            from xai_sdk import Client as XAIClient  # type: ignore
            self._sdk = XAIClient(api_key=self.api_key)
            return self._sdk
        except Exception:
            self._sdk = None
            return None

    def complete(self, prompt: str, max_tokens: int = 600, temperature: float = 0.6, stream: bool = False) -> str:
        if stream:
            return "".join(self.stream_complete(prompt, max_tokens=max_tokens, temperature=temperature))
        self.last_error = None
        # Try SDK first
        sdk = self._ensure_sdk()
        if sdk is not None:
            try:
                from xai_sdk.chat import user as xai_user  # type: ignore

                chat = sdk.chat.create(model=self.model)
                chat.append(xai_user(prompt))
                resp = chat.sample()
                text = getattr(resp, "content", None)
                if isinstance(text, str) and text.strip():
                    return text.strip()
                return str(resp)
            except Exception as e:
                self.last_error = str(e)
                # fall through to HTTP

        # HTTP fallback: OpenAI-compatible Chat Completions
        try:
            import requests
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            body = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": int(max_tokens),
                "temperature": float(temperature),
                "stream": False,
            }
            resp = requests.post("https://api.x.ai/v1/chat/completions", headers=headers, json=body, timeout=self.timeout)
            if resp.status_code != 200:
                self.last_error = resp.text or ""
                return ""
            data = resp.json()
            try:
                return (data["choices"][0]["message"]["content"] or "").strip()
            except Exception:
                return ""
        except Exception as e:
            self.last_error = str(e)
            return ""

    def stream_complete(self, prompt: str, max_tokens: int = 600, temperature: float = 0.6) -> Generator[str, None, None]:
        # No streaming support yet; yield once using non-stream result
        out = self.complete(prompt, max_tokens=max_tokens, temperature=temperature, stream=False)
        if out:
            yield out

