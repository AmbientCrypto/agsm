from __future__ import annotations

from typing import Optional, Generator, Dict, Any, List
from pathlib import Path
import os


class GoogleClient:
    """
    Minimal Google Gemini client using the REST API.

    - Non-streaming `complete()` is supported.
    - Streaming is not implemented; `stream_complete()` yields the non-stream result once.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash", timeout: int = 180):
        self.api_key = api_key or self._load_key()
        self.model = model or "gemini-2.5-flash"
        self.timeout = int(timeout)
        self.last_usage: Optional[Dict[str, Any]] = None
        self.last_error_text: Optional[str] = None

    def _load_key(self) -> str:
        # Prefer env vars
        for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            v = os.getenv(k)
            if v and v.strip():
                return v.strip()
        # Local file fallbacks
        for cand in [Path.cwd() / "google_api_key.txt", Path.home() / ".google_api_key"]:
            if cand.exists():
                t = cand.read_text(encoding="utf-8").strip()
                if t:
                    return t
        raise RuntimeError("GEMINI_API_KEY/GOOGLE_API_KEY not found. Create google_api_key.txt or set env var.")

    def complete(self, prompt: str, max_tokens: int = 600, temperature: float = 0.6, stream: bool = False) -> str:
        if stream:
            return "".join(self.stream_complete(prompt, max_tokens=max_tokens, temperature=temperature))
        import requests

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        body = {
            "contents": [{"parts": [{"text": str(prompt)}]}],
            "generationConfig": {
                "temperature": float(temperature),
                "maxOutputTokens": int(max_tokens),
            },
        }
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=self.timeout)
        except Exception as e:
            self.last_error_text = str(e)
            raise
        if resp.status_code != 200:
            self.last_error_text = resp.text or ""
            raise RuntimeError(f"Google Gemini API error {resp.status_code}: {self.last_error_text[:200]}")
        data = resp.json()
        self.last_usage = data.get("usageMetadata") if isinstance(data, dict) else None
        chunks: List[str] = []
        if isinstance(data, dict):
            for cand in data.get("candidates", []) or []:
                content = cand.get("content") if isinstance(cand, dict) else None
                if isinstance(content, dict):
                    for part in content.get("parts", []) or []:
                        if isinstance(part, dict) and isinstance(part.get("text"), str):
                            chunks.append(str(part["text"]))
        return "".join(chunks).strip()

    def stream_complete(self, prompt: str, max_tokens: int = 600, temperature: float = 0.6) -> Generator[str, None, None]:
        # No streaming: yield once with non-stream result
        try:
            out = self.complete(prompt, max_tokens=max_tokens, temperature=temperature, stream=False)
            if out:
                yield out
        except Exception:
            return

