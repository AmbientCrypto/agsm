#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
from typing import Optional


def probe_with_wrapper(model: str, effort: Optional[str]) -> None:
    import sys
    sys.path.insert(0, ".")
    try:
        from openai_client import OpenAIClient
        client = OpenAIClient(model=model, timeout=60)
    except Exception as e:
        print(f"[wrapper] cannot init client: {type(e).__name__}: {e}")
        return
    prompt = "Answer with one word only. Do not include any reasoning or extra text. What is the capital of Nebraska?"
    t0 = time.time()
    try:
        text = client.complete(prompt, max_tokens=256, temperature=0.0, reasoning_effort=effort)
        dt = time.time() - t0
        print(f"[wrapper] effort={effort} secs={dt:.2f} len={len(text or '')} text={repr((text or '')[:200])}")
        print(f"[wrapper] usage={getattr(client, 'last_usage', None)} error={getattr(client, 'last_error', None)}")
    except Exception as e:
        dt = time.time() - t0
        print(f"[wrapper] effort={effort} EXC after {dt:.2f}s: {type(e).__name__}: {e}")


def _load_openai_key() -> Optional[str]:
    k = os.getenv("OPENAI_API_KEY")
    if k and k.strip():
        return k.strip()
    from pathlib import Path
    candidates = [
        Path.cwd() / "openai_api_key.txt",
        Path(__file__).resolve().parent.parent / "openai_api_key.txt",
        Path.home() / ".openai_api_key",
    ]
    for p in candidates:
        try:
            if p.exists():
                txt = p.read_text(encoding='utf-8').strip()
                if txt:
                    return txt
        except Exception:
            continue
    return None


def probe_with_sdk(model: str, effort: Optional[str]) -> None:
    try:
        from openai import OpenAI
    except Exception as e:
        print(f"[sdk] OpenAI SDK missing: {e}")
        return
    key = _load_openai_key()
    if not key:
        print("[sdk] No API key found in env, repo root, or ~/.openai_api_key")
        return
    client = OpenAI(api_key=key)
    prompt = "Answer with one word only. Do not include any reasoning or extra text. What is the capital of Nebraska?"
    kwargs = {
        "model": model,
        "input": prompt,
        "max_output_tokens": 256,
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "tools": [],
        "store": False,
    }
    if effort:
        kwargs["reasoning"] = {"effort": effort, "summary": "auto"}
    t0 = time.time()
    try:
        resp = client.responses.create(**kwargs)
        dt = time.time() - t0
        # Try convenient fields first
        text = getattr(resp, "output_text", None)
        if not text:
            # Fallback to output list traversal
            text = ""
            output = getattr(resp, "output", None)
            if output is None and hasattr(resp, "to_dict"):
                output = resp.to_dict().get("output")
            for item in (output or []):
                itype = getattr(item, "type", "") or (item.get("type") if isinstance(item, dict) else "")
                if itype == "message":
                    content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
                    if isinstance(content, list):
                        for c in content:
                            ctype = getattr(c, "type", "") or (c.get("type") if isinstance(c, dict) else "")
                            if ctype in ("output_text", "text"):
                                text += (getattr(c, "text", None) or (c.get("text") if isinstance(c, dict) else "") or "")
        print(f"[sdk] effort={effort} secs={dt:.2f} len={len(text or '')} text={repr((text or '')[:200])}")
        print(f"[sdk] usage={getattr(resp, 'usage', None)}")
    except Exception as e:
        dt = time.time() - t0
        print(f"[sdk] effort={effort} EXC after {dt:.2f}s: {type(e).__name__}: {e}")


def main() -> int:
    model = os.getenv("OPENAI_TEST_MODEL", "gpt-5")
    for effort in ("low", "medium", "high"):
        probe_with_wrapper(model, effort)
    print("-- SDK direct --")
    for effort in ("low", "medium", "high"):
        probe_with_sdk(model, effort)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
