#!/usr/bin/env python3
"""
Connectivity check for the Ambient API (manual script, not a pytest test).

Usage:
  AMBIENT_API_KEY=... python tools/check_ambient_connection.py

Optionally, create a key file and point to it via:
  AMBIENT_API_KEY_FILE=/path/to/ambient_api_key.txt python tools/check_ambient_connection.py

This script intentionally avoids the pytest discovery pattern to prevent
accidental network calls during automated test runs.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
import sys

import requests

# Optional dotenv loading; avoid hard dependency if not installed
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - env convenience only
    def load_dotenv():  # type: ignore
        return False

# Ensure project root is importable when running from tools/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_key_from_env_or_file() -> str | None:
    """Load API key from env or common files (non-fatal if missing)."""
    k = os.getenv("AMBIENT_API_KEY")
    if k and k.strip():
        return k.strip()
    f = os.getenv("AMBIENT_API_KEY_FILE")
    if f:
        p = Path(f).expanduser()
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    for cand in [Path.cwd() / "ambient_api_key.txt", Path.home() / ".ambient_api_key"]:
        try:
            if cand.exists():
                key = cand.read_text(encoding="utf-8").strip()
                if key:
                    return key
        except Exception:
            pass
    return None


def quick_check(api_key: str, model: str, base_url: str, timeout: int) -> int:
    """Single POST without retries; prints status and minimal content."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
        "max_tokens": 10,
        "temperature": 0,
        "stream": False,
    }
    start = time.time()
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        elapsed = time.time() - start
        print(f"HTTP {r.status_code} in {elapsed:.1f}s")
        ctype = r.headers.get("content-type", "")
        if r.status_code == 200 and "json" in ctype:
            try:
                data = r.json()
                text = (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
                print(f"OK: {text[:160]}")
                return 0
            except Exception:
                print("OK but failed to parse JSON body")
                return 0
        else:
            body = (r.text or "")
            print((body[:200] + ("…" if len(body) > 200 else "")).strip())
            return 1
    except Exception as e:
        elapsed = time.time() - start
        print(f"✗ Exception after {elapsed:.1f}s: {e}")
        return 2


def main() -> int:
    # Load .env if present
    try:
        load_dotenv()
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Ambient API connectivity check")
    parser.add_argument("--quick", action="store_true", help="Run a single 10s request without retries")
    parser.add_argument("--timeout", type=int, default=10, help="Timeout seconds for quick mode (default: 10)")
    parser.add_argument("--model", default=os.getenv("AMBIENT_MODEL", "deepseek-ai/DeepSeek-R1"))
    parser.add_argument("--base-url", default=os.getenv("AMBIENT_BASE_URL", "https://api.ambient.xyz/v1"))
    args = parser.parse_args()

    api_key = _load_key_from_env_or_file()
    if not api_key:
        print("Error: AMBIENT_API_KEY is not set and no key file found.")
        print("- Export AMBIENT_API_KEY, or set AMBIENT_API_KEY_FILE to a key path.")
        print("- Or create ambient_api_key.txt in the project root (or ~/.ambient_api_key)")
        return 2

    print("Testing Ambient API connection...")
    print(f"API Key: {api_key[:8]}..." if len(api_key) > 8 else "***")

    if args.quick:
        return quick_check(api_key, args.model, args.base_url, args.timeout)

    # Standard (may retry per client settings)
    try:
        # Lazy import to avoid dependency for --quick path
        from ambient_api_client_v3 import AmbientAPIClientV3  # type: ignore
        client = AmbientAPIClientV3(
            api_key=api_key,
            model=args.model,
            stream_output=False,
            timeout=max(10, args.timeout),
        )
        prompt = "What is 2+2? Reply with just the number."
        print(f"Sending test prompt: {prompt}")
        start = time.time()
        response = client.complete(prompt, max_tokens=10, temperature=0, stream=False)
        elapsed = time.time() - start
        print(f"Response received in {elapsed:.1f}s: {response}")
        print("✓ Ambient API is working correctly!")
        return 0
    except Exception as e:
        elapsed = time.time() - start if 'start' in locals() else 0
        print(f"✗ Error after {elapsed:.1f}s: {e}")
        print(f"Error type: {type(e).__name__}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
