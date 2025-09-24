#!/usr/bin/env python3
"""
Together AI Model Probe
=======================

Purpose
-------
Quickly exercise Together models from a providers JSON file to diagnose
common issues (invalid model name, token limit errors, timeouts, streaming
support, etc.). Designed for targeted probes of problematic models.

Usage
-----
python tools/probe_together_models.py \
  --providers-file externalProvidersAndModelsV3-retries.json \
  --timeout 60 \
  --max-tokens 64 512 2048 8192 31000 \
  --prompt "Return the word OK." \
  --stream

Notes
-----
- Does not print API keys. Keys are resolved from each row's api_key, api_key_env,
  or environment variable `TOGETHER_API_KEY`.
- Hits Together's OpenAI-compatible `chat/completions` endpoint directly
  using requests to capture HTTP status codes and error bodies for diagnosis.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


API_URL = "https://api.together.xyz/v1/chat/completions"


@dataclass
class Spec:
    provider: str
    model: str
    reasoning_level: str
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None

    @staticmethod
    def from_obj(o: Dict[str, Any]) -> "Spec":
        return Spec(
            provider=str(o.get("provider", "")).strip(),
            model=str(o.get("model", "")).strip(),
            reasoning_level=str(o.get("reasoning_level", "low")).strip(),
            api_key=(o.get("api_key") or None),
            api_key_env=(o.get("api_key_env") or None),
        )

    def resolve_key(self) -> Optional[str]:
        # Explicit key field
        if self.api_key and str(self.api_key).strip():
            return str(self.api_key).strip()
        # If api_key_env names an env var containing key
        if self.api_key_env and os.getenv(self.api_key_env):
            return os.getenv(self.api_key_env)
        # If api_key_env looks like a literal Together key, treat it as such
        if self.api_key_env and str(self.api_key_env).strip().startswith("tgp_"):
            return str(self.api_key_env).strip()
        # Fallback to global env var
        return os.getenv("TOGETHER_API_KEY")


def diag_from_error(text: str, status: int) -> str:
    s = text.lower()
    if status == 401:
        return "auth_error: invalid or missing API key"
    if status == 404 or "model not found" in s or "invalid model" in s:
        return "model_error: invalid model name or not available"
    if status == 429 or "rate limit" in s or "quota" in s:
        return "rate_limit: backoff or reduce concurrency"
    if "max_tokens" in s and ("too large" in s or "exceeds" in s or "maximum" in s):
        return "tokens_error: reduce max_tokens for this model"
    if "context" in s and ("exceeded" in s or "too long" in s):
        return "context_error: prompt+max_tokens exceed model context"
    if status >= 500:
        return "server_error: provider-side issue"
    if "stream" in s and ("unsupported" in s or "not supported" in s):
        return "streaming_error: model may not support streaming"
    return "unknown_error"


def together_chat_completion(api_key: str, model: str, prompt: str, max_tokens: int, temperature: float, timeout: int, stream: bool) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": bool(stream),
    }
    t0 = time.time()
    if stream:
        resp = requests.post(API_URL, headers=headers, json=body, timeout=timeout, stream=True)
        meta: Dict[str, Any] = {"ok": False, "status": resp.status_code, "elapsed_ms": int((time.time() - t0) * 1000)}
        if resp.status_code != 200:
            txt = resp.text[:500] if hasattr(resp, "text") else ""
            meta.update({
                "ok": False,
                "error": txt,
                "diagnosis": diag_from_error(txt, resp.status_code),
            })
            return meta
        # Try to collect a few tokens and confirm stream terminates with [DONE]
        got = []
        done = False
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                payload = line[len("data: "):].strip()
                if payload == "[DONE]":
                    done = True
                    break
                try:
                    obj = json.loads(payload)
                    delta = obj.get("choices", [{}])[0].get("delta", {}).get("content")
                    if delta:
                        got.append(str(delta))
                except Exception:
                    # Ignore malformed chunks
                    continue
            if len("".join(got)) >= 40:
                # Enough sample content
                pass
        meta.update({
            "ok": bool(got),
            "first_tokens": "".join(got)[:120],
            "completed": done,
            "elapsed_ms": int((time.time() - t0) * 1000),
        })
        return meta
    else:
        resp = requests.post(API_URL, headers=headers, json=body, timeout=timeout)
        elapsed_ms = int((time.time() - t0) * 1000)
        if resp.status_code != 200:
            txt = resp.text[:500] if hasattr(resp, "text") else ""
            return {
                "ok": False,
                "status": resp.status_code,
                "elapsed_ms": elapsed_ms,
                "error": txt,
                "diagnosis": diag_from_error(txt, resp.status_code),
            }
        data = resp.json()
        try:
            text = (data["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            text = ""
        return {
            "ok": bool(text),
            "status": resp.status_code,
            "elapsed_ms": elapsed_ms,
            "text_sample": text[:200],
            "usage": data.get("usage"),
        }


def _extract_final_answer(text: str) -> Optional[float]:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "final_answer" in obj:
            v = obj.get("final_answer")
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str):
                try:
                    return float(v.replace(",", "").strip())
                except Exception:
                    return None
    except Exception:
        pass
    # Try to find final_answer number fragment
    import re
    m = re.search(r"final_answer\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", text)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def _load_results_failures(results_path: Path, dataset_dir: Path, dataset_base: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load failed records from a benchmark results JSON/JSONL and map to questions.

    Returns a list of dicts with keys: id, difficulty, question, actual.
    """
    # Load result records (pretty JSON array or JSONL)
    txt = results_path.read_text(encoding="utf-8")
    recs: List[Dict[str, Any]]
    try:
        obj = json.loads(txt)
        if isinstance(obj, list):
            recs = obj
        else:
            raise ValueError("unexpected JSON shape")
    except Exception:
        # Try JSONL
        recs = []
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                continue

    # Index dataset items per difficulty by id for quick lookup
    def _load_dataset_for_diff(d: int) -> Dict[str, Dict[str, Any]]:
        mapping: Dict[str, Dict[str, Any]] = {}
        for suffix in ("_lite.json", ".json"):
            p = dataset_dir / f"{dataset_base}_diff{d}{suffix}"
            if not p.exists():
                continue
            try:
                arr = json.loads(p.read_text(encoding="utf-8"))
                for it in arr:
                    iid = str(it.get("id"))
                    if iid:
                        mapping[iid] = it
                break
            except Exception:
                continue
        return mapping

    by_diff_cache: Dict[int, Dict[str, Dict[str, Any]]] = {}
    out: List[Dict[str, Any]] = []
    for rec in recs:
        try:
            if rec.get("matched"):
                continue
            qid = str(rec.get("id"))
            diff = int(rec.get("difficulty"))
            actual = rec.get("actual")
            if not qid:
                continue
            if diff not in by_diff_cache:
                by_diff_cache[diff] = _load_dataset_for_diff(diff)
            item = by_diff_cache[diff].get(qid)
            if not item:
                # attempt fallback: scan all diffs lazily
                for d in range(1, 11):
                    if d not in by_diff_cache:
                        by_diff_cache[d] = _load_dataset_for_diff(d)
                    if qid in by_diff_cache[d]:
                        diff = d
                        item = by_diff_cache[d][qid]
                        break
            if not item:
                continue
            qtext = item.get("question") or ""
            out.append({"id": qid, "difficulty": diff, "question": qtext, "actual": actual})
            if limit and len(out) >= limit:
                break
        except Exception:
            continue
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Probe Together models for basic health and parameter support")
    ap.add_argument("--providers-file", default="externalProvidersAndModelsV3-retries.json", help="Path to providers JSON list")
    ap.add_argument("--timeout", type=int, default=60, help="Per-request timeout seconds")
    ap.add_argument("--max-tokens", nargs="*", type=int, default=[64, 512, 2048, 8192, 31000], help="List of max_tokens to probe")
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for probes")
    ap.add_argument("--prompt", default="Return the word OK.", help="Probe prompt to send")
    ap.add_argument("--math-prompt", default="Compute 2+2 and return just 4.", help="Secondary short math prompt")
    ap.add_argument("--stream", action="store_true", help="Also run streaming probes")
    ap.add_argument("--concurrency", type=int, default=1, help="If >1, issues that many parallel non-stream requests for a burst test")
    # Failure replay from a results file (uses dataset to recover prompts)
    ap.add_argument("--from-results", type=str, default=None, help="Path to a benchmark results JSON/JSONL to replay failed questions")
    ap.add_argument("--dataset-dir", type=str, default="algebraTest", help="Directory containing dataset files")
    ap.add_argument("--dataset-base", type=str, default="AGSM8K", help="Dataset base name (e.g., AGSM8K or AGSM8K-V2)")
    ap.add_argument("--max-cases", type=int, default=10, help="Max failed cases to replay from results")
    ap.add_argument("--only-model", type=str, default=None, help="When provided, restrict probes to this exact Together model name")
    args = ap.parse_args(argv)

    pfile = Path(args.providers_file)
    if not pfile.exists():
        print(f"Providers file not found: {pfile}", file=sys.stderr)
        return 2
    try:
        rows = json.loads(pfile.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Failed to read providers file: {e}", file=sys.stderr)
        return 2

    specs = [Spec.from_obj(r) for r in rows if str(r.get("provider", "")).strip().lower() == "together"]
    if args.only_model:
        want = args.only_model.strip()
        specs = [s for s in specs if s.model == want]
    if not specs:
        print("No Together rows found in providers file", file=sys.stderr)
        return 2

    print(f"Loaded {len(specs)} Together model spec(s) from {pfile.name}")
    print("Running probes...\n")

    for sp in specs:
        key = sp.resolve_key()
        print(f"- Model: {sp.model} (reasoning={sp.reasoning_level})")
        if not key:
            print("  ! Missing API key; set TOGETHER_API_KEY or provide in the file")
            print()
            continue

        # If replaying failed questions from a results file, do that first
        if args.from_results:
            rpath = Path(args.from_results)
            fails = _load_results_failures(rpath, Path(args.dataset_dir), args.dataset_base, limit=args.max_cases)
            if not fails:
                print("  replay: no failed cases found or unable to map to dataset questions")
            else:
                print(f"  replay: testing {len(fails)} failed questions from {rpath.name}")
                for i, f in enumerate(fails, 1):
                    qid = f.get("id")
                    diff = f.get("difficulty")
                    qtext = f.get("question") or ""
                    print(f"    [{i}/{len(fails)}] diff={diff} qid={qid}")
                    try:
                        res = together_chat_completion(key, sp.model, qtext, max(1024, max(args.max_tokens or [1024])), 0.0, args.timeout, stream=False)
                    except requests.Timeout:
                        res = {"ok": False, "status": -1, "error": "timeout", "diagnosis": "timeout"}
                    ok = res.get("ok")
                    if ok:
                        txt = res.get("text_sample") or ""
                        proposed = _extract_final_answer(txt)
                        actual = f.get("actual")
                        if proposed is not None and isinstance(actual, (int, float)):
                            diff_abs = abs(float(proposed) - float(actual))
                            denom = abs(float(actual)) or 1.0
                            err_pct = 100.0 * diff_abs / denom
                            print(f"      -> OK ({res.get('elapsed_ms')} ms) proposed={proposed} actual={actual} err%={err_pct:.2f} sample='{txt[:100]}'")
                        else:
                            print(f"      -> OK ({res.get('elapsed_ms')} ms) sample='{txt[:100]}'")
                    else:
                        print(f"      -> FAIL status={res.get('status')} diag={res.get('diagnosis')} err='{(res.get('error') or '')[:160]}'")

        # Non-stream probes across max_tokens sweep
        for mt in args.max_tokens:
            try:
                res = together_chat_completion(key, sp.model, args.prompt, mt, args.temperature, args.timeout, stream=False)
            except requests.Timeout:
                res = {"ok": False, "status": -1, "error": "timeout", "diagnosis": "timeout"}
            status = res.get("status")
            ok = res.get("ok")
            diag = res.get("diagnosis")
            el = res.get("elapsed_ms")
            if ok:
                ts = (res.get("text_sample") or "")
                print(f"  mt={mt:<6} nonstream: OK ({el} ms) sample='{ts[:60]}'")
            else:
                err = (res.get("error") or "")
                print(f"  mt={mt:<6} nonstream: FAIL status={status} diag={diag} err='{err[:120].replace(os.getenv('TOGETHER_API_KEY',''), '***')}'")

        # Short math prompt (sanity)
        try:
            res2 = together_chat_completion(key, sp.model, args.math_prompt, 16, args.temperature, args.timeout, stream=False)
        except requests.Timeout:
            res2 = {"ok": False, "status": -1, "error": "timeout", "diagnosis": "timeout"}
        ok2 = res2.get("ok")
        if ok2:
            print(f"  math_short nonstream: OK sample='{(res2.get('text_sample') or '')[:60]}'")
        else:
            print(f"  math_short nonstream: FAIL status={res2.get('status')} diag={res2.get('diagnosis')} err='{(res2.get('error') or '')[:120]}'")

        # Streaming probe (optional)
        if args.stream:
            try:
                res3 = together_chat_completion(key, sp.model, args.prompt, 64, args.temperature, args.timeout, stream=True)
            except requests.Timeout:
                res3 = {"ok": False, "status": -1, "error": "timeout", "diagnosis": "timeout"}
            ok3 = res3.get("ok")
            if ok3:
                print(f"  stream_short: OK completed={res3.get('completed')} sample='{(res3.get('first_tokens') or '')[:60]}'")
            else:
                print(f"  stream_short: FAIL status={res3.get('status')} diag={res3.get('diagnosis')} err='{(res3.get('error') or '')[:120]}'")

        # Concurrency burst probe (optional)
        if args.concurrency and args.concurrency > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            mt_burst = 256
            n = int(args.concurrency)
            print(f"  burst x{n} nonstream (mt={mt_burst}): starting...")
            with ThreadPoolExecutor(max_workers=n) as ex:
                futs = [
                    ex.submit(
                        together_chat_completion,
                        key,
                        sp.model,
                        args.prompt,
                        mt_burst,
                        args.temperature,
                        args.timeout,
                        False,
                    )
                    for _ in range(n)
                ]
                oks = 0
                fails: List[str] = []
                statuses: List[Any] = []
                for fu in as_completed(futs):
                    try:
                        r = fu.result()
                    except Exception as e:  # request-level exception
                        fails.append(str(e))
                        continue
                    if r.get("ok"):
                        oks += 1
                    else:
                        statuses.append(r.get("status"))
                        fails.append(r.get("diagnosis") or r.get("error") or "fail")
                if oks == n:
                    print(f"  burst: OK all {n}")
                else:
                    # Show top few error kinds
                    from collections import Counter
                    c = Counter([str(x) for x in fails])
                    cs = ", ".join([f"{k}:{v}" for k, v in c.most_common(3)])
                    stc = Counter([str(s) for s in statuses])
                    sts = ", ".join([f"{k}:{v}" for k, v in stc.most_common(3)])
                    print(f"  burst: {oks}/{n} OK; statuses[{sts}] fails[{cs}]")

        print()

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
