"""
Algebra Dataset Benchmarking Tool (AGSM harness)
================================================

Overview
--------
This CLI benchmarks one or more LLM providers/models against AGSM algebra word
problem datasets organized by difficulty tiers. For each question, it asks the
model to return a JSON object containing a numeric `final_answer`, then grades
the result against ground truth with a configurable percent‑error threshold.
Optionally, it plots a per‑model line chart of pass rates across difficulty.

Supported providers out of the box: Ambient, Together, OpenAI, Anthropic,
and xAI Grok. The tool reuses the existing thin clients in this repo where
available and falls back to HTTP requests or SDKs when appropriate.

Dataset Inputs
--------------
- Location: `algebraTest/`
- Expected files: `<BASE>_diff1_lite.json` through `<BASE>_diff10_lite.json`.
  By default `<BASE>` is `AGSM8K`, so files are
  `AGSM8K_diff{1..10}_lite.json`.
- Shape per item (example):
  {
    "id": "...",
    "question": "...",
    "deterministic_answer": -18.0
  }

Providers / Models Configuration
--------------------------------
- File: `defaultProvidersAndModels.json` (or pass a custom path).
- Format: list of entries with fields:
  - provider: "ambient" | "together" | "openai" | "anthropic" | "grok"
  - model: provider-specific model name (e.g., "gpt-4o-mini", "grok-4")
  - reasoning_level: "low" | "medium" | "high"
  - api_key (optional): literal API key for the row
  - api_key_env (optional): either the name of an env var that contains
    the key, or the literal key itself (useful for local, gitignored files)

Reasoning Level Mapping
-----------------------
- OpenAI (Responses API models such as `gpt-5-...`, `o3`, `o4` families):
  - Sets `reasoning = {"effort": "low"|"medium"|"high"}` when applicable.
  - Temperature still applied where supported by the model.
- Others (Ambient, Together, Anthropic, Grok):
  - Temperature scales with level; token budget is set high globally:
    - low: temperature=0.0, max_tokens=50000
    - medium: temperature=0.3, max_tokens=50000
    - high: temperature=0.7, max_tokens=50000

Tolerant Parsing and Grading
----------------------------
The tool attempts to parse a numeric `final_answer` from model output.
It prefers strict JSON; it also detects fenced JSON code blocks (```json ... ```)
before falling back to JSON fragments, a `final_answer` regex, or a last‑number
heuristic. Each record logs:
- `parsing_mode`: one of `json`, `json_fenced`, `json_fragment`,
  `final_answer_regex`, `last_number`, or `none`.
- `tolerant_parsing_used`: true when the mode is not `json`.
Each record also includes convenience fields for debugging:
- `final_answer`: the parsed numeric value (same as `proposed`).
- `response_text`: the think‑stripped response text used for parsing.
- `duration_seconds`: latency to produce the response.
- `started_at`, `completed_at`: ISO UTC timestamps for request start/end.
- `error_reason`: exception info when a request fails (else null).

Outputs
-------
All run artifacts are saved under:
  `algebraTest/benchmark_runs/<RUN_NAME>/`

- Per-model records: `results_<provider>_<model>_<reasoning>.jsonl`
  - Contains the raw response, extracted thinking (if available), actual and
    proposed values, difference, error percent, and match status.
- Per-model pretty results: `results_<provider>_<model>_<reasoning>.json`
  - Same content as the JSONL but as a single pretty-printed JSON array for readability.
- Per-model summary: `summary_<provider>_<model>_<reasoning>.json`
  - Aggregated pass rates by difficulty.
- Global summary: `summary_all.json`
  - Includes `"__provider_latency__"` with aggregate latency stats per provider
    (count, avg, p50, p95, min, max).
- Run metadata: `run_meta.json`
  - Captures the dataset base (`base_name`), sampling settings, and timestamps.
  - Used by finalize to restrict dataset lookups to the relevant base only.
- Graphs (requires `matplotlib`):
  - All models in one: `graph_<RUN_NAME>.png` (legend placed outside to avoid overlap).
  - Per-provider graphs: `graph_<RUN_NAME>_provider_<provider>.png`.
  - Per-provider per-model graphs (series per reasoning level):
    `graph_<RUN_NAME>_provider_<provider>_model_<model>.png`.
  - Providers overall bar: `graph_<RUN_NAME>_providers_overall.png` — overall % correct
    aggregated across all difficulties and models per provider.
  - Provider/Model overall bar: `graph_<RUN_NAME>_providers_models_overall.png` — overall %
    correct aggregated across all difficulties and reasoning levels for each
    provider/model pair (one bar per provider:model).
  - Provider models bar: `graph_<RUN_NAME>_provider_<provider>_models_overall.png` — overall %
    correct across all difficulties for each model (reasoning level) within a provider.
  - Missed-by-all line: `graph_<RUN_NAME>_missed_by_all.png` — percent of items per difficulty
    that no model answered correctly (requires comparable sampling and `missed_by_all.json`).
  - Provider/Model/Reasoning bars: `graph_<RUN_NAME>_providers_models_reasoning_overall.png` — overall %
    correct with a separate bar for each provider:model:reasoning triple.
  - Grouped per-difficulty bars (defaults d1 and d10):
    `graph_<RUN_NAME>_providers_models_levels_grouped_d1_d10.png` — grouped by model with distinct hatches per difficulty.
  - Zero-incorrect line: `graph_<RUN_NAME>_zero_incorrect_by_difficulty.png` — percent of problems per difficulty unanswered by any model.
 - Missed-by-all logs (optional; build via finalize):
   - Only generated when explicitly requested during finalize (see below) and when sampling is comparable (default or `--random --seed`).
   - `missed_by_all.json` — array of objects with: `id`, `difficulty`, `question`, `actual`, and `answers_by_model` mapping `provider:model:reasoning` → proposed value.
   - `missed_by_all_detailed.json` — same as above plus (when available): `equation_template`, `eq_system_str`, and `solution_eval`.
   - Consensus variants: `missed_by_all_consensus.json` and `_detailed.json` are produced only when consensus grading is requested.
   - Construction is memory-safe: finalize streams dataset files and prefers `*_lite.json` for lookups; it also restricts lookups to the run’s `base_name`.

Graph customization & consensus
-------------------------------
- Provider renames for labels/titles: pass a map like `{ "ambient": "zai" }` via `finalize_run_outputs(..., provider_renames=...)` or CLI `--graph-provider-rename ambient:zai` (repeatable).
- Omit run name in titles: `omit_run_name_in_titles=True` or CLI `--graph-omit-run-name-in-titles`.
- Show grading type in titles (Ground truth vs Consensus): `show_grading_type_in_titles=True` or CLI `--graph-show-grading-type-in-titles`.
- Randomize provider line colors (per-provider model series; deterministic per provider): `randomize_provider_line_colors=True` or CLI `--graph-randomize-provider-line-colors`.
- Filtered difficulties for grouped charts (e.g., 1 and 10): set `filtered_difficulties=[1,10]` or CLI `--graph-filtered-difficulties 1,10`.
- Consensus regrade: enable `do_consensus=True` (or CLI `--graph-consensus`) to emit parallel `_consensusGrade` graphs based on a cross-model consensus key.

Reference Lines for Per‑Model Graphs
------------------------------------
For each provider/model we add "ref" charts that overlay two baseline lines for each reasoning series:
- Naive(d1): assumes independent sub‑problems with per‑sub success equal to the observed pass rate at difficulty 1. At difficulty d, the expected pass is `pass(d) = 100 × (pass1^d)`, shown as a slate‑gray dashed line.
- Fit(all d): estimates an effective per‑sub success p by fitting `log(Pd) ≈ d·log(p)` across all difficulties using weighted least squares (weights = counts Nd) with smoothing `Pd = (Sd+0.5)/(Nd+1)` to avoid log(0). The fitted baseline is `pass(d) = 100 × (p^d)`, shown as an orange dashed line. This leverages all observed difficulties and provides a more stable reference than a single d=1 snapshot.
Both baselines are rendered for ground truth and consensus charts:
- Ground truth: `graph_<RUN>_provider_<provider>_model_<model>_ref.png`
- Consensus: `graph_<RUN>_provider_<provider>_model_<model>_ref_consensusGrade.png`
Per‑reasoning variants (single‑series) are also generated with `_reasoning_<reasoning>` in the filename and analogous `_ref`/`_ref_consensusGrade` suffixes. The WLS fit corresponds to the AGSM paper’s “expected vs actual” analysis under the iid sub‑problem approximation.

CLI Usage
---------
- Basic run on defaults:
  python algebra_dataset_benchmarking_tool.py \
    --n-per-file 10 \
    --error-pct 0.5

- Randomized sampling, with graph and custom run name:
  python algebra_dataset_benchmarking_tool.py \
    --n-per-file 25 \
    --error-pct 1.0 \
    --random \
    --graph \
    --run-name smoke_benchmark

- Ground-truth graphs with custom labels/colors and grouped d1 vs d10 bars:
  python algebra_dataset_benchmarking_tool.py \
    --base-name AGSM8K-V3-prod \
    --n-per-file 50 \
    --error-pct 0.5 \
    --workers-per-provider 10 \
    --provider-max-parallel 10 \
    --request-timeout 640 \
    --max-tokens 31000 \
    --run-name general_runV7 \
    --providers-file externalProvidersAndModelsV6.json \
    --graph \
    --sort-bars-desc \
    --graph-provider-rename ambient:zai \
    --graph-omit-run-name-in-titles \
    --graph-show-grading-type-in-titles \
    --graph-randomize-provider-line-colors \
    --graph-filtered-difficulties 1,10

- Include consensus regrade charts alongside ground-truth charts: add `--graph-consensus`

- Use a custom provider list and filter OpenAI + Together rows only:
  python algebra_dataset_benchmarking_tool.py \
    --providers-file myProviders.json \
    --select "^(openai|together):" \
    --n-per-file 5 \
    --error-pct 1.0

- Random sampling with shared seed (all models get the same items):
  python algebra_dataset_benchmarking_tool.py \
    --n-per-file 10 \
    --error-pct 1.0 \
    --random \
    --seed 123

Live Smoke Tester
-----------------
For a quick connectivity and correctness check, use:
  python live_api_smoke_test.py \
    --run-name smoke_all \
    --max-workers 6 \
    --timeout 60

This sends the prompt "What is the capital of Nebraska?" to each configured
provider/model and checks that the response contains "Lincoln". Per-model logs
are written to `logs/live_smoke_<run_name>/...` and a JSON summary is saved
under `algebraTest/benchmark_runs/<run_name>/`.

Requirements & Setup
--------------------
- Core deps: see `requirements.txt`.
- Optional SDKs for enhanced support:
  - `pip install anthropic xai-sdk` (Anthropic, xAI Grok)
  - `pip install matplotlib` (graph rendering)
- API keys: prefer environment variables
  - `AMBIENT_API_KEY`, `TOGETHER_API_KEY`, `OPENAI_API_KEY`,
    `ANTHROPIC_API_KEY`, `XAI_API_KEY`
  - You may also place literal keys in your providers JSON. Avoid committing
    secrets; keep such files gitignored.

Performance & Controls
----------------------
- Use parallel flags to balance throughput vs. rate limits:
  - `--workers-per-provider N`: subprocess workers per provider (default 1).
  - `--provider-max-parallel M`: max providers processed concurrently (0=all).
- Token/timeout tuning:
  - `--max-tokens K`: override the per-call `max_tokens` (default 50000).
  - `--request-timeout S`: per-request timeout in seconds (default 300).
- Rate limiting (per-process, per provider):
  - `--provider-qps Q`: approximate QPS cap per provider per process (default 0=off).
    For tighter control across processes, prefer lowering `--workers-per-provider`.

Sampling semantics
------------------
- Default (no `--random`): selects the first N items from each difficulty file.
  All models/providers therefore evaluate on identical items.
- With `--random --seed <S>`: selects a randomized subset deterministically from
  each difficulty (shared across models/providers), ensuring fair comparability.
- With `--random` and no `--seed`: each spec selects independently, so subsets
  may differ across models/providers and across runs.

Troubleshooting
---------------
- Slow providers or hangs: use `live_api_smoke_test.py` with `--max-workers`
  and `--timeout` to isolate and log issues.
- Missing SDKs: install provider SDKs or rely on HTTP fallbacks where supported.
- No JSON answers: tolerant parsing kicks in; see `parsing_mode` fields in
  per-model JSONL for diagnostics.

Extending Providers
-------------------
To add new providers or models, append rows in the providers JSON and, if
needed, implement a new `ProviderAdapter` subclass that mirrors the simple
`complete(prompt)` interface.

Parallel Execution & Progress Bars
----------------------------------
Run benchmarks in parallel processes with per-provider progress bars:

  python algebra_dataset_benchmarking_tool.py \
    --workers-per-provider 2 \
    --provider-max-parallel 3 \
    --n-per-file 10 \
    --error-pct 1.0

- `--workers-per-provider`: number of subprocess workers for each provider.
- `--provider-max-parallel`: cap how many providers run concurrently (0=all).
- Real-time progress bars use `tqdm` if installed; otherwise the tool prints
  simple status lines.

Token/Timeout/Rate Limits
-------------------------
- `--max-tokens K`: per-call max output tokens (default: 50000). Providers cap
  to their own limits; larger values can increase latency.
- `--request-timeout S`: per-request timeout seconds (default: 300), passed to
  underlying clients/HTTP where supported.
- `--provider-qps Q`: approximate per-process QPS cap for each provider (0=off).
  Combine with `--workers-per-provider` and `--provider-max-parallel` for a
  balanced throughput that respects provider throttles.

Implementation notes: each provider is launched in parallel in its own process
pool; per-provider tasks complete independently and progress bars update via
as-completed futures, so multiple providers can advance simultaneously.
After each run (sequential or parallel), graphs are refreshed via
`finalize_run_outputs(...)` so CLI graph customizations are applied consistently.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
import dataclasses as _dc
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


# Optional import for think-block handling from existing codebase
def _extract_think_blocks(text: str) -> Tuple[Optional[str], str]:
    """Extract <think>...</think> blocks if present; return (thinking, stripped).

    Falls back gracefully if format is absent.
    """
    if not text:
        return None, ""
    try:
        # Try importing the helper from algebra_dataset_generatorV2 for consistency
        from algebra_dataset_generatorV2 import extract_think_blocks as _etb  # type: ignore

        thinking, stripped = _etb(text)
        return thinking, stripped
    except Exception:
        s = str(text)
        out_think: List[str] = []
        # Greedy scan for <think>...
        pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
        for m in pattern.finditer(s):
            out_think.append(m.group(1))
        stripped = pattern.sub("", s)
        thinking = "\n".join(out_think) if out_think else None
        return thinking, stripped.strip()


def _coerce_float(x: Any) -> Optional[float]:
    try:
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            # Remove commas and spaces
            xs = x.strip().replace(",", "")
            # Common wrappers
            xs = xs.strip("\"'` ")
            return float(xs)
    except Exception:
        return None
    return None


def _extract_json_and_number(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[float], str]:
    """Best-effort extraction of a JSON object and numeric final_answer.

    Strategy (robust to LaTeX braces and code fences):
    - Prefer fenced JSON blocks (```json ... ``` or ``` ... ```). Parse the first
      valid JSON object found within any fenced block.
    - Try parsing the entire text as JSON.
    - Try parsing any balanced JSON-looking fragment (scan for braces and test each).
    - Regex for "final_answer: <number>" anywhere.
    - Fallback to the last standalone number in the text (avoids grabbing early coefficients).

    Returns (json_obj, number, mode) where either may be None, and mode is one of
    'json', 'json_fenced', 'json_fragment', 'final_answer_regex', 'last_number', or 'none'.
    """
    if not text:
        return None, None, "none"
    s = str(text)
    s = s.strip()

    # 1) Look for fenced code blocks containing JSON
    try:
        fence_pat = re.compile(r"```(?:json|JSON)?\s*([\s\S]*?)\s*```", re.MULTILINE)
        for m in fence_pat.finditer(s):
            block = m.group(1).strip()
            # Heuristic: extract the innermost JSON object in this block
            lb = block.find("{")
            rb = block.rfind("}")
            if lb != -1 and rb != -1 and rb > lb:
                frag = block[lb : rb + 1]
                try:
                    obj = json.loads(frag)
                    num = _coerce_float((obj or {}).get("final_answer"))
                    return obj, num, "json_fenced"
                except Exception:
                    pass
            # As a direct attempt, try the whole block as JSON
            try:
                obj = json.loads(block)
                num = _coerce_float((obj or {}).get("final_answer"))
                return obj, num, "json_fenced"
            except Exception:
                continue
    except Exception:
        pass

    # 2) Try direct JSON parse of the whole string
    try:
        obj = json.loads(s)
        num = _coerce_float((obj or {}).get("final_answer"))
        return obj, num, "json"
    except Exception:
        pass

    # 3) Scan for candidate JSON fragments by balanced braces to avoid LaTeX {}/{}
    try:
        starts: List[int] = [i for i, ch in enumerate(s) if ch == "{"]
        for st in starts:
            depth = 0
            for j in range(st, len(s)):
                if s[j] == "{":
                    depth += 1
                elif s[j] == "}":
                    depth -= 1
                    if depth == 0:
                        frag = s[st : j + 1]
                        try:
                            obj = json.loads(frag)
                            num = _coerce_float((obj or {}).get("final_answer"))
                            return obj, num, "json_fragment"
                        except Exception:
                            break  # stop at this start; try next
            # continue with next start
    except Exception:
        pass

    # 4) Regex for explicit final_answer mention
    m = re.search(r"\bfinal_answer\b\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", s)
    if m:
        return None, _coerce_float(m.group(1)), "final_answer_regex"

    # 5) Fallback: use the last standalone number to avoid grabbing early coefficients (e.g., '8*(sub_1)')
    numbers = re.findall(r"([-+]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", s)
    if numbers:
        return None, _coerce_float(numbers[-1]), "last_number"

    return None, None, "none"


def _percent_error(actual: float, proposed: float) -> float:
    denom = abs(actual)
    if denom < 1e-12:  # Avoid division by ~0
        return 0.0 if abs(proposed - actual) <= 1e-12 else float("inf")
    return 100.0 * (abs(proposed - actual) / denom)


def _compute_latency_stats(durations: List[float]) -> Dict[str, Any]:
    if not durations:
        return {"count": 0, "avg": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    ds = sorted(float(x) for x in durations if isinstance(x, (int, float)))
    n = len(ds)
    avg = sum(ds) / n
    def _pct(p: float) -> float:
        if n == 1:
            return ds[0]
        idx = max(0, min(n - 1, int(round((p / 100.0) * (n - 1)))))
        return ds[idx]
    return {
        "count": n,
        "avg": avg,
        "p50": _pct(50),
        "p95": _pct(95),
        "min": ds[0],
        "max": ds[-1],
    }


@dataclass
class ModelSpec:
    provider: str
    model: str
    reasoning_level: str  # "low" | "medium" | "high"
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    verbosity: Optional[str] = None  # e.g., for OpenAI Responses text verbosity

    @staticmethod
    def from_obj(o: Dict[str, Any]) -> "ModelSpec":
        return ModelSpec(
            provider=o.get("provider", "").strip().lower(),
            model=o.get("model", "").strip(),
            reasoning_level=(o.get("reasoning_level", "low") or "low").strip().lower(),
            api_key=(o.get("api_key") or None),
            api_key_env=(o.get("api_key_env") or None),
            verbosity=(o.get("verbosity") or None),
        )

    def resolve_key(self) -> Optional[str]:
        # Direct literal
        if self.api_key and str(self.api_key).strip():
            return str(self.api_key).strip()
        # Env var name pointing to key
        if self.api_key_env and os.getenv(self.api_key_env):
            return os.getenv(self.api_key_env)
        # Heuristic: some users place literal keys into api_key_env
        if self.api_key_env and str(self.api_key_env).strip():
            v = str(self.api_key_env).strip()
            looks_like_key = False
            pl = self.provider.lower()
            if pl == "openai":
                looks_like_key = v.startswith("sk-")
            elif pl == "together":
                looks_like_key = v.startswith("tgp_")
            elif pl == "grok":
                looks_like_key = v.startswith("xai-")
            elif pl == "anthropic":
                looks_like_key = v.startswith("sk-ant-") or v.startswith("api")
            elif pl == "ambient":
                looks_like_key = bool(re.match(r"^[0-9a-fA-F-]{20,}$", v))
            elif pl == "google":
                looks_like_key = v.startswith("AIza")
            if looks_like_key:
                return v
        # Fallbacks by provider: environment
        env_by_provider = {
            "openai": "OPENAI_API_KEY",
            "ambient": "AMBIENT_API_KEY",
            "together": "TOGETHER_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "grok": "XAI_API_KEY",  # xAI Grok
            "google": "GEMINI_API_KEY",
        }
        ek = env_by_provider.get(self.provider)
        val = os.getenv(ek) if ek else None
        if not val and self.provider == "google":
            # Try alternate env var
            val = os.getenv("GOOGLE_API_KEY")
        return val


class ProviderAdapter:
    def __init__(self, spec: ModelSpec, timeout: int = 300):
        self.spec = spec
        self.timeout = timeout
        self.last_usage: Any = None
        self.last_reasoning: Optional[str] = None
        self.last_error: Optional[str] = None

    def complete(self, prompt: str, max_tokens: int = 600, temperature: float = 0.2) -> str:
        raise NotImplementedError


class ProviderRateLimitError(RuntimeError):
    def __init__(self, provider: str, message: str = "rate limit or quota exhausted", code: Optional[int] = None):
        self.provider = provider
        self.code = code
        super().__init__(f"{provider} rate limit: {message}")


class AmbientAdapter(ProviderAdapter):
    def __init__(self, spec: ModelSpec, timeout: int = 120):
        super().__init__(spec, timeout)
        self._client = None

    def complete(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.2) -> str:
        from ambient_api_client_v3 import AmbientAPIClientV3

        if self._client is None:
            self._client = AmbientAPIClientV3(
                api_key=self.spec.resolve_key(),
                model=self.spec.model or "deepseek-ai/DeepSeek-R1",
                timeout=max(60, self.timeout),
                stream_output=False,
            )
        text = self._client.complete(prompt, max_tokens=max_tokens, temperature=temperature)
        self.last_usage = getattr(self._client, "last_usage", None)
        self.last_reasoning = getattr(self._client, "last_reasoning_content", None)
        return text


class TogetherAdapter(ProviderAdapter):
    def __init__(self, spec: ModelSpec, timeout: int = 120):
        super().__init__(spec, timeout)
        self._client = None
        self._session = None

    def complete(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.2) -> str:
        # Prefer direct HTTP to enforce per-request timeouts reliably
        import requests

        api_key = self.spec.resolve_key()
        if not api_key:
            raise RuntimeError("TOGETHER_API_KEY not set")

        if self._session is None:
            self._session = requests.Session()

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.spec.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        resp = self._session.post(
            "https://api.together.xyz/v1/chat/completions",
            headers=headers,
            json=body,
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Together API error {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        try:
            self.last_usage = data.get("usage")
        except Exception:
            self.last_usage = None
        try:
            return (data["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            return ""


class OpenAIAdapter(ProviderAdapter):
    def __init__(self, spec: ModelSpec, timeout: int = 120):
        super().__init__(spec, timeout)
        self._client = None

    def complete(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.2) -> str:
        from openai_client import OpenAIClient

        # Map reasoning level to OpenAI Responses 'reasoning.effort' where supported
        effort_map = {"low": "low", "medium": "medium", "high": "high"}
        effort = effort_map.get(self.spec.reasoning_level, None)
        if self._client is None:
            self._client = OpenAIClient(api_key=self.spec.resolve_key(), model=self.spec.model or "gpt-4o-mini", timeout=self.timeout)
        try:
            tv = (self.spec.verbosity or "low").lower()
            text = self._client.complete(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=effort,
                text_verbosity=tv,
            )
        except TypeError:
            # Older client without reasoning_effort support
            text = self._client.complete(prompt, max_tokens=max_tokens, temperature=temperature)
        self.last_usage = getattr(self._client, "last_usage", None)
        # Capture any reasoning summary and provider error
        self.last_reasoning = getattr(self._client, "last_reasoning", None)
        self.last_error = getattr(self._client, "last_error", None)
        return text


class AnthropicAdapter(ProviderAdapter):
    def __init__(self, spec: ModelSpec, timeout: int = 120):
        super().__init__(spec, timeout)
        self._client = None

    def complete(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.2) -> str:
        # Preferred: SDK usage as in sample; fallback to HTTP
        api_key = self.spec.resolve_key()
        if not api_key:
            # Try local file fallbacks to ease dev
            for cand in [Path.cwd() / "anthropic_api_key.txt", Path.home() / ".anthropic_api_key"]:
                if cand.exists():
                    txt = cand.read_text(encoding="utf-8").strip()
                    if txt:
                        api_key = txt
                        break
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        try:
            import anthropic  # type: ignore

            if self._client is None:
                # anthropic SDK may not accept timeout globally; rely on per-request
                self._client = anthropic.Anthropic(api_key=api_key)
            # Enforce an inherent ceiling for Anthropic output tokens (31k)
            mt = min(int(max_tokens or 0), 31000)
            message = self._client.messages.create(
                model=self.spec.model or "claude-3-5-sonnet-20240620",
                max_tokens=mt,
                temperature=temperature,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
            # SDK returns message.content list blocks
            chunks: List[str] = []
            for blk in getattr(message, "content", []) or []:
                try:
                    if getattr(blk, "type", None) == "text":
                        t = getattr(blk, "text", None)
                        if t:
                            chunks.append(str(t))
                except Exception:
                    pass
            text = "".join(chunks).strip()
            # usage may be present on message
            self.last_usage = getattr(message, "usage", None)
            self.last_reasoning = None
            return text
        except Exception:
            import requests

            headers = {
                "x-api-key": api_key,
                # Version pinned for broad compatibility
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            # Enforce an inherent ceiling for Anthropic output tokens (31k)
            mt = min(int(max_tokens or 0), 31000)
            body = {
                "model": self.spec.model or "claude-3-5-sonnet-20240620",
                "max_tokens": mt,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            resp = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=body, timeout=self.timeout)
            if resp.status_code != 200:
                txt = resp.text or ""
                raise RuntimeError(f"Anthropic API error {resp.status_code}: {txt[:200]}")
            data = resp.json()
            chunks: List[str] = []
            for blk in data.get("content", []) or []:
                if isinstance(blk, dict) and blk.get("type") == "text":
                    t = blk.get("text")
                    if t:
                        chunks.append(str(t))
            text = "".join(chunks).strip()
            self.last_usage = data.get("usage")
            self.last_reasoning = None
            return text


class GrokAdapter(ProviderAdapter):
    def __init__(self, spec: ModelSpec, timeout: int = 120):
        super().__init__(spec, timeout)
        self._client = None

    def complete(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.2) -> str:
        # Prefer xai_sdk as in sample; fallback to HTTP OpenAI-compatible endpoint
        api_key = self.spec.resolve_key()
        if not api_key:
            # Try local file fallbacks to ease dev
            for cand in [Path.cwd() / "xai_api_key.txt", Path.home() / ".xai_api_key"]:
                if cand.exists():
                    txt = cand.read_text(encoding="utf-8").strip()
                    if txt:
                        api_key = txt
                        break
        if not api_key:
            raise RuntimeError("XAI_API_KEY not set for Grok")
        model = self.spec.model or "grok-4"
        try:
            from xai_sdk import Client as XAIClient  # type: ignore
            from xai_sdk.chat import user as xai_user  # type: ignore

            if self._client is None:
                self._client = XAIClient(api_key=api_key)
            chat = self._client.chat.create(model=model)
            chat.append(xai_user(prompt))
            response = chat.sample()
            text = getattr(response, "content", None)
            if isinstance(text, str) and text.strip():
                return text.strip()
            # Fallback to string form
            return str(response)
        except Exception as e:
            # Try HTTP fallback below; if SDK exception hints at 429, raise rate limit
            msg = str(e).lower()
            if "429" in msg or "spending limit" in msg or "rate limit" in msg or "credits" in msg:
                raise ProviderRateLimitError("grok", message=str(e), code=429)
            import requests

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            body = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            }
            resp = requests.post("https://api.x.ai/v1/chat/completions", headers=headers, json=body, timeout=self.timeout)
            if resp.status_code == 429:
                raise ProviderRateLimitError("grok", message=resp.text[:200], code=429)
            if resp.status_code != 200:
                raise RuntimeError(f"xAI Grok API error {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            try:
                return (data["choices"][0]["message"]["content"] or "").strip()
            except Exception:
                return ""


class GoogleAdapter(ProviderAdapter):
    def __init__(self, spec: ModelSpec, timeout: int = 120):
        super().__init__(spec, timeout)
        self._session = None

    def _resolve_api_key(self) -> str:
        k = self.spec.resolve_key()
        if k and str(k).strip():
            return str(k).strip()
        # Local file fallbacks for convenience
        for cand in [Path.cwd() / "google_api_key.txt", Path.home() / ".google_api_key"]:
            if cand.exists():
                txt = cand.read_text(encoding="utf-8").strip()
                if txt:
                    return txt
        raise RuntimeError("GEMINI_API_KEY/GOOGLE_API_KEY not set; create google_api_key.txt or set env var.")

    def complete(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.2) -> str:
        import requests

        api_key = self._resolve_api_key()
        if self._session is None:
            self._session = requests.Session()
        model = self.spec.model or "gemini-2.0-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        }
        body = {
            "contents": [
                {
                    "parts": [
                        {"text": str(prompt)},
                    ]
                }
            ],
            "generationConfig": {
                "temperature": float(temperature),
                "maxOutputTokens": int(max_tokens),
            },
        }
        resp = self._session.post(url, headers=headers, json=body, timeout=self.timeout)
        if resp.status_code != 200:
            txt = resp.text or ""
            # Best-effort provider error capture
            self.last_error = txt[:500]
            raise RuntimeError(f"Google Gemini API error {resp.status_code}: {txt[:200]}")
        data = resp.json()
        # Attempt to capture usage if present (Gemini REST may not provide detailed usage)
        try:
            self.last_usage = data.get("usageMetadata")
        except Exception:
            self.last_usage = None
        # Extract text from candidates -> content.parts[].text
        out_chunks: List[str] = []
        for cand in data.get("candidates", []) or []:
            content = cand.get("content") if isinstance(cand, dict) else None
            if not isinstance(content, dict):
                continue
            for part in content.get("parts", []) or []:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    out_chunks.append(str(part["text"]))
        return "".join(out_chunks).strip()

def make_adapter(spec: ModelSpec, timeout: int = 300) -> ProviderAdapter:
    p = spec.provider.lower()
    if p == "ambient":
        return AmbientAdapter(spec, timeout=timeout)
    if p == "together":
        return TogetherAdapter(spec, timeout=timeout)
    if p == "openai":
        return OpenAIAdapter(spec, timeout=timeout)
    if p == "anthropic":
        return AnthropicAdapter(spec, timeout=timeout)
    if p == "grok":
        return GrokAdapter(spec, timeout=timeout)
    if p == "google":
        return GoogleAdapter(spec, timeout=timeout)
    raise ValueError(f"Unsupported provider: {spec.provider}")


def load_providers_config(path: Path) -> List[ModelSpec]:
    if not path.exists():
        raise FileNotFoundError(f"Providers file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "providers" in data:
        items = data["providers"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("Invalid providers config; expected list or {\"providers\": [...]}.")
    return [ModelSpec.from_obj(o) for o in items]


def pick_items(items: List[Dict[str, Any]], k: int, randomize: bool, seed: Optional[int]) -> List[Dict[str, Any]]:
    n = min(k, len(items))
    if randomize:
        rnd = random.Random(seed)
        idxs = list(range(len(items)))
        rnd.shuffle(idxs)
        return [items[i] for i in idxs[:n]]
    else:
        return items[:n]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def slugify(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("._-")


# === Parallel worker for benchmark tasks (picklable, top-level) ===
def _sanitize_for_json(obj: Any) -> Any:
    """Best-effort conversion of arbitrary objects into JSON-serializable forms.

    - Dataclasses -> asdict
    - Has model_dump()/dict()/to_dict() -> call and recurse
    - Mappings, sequences, sets -> recurse
    - Fallback -> string representation
    """
    # Primitives
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    # Decimals -> float for JSON
    try:
        from decimal import Decimal as _Decimal
        if isinstance(obj, _Decimal):
            return float(obj)
    except Exception:
        pass
    # Dataclasses
    try:
        if _dc.is_dataclass(obj):
            return _sanitize_for_json(_dc.asdict(obj))
    except Exception:
        pass
    # Pydantic v2 style
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return _sanitize_for_json(fn())
            except Exception:
                continue
    # Mapping
    from collections.abc import Mapping
    if isinstance(obj, Mapping):
        try:
            return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
        except Exception:
            # Last-resort stringify mapping
            return {str(k): str(v) for k, v in obj.items()}
    # Sequence or set
    from collections.abc import Sequence, Set
    if isinstance(obj, (Sequence, Set)) and not isinstance(obj, (str, bytes, bytearray)):
        try:
            return [_sanitize_for_json(x) for x in list(obj)]
        except Exception:
            return [str(x) for x in list(obj)]
    # Objects with __dict__
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict) and d:
        try:
            return _sanitize_for_json(d)
        except Exception:
            pass
    # Fallback
    try:
        return str(obj)
    except Exception:
        return "<unserializable>"


# Per-process adapter cache for client reuse
_ADAPTER_CACHE: Dict[str, "ProviderAdapter"] = {}


def _get_adapter_cached(spec: "ModelSpec", timeout: int) -> "ProviderAdapter":
    key = f"{spec.provider}:{spec.model}:{spec.reasoning_level}"
    ad = _ADAPTER_CACHE.get(key)
    if ad is None:
        ad = make_adapter(spec, timeout=timeout)
        _ADAPTER_CACHE[key] = ad
    return ad


# Simple per-process rate limiter per provider
class _RateLimiter:
    def __init__(self, qps: float):
        self.qps = max(0.0, float(qps or 0.0))
        self.min_interval = (1.0 / self.qps) if self.qps > 0 else 0.0
        self._last = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        import time as _t

        now = _t.monotonic()
        elapsed = now - self._last
        if elapsed < self.min_interval:
            _t.sleep(self.min_interval - elapsed)
        self._last = _t.monotonic()


_PROVIDER_LIMITERS: Dict[str, _RateLimiter] = {}


def _get_provider_limiter(provider: str, qps: float) -> _RateLimiter:
    key = f"{provider}:{qps}"
    rl = _PROVIDER_LIMITERS.get(key)
    if rl is None:
        rl = _RateLimiter(qps)
        _PROVIDER_LIMITERS[key] = rl
    return rl


def _provider_stop_flag_path(provider: str) -> Path:
    return Path("logs") / f"stop_{slugify(provider)}.flag"


def _provider_is_stopped(provider: str) -> bool:
    return _provider_stop_flag_path(provider).exists()


def _provider_mark_stopped(provider: str, reason: Optional[str] = None) -> None:
    p = _provider_stop_flag_path(provider)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        p.write_text((reason or "stopped"), encoding="utf-8")
    except Exception:
        pass

def benchmark_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single question for a given spec (executed in a subprocess).

    Expects task dict with keys: spec_dict, question, actual, acceptable_error_pct,
    max_tokens, temperature, difficulty, qid.
    """
    spec_dict = task.get("spec_dict") or {}
    spec = ModelSpec(**spec_dict)
    request_timeout = int(task.get("request_timeout", 180))
    provider_qps = float(task.get("provider_qps", 0.0) or 0.0)
    adapter = _get_adapter_cached(spec, timeout=request_timeout)
    question = task["question"]
    max_tokens = task["max_tokens"]
    temperature = task["temperature"]
    actual = task["actual"]
    acceptable_error_pct = task["acceptable_error_pct"]
    diff = task["difficulty"]
    import time as _t
    t0 = _t.time()
    start_iso = datetime.utcnow().isoformat() + "Z"
    # If provider is marked stopped, do not issue API call
    if _provider_is_stopped(spec.provider):
        t1 = _t.time()
        completed_iso = datetime.utcnow().isoformat() + "Z"
        return {
            "id": task.get("qid"),
            "difficulty": diff,
            "provider": spec.provider,
            "model": spec.model,
            "reasoning_level": spec.reasoning_level,
            "acceptable_error_pct": acceptable_error_pct,
            "actual": actual,
            "proposed": None,
            "difference": None,
            "error_pct": None,
            "matched": False,
            "tolerant_parsing_used": False,
            "parsing_mode": "none",
            "thinking": None,
            "raw_response": "",
            "final_answer": None,
            "response_text": "",
            "duration_seconds": round(t1 - t0, 6),
            "started_at": start_iso,
            "completed_at": completed_iso,
            "error_reason": f"provider_stopped:{spec.provider}",
            "usage": None,
            "parsed_json": None,
            "spec_key": f"{spec.provider}:{spec.model}:{spec.reasoning_level}",
            "provider": spec.provider,
            "provider_stopped": True,
        }

    # Backoff + retry: 3 attempts (initial, +30s, +90s)
    attempts = 0
    raw = ""
    error_reason = None
    last_exc: Optional[Exception] = None
    while attempts < 3:
        try:
            # Per-provider, per-process rate limiting
            _get_provider_limiter(spec.provider, provider_qps).wait()
            raw = adapter.complete(question, max_tokens=max_tokens, temperature=temperature)
            if isinstance(raw, str) and raw.strip():
                # Success
                error_reason = None
                break
            else:
                # Treat empty as error
                error_reason = "empty_response"
                last_exc = None
                attempts += 1
        except Exception as e:
            last_exc = e
            error_reason = f"{type(e).__name__}: {e}"
            attempts += 1
        if attempts == 1:
            import time as _t
            _t.sleep(30)
        elif attempts == 2:
            import time as _t
            _t.sleep(90)
        else:
            break
    # If repeated ProviderRateLimitError, mark provider stopped
    if attempts >= 3 and isinstance(last_exc, ProviderRateLimitError):
        _provider_mark_stopped(spec.provider, str(last_exc))
    t1 = _t.time()
    completed_iso = datetime.utcnow().isoformat() + "Z"
    thinking_source = getattr(adapter, "last_reasoning", None)
    provider_error = getattr(adapter, "last_error", None)
    # Extract thinking and result only on non-error, non-empty responses
    if error_reason is None and not (isinstance(raw, str) and raw.startswith("<error:")) and str(raw).strip():
        think_blocks, stripped = _extract_think_blocks(raw)
        thinking = thinking_source or think_blocks
        obj, proposed, parse_mode = _extract_json_and_number(stripped or raw)
    else:
        thinking = thinking_source
        obj, proposed, parse_mode = None, None, "none"
        if error_reason is None and (not str(raw).strip()):
            error_reason = "empty_response"
    diff_abs = None
    err_pct = None
    matched = False
    if proposed is not None:
        diff_abs = abs(proposed - actual)
        err_pct = _percent_error(actual, proposed)
        matched = bool(err_pct <= acceptable_error_pct)
    rec = {
        "id": task.get("qid"),
        "difficulty": diff,
        "provider": spec.provider,
        "model": spec.model,
        "reasoning_level": spec.reasoning_level,
        "acceptable_error_pct": acceptable_error_pct,
        "question": question,
        "actual": actual,
        "proposed": proposed,
        "difference": diff_abs,
        "error_pct": err_pct,
        "matched": matched,
        "tolerant_parsing_used": parse_mode != "json",
        "parsing_mode": parse_mode,
        "thinking": thinking,
        "raw_response": raw if isinstance(raw, str) else str(raw),
        # Added convenience fields
        "final_answer": proposed,
        "response_text": (stripped if (error_reason is None and 'stripped' in locals() and stripped) else (raw if isinstance(raw, str) else str(raw))),
        "duration_seconds": round(t1 - t0, 6),
        "started_at": start_iso,
        "completed_at": completed_iso,
        "error_reason": error_reason,
        "usage": _sanitize_for_json(getattr(adapter, "last_usage", None)),
        "provider_error": provider_error,
        "parsed_json": obj,
        # return spec_key components to group in parent
        "spec_key": f"{spec.provider}:{spec.model}:{spec.reasoning_level}",
        "provider": spec.provider,
    }
    return rec


def run_benchmark(
    base_name: str,
    n_per_file: int,
    acceptable_error_pct: float,
    randomize: bool,
    render_graph: bool,
    providers_file: Path,
    run_name: str,
    seed: Optional[int] = None,
    select_regex: Optional[str] = None,
    workers_per_provider: int = 1,
    provider_max_parallel: int = 0,
    max_tokens: int = 50000,
    request_timeout: int = 180,
    provider_qps: float = 0.0,
    sort_bars_desc: bool = False,
    # Graph finalize options (propagated to finalize_run_outputs)
    graph_provider_renames: Optional[Dict[str, str]] = None,
    graph_do_consensus: bool = False,
    graph_filtered_difficulties: Optional[List[int]] = None,
    graph_omit_run_name_in_titles: bool = False,
    graph_show_grading_type_in_titles: bool = False,
    graph_randomize_provider_line_colors: bool = False,
) -> None:
    algebra_dir = Path("algebraTest")
    specs = load_providers_config(providers_file)
    # Optional filter on provider/model/reasoning
    if select_regex:
        rx = re.compile(select_regex)
        specs = [s for s in specs if rx.search(f"{s.provider}:{s.model}:{s.reasoning_level}")]
    if not specs:
        print(f"[WARN] No providers matched selection '{select_regex}' in {providers_file}.")
        print("Nothing to benchmark; exiting.")
        return None

    # Prepare output dirs
    out_root = algebra_dir / "benchmark_runs" / run_name
    ensure_dir(out_root)
    # Persist minimal run metadata for later finalize steps
    try:
        meta = {
            "base_name": base_name,
            "n_per_file": int(n_per_file),
            "acceptable_error_pct": float(acceptable_error_pct),
            "randomize": bool(randomize),
            "seed": int(seed) if seed is not None else None,
            "providers_file": str(providers_file),
            "run_name": run_name,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        (out_root / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    # Summary container: per spec, per difficulty pass rates
    summary: Dict[str, Dict[str, Any]] = {}
    durations_by_provider: Dict[str, List[float]] = {}

    # Determine if sampling is comparable across models (default or random with seed)
    comparable_sampling = (not randomize) or (randomize and seed is not None)
    # Aggregator for problems missed by all models (filled only when comparable)
    spec_keys_in_run: List[str] = [f"{s.provider}:{s.model}:{s.reasoning_level}" for s in specs]
    missed_agg: Dict[Tuple[int, str], Dict[str, Any]] = {}

    # If parallel processing requested, delegate to parallel runner
    if workers_per_provider and workers_per_provider > 1 or (provider_max_parallel and provider_max_parallel > 0):
        return run_benchmark_parallel(
            base_name=base_name,
            n_per_file=n_per_file,
            acceptable_error_pct=acceptable_error_pct,
            randomize=randomize,
            render_graph=render_graph,
            providers_file=providers_file,
            run_name=run_name,
            seed=seed,
            select_regex=select_regex,
            workers_per_provider=workers_per_provider,
            provider_max_parallel=provider_max_parallel,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
            provider_qps=provider_qps,
            sort_bars_desc=sort_bars_desc,
            graph_provider_renames=graph_provider_renames,
            graph_do_consensus=graph_do_consensus,
            graph_filtered_difficulties=graph_filtered_difficulties,
            graph_omit_run_name_in_titles=graph_omit_run_name_in_titles,
            graph_show_grading_type_in_titles=graph_show_grading_type_in_titles,
            graph_randomize_provider_line_colors=graph_randomize_provider_line_colors,
        )

    for spec in specs:
        key = f"{spec.provider}:{spec.model}:{spec.reasoning_level}"
        summary[key] = {"provider": spec.provider, "model": spec.model, "reasoning_level": spec.reasoning_level, "by_difficulty": {}}
        # Per-spec detailed results path
        spec_slug = slugify(f"{spec.provider}_{spec.model}_{spec.reasoning_level}")
        spec_out_path = out_root / f"results_{spec_slug}.jsonl"
        spec_out_path_pretty = out_root / f"results_{spec_slug}.json"
        spec_duration_sum = 0.0
        spec_duration_count = 0
        per_spec_records: List[Dict[str, Any]] = []
        with spec_out_path.open("w", encoding="utf-8") as spec_out:
            for diff in range(1, 11):
                infile = algebra_dir / f"{base_name}_diff{diff}_lite.json"
                if not infile.exists():
                    print(f"[WARN] Missing input file: {infile}")
                    summary[key]["by_difficulty"][str(diff)] = {"count": 0, "passed": 0, "pass_pct": 0.0}
                    continue
                try:
                    data = json.loads(infile.read_text(encoding="utf-8"))
                except Exception as e:
                    print(f"[ERROR] Failed reading {infile}: {e}")
                    summary[key]["by_difficulty"][str(diff)] = {"count": 0, "passed": 0, "pass_pct": 0.0}
                    continue

                subset = pick_items(data, n_per_file, randomize=randomize, seed=(None if seed is None else seed + diff))
                # Map reasoning-level to temperature and token budget
                temp_map = {"low": 0.0, "medium": 0.3, "high": 0.7}
                temperature = temp_map.get(spec.reasoning_level, 0.2)
                adapter = make_adapter(spec, timeout=request_timeout)

                import time as _t
                passed = 0
                for item in subset:
                    qid = item.get("id")
                    question = item.get("question") or ""
                    actual = _coerce_float(item.get("deterministic_answer"))
                    if actual is None:
                        # Skip malformed items
                        continue
                    # Send prompt exactly as provided to keep dataset contract
                    t0 = _t.time()
                    start_iso = datetime.utcnow().isoformat() + "Z"
                    # If provider is marked stopped, skip
                    if _provider_is_stopped(spec.provider):
                        raw = ""
                        t1 = _t.time()
                        completed_iso = datetime.utcnow().isoformat() + "Z"
                        error_reason = f"provider_stopped:{spec.provider}"
                    else:
                        # Backoff + retry: 3 attempts
                        attempts = 0
                        raw = ""
                        error_reason = None
                        last_exc = None
                        while attempts < 3:
                            try:
                                raw = adapter.complete(question, max_tokens=max_tokens, temperature=temperature)
                                if isinstance(raw, str) and raw.strip():
                                    error_reason = None
                                    break
                                else:
                                    error_reason = "empty_response"
                                    last_exc = None
                                    attempts += 1
                            except Exception as e:
                                last_exc = e
                                error_reason = f"{type(e).__name__}: {e}"
                                attempts += 1
                            if attempts == 1:
                                _t.sleep(30)
                            elif attempts == 2:
                                _t.sleep(90)
                            else:
                                break
                        t1 = _t.time()
                        completed_iso = datetime.utcnow().isoformat() + "Z"
                        if attempts >= 3 and isinstance(last_exc, ProviderRateLimitError):
                            _provider_mark_stopped(spec.provider, str(last_exc))

                    # Try to capture thinking either from the provider or inline think blocks
                    thinking_source = adapter.last_reasoning
                    if error_reason is None and not (isinstance(raw, str) and raw.startswith("<error:")) and str(raw).strip():
                        think_blocks, stripped = _extract_think_blocks(raw)
                        thinking = thinking_source or think_blocks
                        obj, proposed, parse_mode = _extract_json_and_number(stripped or raw)
                    else:
                        thinking = thinking_source
                        obj, proposed, parse_mode = None, None, "none"
                        if error_reason is None and (not str(raw).strip()):
                            error_reason = "empty_response"
                    diff_abs = None
                    err_pct = None
                    matched = False
                    if proposed is not None:
                        diff_abs = abs(proposed - actual)
                        err_pct = _percent_error(actual, proposed)
                        matched = bool(err_pct <= acceptable_error_pct)
                        if matched:
                            passed += 1

                    rec = {
                        "id": qid,
                        "difficulty": diff,
                        "provider": spec.provider,
                        "model": spec.model,
                        "reasoning_level": spec.reasoning_level,
                        "acceptable_error_pct": acceptable_error_pct,
                        "question": question,
                        "actual": actual,
                        "proposed": proposed,
                        "difference": diff_abs,
                        "error_pct": err_pct,
                        "matched": matched,
                        "tolerant_parsing_used": parse_mode != "json",
                        "parsing_mode": parse_mode,
                        "thinking": thinking,
                        "raw_response": raw if isinstance(raw, str) else str(raw),
                        # Added convenience fields
                        "final_answer": proposed,
                        "response_text": (stripped if (error_reason is None and 'stripped' in locals() and stripped) else (raw if isinstance(raw, str) else str(raw))),
                        "duration_seconds": round(t1 - t0, 6),
                        "started_at": start_iso,
                        "completed_at": completed_iso,
                        "error_reason": error_reason,
                        "usage": adapter.last_usage,
                        "provider_error": getattr(adapter, "last_error", None),
                        "parsed_json": obj,
                    }
                    clean_rec = _sanitize_for_json(rec)
                    spec_out.write(json.dumps(clean_rec, ensure_ascii=False) + "\n")
                    per_spec_records.append(clean_rec)
                    # Print any per-call error to console for visibility
                    if rec.get("error_reason") or rec.get("provider_error"):
                        print(f"[ERROR] {spec.provider}:{spec.model}:{spec.reasoning_level} qid={qid} diff={diff} reason={rec.get('error_reason')} provider_error={rec.get('provider_error')}")

                    # Aggregate per-question answers for 'missed by all' report
                    if comparable_sampling and qid:
                        k = (diff, str(qid))
                        entry = missed_agg.get(k)
                        if not entry:
                            entry = {
                                "id": str(qid),
                                "difficulty": diff,
                                "question": question,
                                "actual": actual,
                                "answers_by_model": {},
                                "_matches": {},
                            }
                            missed_agg[k] = entry
                        spec_key = f"{spec.provider}:{spec.model}:{spec.reasoning_level}"
                        entry["answers_by_model"][spec_key] = rec.get("proposed")
                        entry["_matches"][spec_key] = bool(rec.get("matched"))
                    # accumulate durations per provider and per spec
                    dur = rec.get("duration_seconds")
                    if isinstance(dur, (int, float)):
                        durations_by_provider.setdefault(spec.provider, []).append(float(dur))
                        spec_duration_sum += float(dur)
                        spec_duration_count += 1
                # per difficulty stats
                cnt = len(subset)
                pass_pct = (100.0 * passed / cnt) if cnt else 0.0
                summary[key]["by_difficulty"][str(diff)] = {
                    "count": cnt,
                    "passed": passed,
                    "pass_pct": pass_pct,
                }
                # Accumulate durations for provider
                # We don't have per-item durations here to sum directly; if needed, parse from file.

        # Write per-spec rollup
        if spec_duration_count > 0:
            summary[key]["avg_duration_seconds"] = spec_duration_sum / spec_duration_count
            summary[key]["duration_count"] = spec_duration_count
        (out_root / f"summary_{spec_slug}.json").write_text(json.dumps(summary[key], ensure_ascii=False, indent=2), encoding="utf-8")
        # Write pretty JSON array alongside JSONL for readability
        try:
            spec_out_path_pretty.write_text(json.dumps(per_spec_records, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[WARN] Failed to write pretty results for {spec_slug}: {e}")

    # Global summary with provider latency breakdown
    summary["__provider_latency__"] = {prov: _compute_latency_stats(durs) for prov, durs in durations_by_provider.items()}
    (out_root / "summary_all.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # If comparable sampling, emit missed_by_all.json
    if comparable_sampling and missed_agg:
        missed_list: List[Dict[str, Any]] = []
        for (_, _), entry in missed_agg.items():
            matches = entry.get("_matches", {})
            # Include only if every model in this run attempted and failed
            if all((k in matches and not matches[k]) for k in spec_keys_in_run):
                e = {k: v for k, v in entry.items() if k != "_matches"}
                missed_list.append(e)
        (out_root / "missed_by_all.json").write_text(json.dumps(missed_list, ensure_ascii=False, indent=2), encoding="utf-8")

    # Optional graphs
    if render_graph:
        try:
            import matplotlib.pyplot as plt  # type: ignore

            def _plot_items(items_kv, title: str, out_path: Path) -> None:
                # Choose layout based on number of series to avoid legend overlap
                n_series = len(items_kv)
                fig_w, fig_h = (10, 6) if n_series <= 8 else (12, 7)
                fig = plt.figure(figsize=(fig_w, fig_h))
                ax = fig.add_subplot(1, 1, 1)
                for key, info in items_kv:
                    xs = list(range(1, 11))
                    ys = [info.get("by_difficulty", {}).get(str(d), {}).get("pass_pct", 0.0) for d in xs]
                    ax.plot(xs, ys, marker="o", label=key)
                ax.set_xticks(list(range(1, 11)))
                ax.set_xticklabels([f"diff{d}" for d in range(1, 11)])
                ax.set_ylabel("% correct")
                ax.set_xlabel("Difficulty")
                ax.set_title(title)
                ax.grid(True, alpha=0.3)

                # Legend placement: put outside to avoid covering the plot
                if n_series > 15:
                    # Place to the right for very large legends
                    box = ax.get_position()
                    ax.set_position([box.x0, box.y0, box.width * 0.74, box.height])
                    ax.legend(
                        loc="center left",
                        bbox_to_anchor=(1.02, 0.5),
                        fontsize="small",
                        frameon=False,
                    )
                else:
                    # Place below with multiple columns
                    ncol = min(max(2, (n_series + 7) // 8), 6)
                    ax.legend(
                        loc="upper center",
                        bbox_to_anchor=(0.5, -0.12),
                        ncol=ncol,
                        fontsize="small",
                        frameon=False,
                    )
                    fig.subplots_adjust(bottom=0.22)

                fig.tight_layout()
                fig.savefig(out_path, bbox_inches="tight")
                plt.close(fig)

            # All models graph
            items = [(k, v) for k, v in summary.items() if isinstance(v, dict) and "by_difficulty" in v]
            if items:
                img_path = out_root / f"graph_{slugify(run_name)}.png"
                _plot_items(items, f"Benchmark pass rates by difficulty — {run_name}", img_path)
                print(f"Saved graph to {img_path}")

            # Per-provider graphs
            by_provider: Dict[str, list] = {}
            for key, info in items:
                provider = key.split(":", 1)[0]
                by_provider.setdefault(provider, []).append((key, info))
            for provider, kv in sorted(by_provider.items()):
                if not kv:
                    continue
                img_path = out_root / f"graph_{slugify(run_name)}_provider_{slugify(provider)}.png"
                _plot_items(kv, f"{provider} — pass rates by difficulty", img_path)
                print(f"Saved provider graph to {img_path}")

            # Per-provider per-model graphs (series = reasoning levels)
            for provider, kv in sorted(by_provider.items()):
                # Group by model within this provider
                by_model: Dict[str, list] = {}
                for key, info in kv:
                    try:
                        _, model, reasoning = key.split(":", 2)
                    except ValueError:
                        parts = key.split(":")
                        model = parts[1] if len(parts) > 1 else key
                        reasoning = parts[2] if len(parts) > 2 else ""
                    # Rebuild label to use reasoning only when possible
                    label = f"{model}:{reasoning}" if not reasoning else reasoning
                    # Clone info but with label key adjusted at plot time
                    by_model.setdefault(model, []).append((label, info))
                for model, series in sorted(by_model.items()):
                    img_path = out_root / f"graph_{slugify(run_name)}_provider_{slugify(provider)}_model_{slugify(model)}.png"
                    _plot_items(series, f"{provider} / {model} — pass rates by difficulty", img_path)
                    print(f"Saved provider-model graph to {img_path}")

            # Providers overall bar chart (aggregate across difficulties and models)
            # Compute provider-level totals
            prov_totals: Dict[str, Dict[str, float]] = {}
            for key, info in items:
                prov = key.split(":", 1)[0]
                bd = info.get("by_difficulty", {})
                passed = sum(float(bd.get(str(d), {}).get("passed", 0)) for d in range(1, 11))
                count = sum(float(bd.get(str(d), {}).get("count", 0)) for d in range(1, 11))
                t = prov_totals.setdefault(prov, {"passed": 0.0, "count": 0.0})
                t["passed"] += passed
                t["count"] += count

            if prov_totals:
                # Build bar chart
                provs = list(prov_totals.keys())
                vals = [
                    (100.0 * prov_totals[p]["passed"] / prov_totals[p]["count"]) if prov_totals[p]["count"] > 0 else 0.0
                    for p in provs
                ]
                if sort_bars_desc and provs:
                    pv = sorted(zip(provs, vals), key=lambda x: (-x[1], str(x[0]).lower()))
                    provs, vals = [p for p, _ in pv], [v for _, v in pv]
                else:
                    provs = sorted(provs)
                    vals = [
                        (100.0 * prov_totals[p]["passed"] / prov_totals[p]["count"]) if prov_totals[p]["count"] > 0 else 0.0
                        for p in provs
                    ]
                fig_w, fig_h = (10, 6) if len(provs) <= 8 else (12, 7)
                fig = plt.figure(figsize=(fig_w, fig_h))
                ax = fig.add_subplot(1, 1, 1)
                bars = ax.bar(range(len(provs)), vals, color="#4c78a8")
                ax.set_xticks(range(len(provs)))
                ax.set_xticklabels([str(p) for p in provs], rotation=20, ha="right")
                ax.set_ylabel("% correct")
                ax.set_title(f"Overall % correct by provider — {run_name}")
                ax.set_ylim(0, max(100.0, (max(vals) if vals else 0.0) * 1.1))
                ax.grid(True, axis="y", alpha=0.3)
                # Annotate bars
                for rect, v in zip(bars, vals):
                    ax.annotate(f"{v:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
                img_path = out_root / f"graph_{slugify(run_name)}_providers_overall.png"
                fig.tight_layout()
                fig.savefig(img_path, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved providers overall bar to {img_path}")

            # Provider/Model (aggregated across reasoning levels) overall bar chart
            # Build totals keyed by provider:model
            prov_model_totals: Dict[str, Dict[str, float]] = {}
            for key, info in items:
                parts = key.split(":")
                prov = parts[0] if parts else key
                model = parts[1] if len(parts) > 1 else ""
                pm_key = f"{prov}:{model}"
                bd = info.get("by_difficulty", {})
                passed = sum(float(bd.get(str(d), {}).get("passed", 0)) for d in range(1, 11))
                count = sum(float(bd.get(str(d), {}).get("count", 0)) for d in range(1, 11))
                t = prov_model_totals.setdefault(pm_key, {"passed": 0.0, "count": 0.0})
                t["passed"] += passed
                t["count"] += count

            if prov_model_totals:
                labels = list(prov_model_totals.keys())
                vals = [
                    (100.0 * prov_model_totals[k]["passed"] / prov_model_totals[k]["count"]) if prov_model_totals[k]["count"] > 0 else 0.0
                    for k in labels
                ]
                if sort_bars_desc and labels:
                    lv = sorted(zip(labels, vals), key=lambda x: (-x[1], str(x[0]).lower()))
                    labels, vals = [k for k, _ in lv], [v for _, v in lv]
                else:
                    labels = sorted(labels)
                    vals = [
                        (100.0 * prov_model_totals[k]["passed"] / prov_model_totals[k]["count"]) if prov_model_totals[k]["count"] > 0 else 0.0
                        for k in labels
                    ]
                fig_w, fig_h = (12, 7) if len(labels) > 8 else (10, 6)
                fig = plt.figure(figsize=(fig_w, fig_h))
                ax = fig.add_subplot(1, 1, 1)
                bars = ax.bar(range(len(labels)), vals, color="#e45756")
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=35, ha="right")
                ax.set_ylabel("% correct")
                ax.set_title(f"Overall % correct by provider/model — {run_name}")
                ax.set_ylim(0, max(100.0, (max(vals) if vals else 0.0) * 1.1))
                ax.grid(True, axis="y", alpha=0.3)
                for rect, v in zip(bars, vals):
                    ax.annotate(f"{v:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
                img_path = out_root / f"graph_{slugify(run_name)}_providers_models_overall.png"
                fig.tight_layout()
                fig.savefig(img_path, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved provider/model overall bar to {img_path}")

            # Provider/Model/Reasoning bar chart (one bar per triple)
            spec_totals: Dict[str, Dict[str, float]] = {}
            for key, info in items:
                # key is provider:model:reasoning
                bd = info.get("by_difficulty", {})
                passed = sum(float(bd.get(str(d), {}).get("passed", 0)) for d in range(1, 11))
                count = sum(float(bd.get(str(d), {}).get("count", 0)) for d in range(1, 11))
                spec_totals[key] = {"passed": passed, "count": count}

            if spec_totals:
                # Preserve grouping: sort by provider, then model, then reasoning in order low,medium,high
                def _key_order(k: str):
                    parts = k.split(":")
                    prov = parts[0] if parts else ""
                    model = parts[1] if len(parts) > 1 else ""
                    reasoning = parts[2] if len(parts) > 2 else ""
                    rord = {"low": 0, "medium": 1, "high": 2}.get(reasoning, 3)
                    return (prov, model, rord, reasoning)
                labels = sorted(spec_totals.keys(), key=_key_order)
                vals = [
                    (100.0 * spec_totals[k]["passed"] / spec_totals[k]["count"]) if spec_totals[k]["count"] > 0 else 0.0
                    for k in labels
                ]
                colors = []
                cmap = {"low": "#1f77b4", "medium": "#ff7f0e", "high": "#2ca02c"}
                for k in labels:
                    reasoning = (k.split(":")[2] if len(k.split(":")) > 2 else "")
                    colors.append(cmap.get(reasoning, "#7f7f7f"))
                fig_w, fig_h = (max(12, len(labels) * 0.35), 7)
                fig = plt.figure(figsize=(fig_w, fig_h))
                ax = fig.add_subplot(1, 1, 1)
                xs = range(len(labels))
                bars = ax.bar(xs, vals, color=colors)
                ax.set_xticks(list(xs))
                ax.set_xticklabels(labels, rotation=40, ha="right")
                ax.set_ylabel("% correct")
                ax.set_title(f"Overall % correct by provider/model/reasoning — {run_name}")
                ax.set_ylim(0, max(100.0, (max(vals) if vals else 0.0) * 1.1))
                ax.grid(True, axis="y", alpha=0.3)
                # Legend for reasoning colors (place to the right, off-plot)
                from matplotlib.patches import Patch
                legend_elems = [Patch(facecolor=cmap[rl], label=rl) for rl in ("low", "medium", "high")]
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.78, box.height])
                ax.legend(handles=legend_elems, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="reasoning")
                for rect, v in zip(bars, vals):
                    ax.annotate(f"{v:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
                img_path = out_root / f"graph_{slugify(run_name)}_providers_models_reasoning_overall.png"
                fig.subplots_adjust(bottom=0.28)
                fig.tight_layout()
                fig.savefig(img_path, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved provider/model/reasoning overall bar to {img_path}")

            # Missed-by-all line chart (comparable sampling only)
            try:
                mba_path = out_root / "missed_by_all.json"
                if comparable_sampling and mba_path.exists():
                    mba = json.loads(mba_path.read_text(encoding="utf-8"))
                    # Count missed per difficulty
                    missed_counts = {d: 0 for d in range(1, 11)}
                    for e in mba:
                        d = int(e.get("difficulty") or 0)
                        if 1 <= d <= 10:
                            missed_counts[d] += 1
                    # Denominator: use counts from the first spec (identical across specs)
                    denom = {d: 0 for d in range(1, 11)}
                    if items:
                        _, first = items[0]
                        for d in range(1, 11):
                            denom[d] = int(first.get("by_difficulty", {}).get(str(d), {}).get("count", 0))
                    xs = list(range(1, 11))
                    ys = [((100.0 * missed_counts[d] / denom[d]) if denom[d] else 0.0) for d in xs]
                    fig = plt.figure(figsize=(10, 5))
                    ax = fig.add_subplot(1, 1, 1)
                    ax.plot(xs, ys, marker="o", color="#d62728", label="missed-by-all")
                    ax.set_xticks(xs)
                    ax.set_xticklabels([f"diff{d}" for d in xs])
                    ax.set_ylabel("% unanswered (all models)")
                    ax.set_xlabel("Difficulty")
                    ax.set_title(f"Missed by all — {run_name}")
                    ax.grid(True, alpha=0.3)
                    img_path = out_root / f"graph_{slugify(run_name)}_missed_by_all.png"
                    fig.tight_layout()
                    fig.savefig(img_path, bbox_inches="tight")
                    plt.close(fig)
                    print(f"Saved missed-by-all line to {img_path}")
            except Exception as e:
                print(f"[WARN] Missed-by-all graph skipped: {e}")
        except Exception as e:
            print(f"[WARN] Graph rendering failed or matplotlib missing: {e}")

    # After completing the run, merge prior summaries and refresh graphs
    try:
        finalize_run_outputs(
            out_root,
            sort_bars_desc=sort_bars_desc,
            provider_renames=(graph_provider_renames or None),
            do_consensus=bool(graph_do_consensus),
            filtered_difficulties=(graph_filtered_difficulties or None),
            omit_run_name_in_titles=bool(graph_omit_run_name_in_titles),
            show_grading_type_in_titles=bool(graph_show_grading_type_in_titles),
            randomize_provider_line_colors=bool(graph_randomize_provider_line_colors),
        )
        print(f"Refreshed merged summaries and graphs under ged summaries and graphs under {out_root}")
    except Exception as e:
        print(f"[WARN] Post-run merge/refresh skipped: {e}")


def run_benchmark_parallel(
    base_name: str,
    n_per_file: int,
    acceptable_error_pct: float,
    randomize: bool,
    render_graph: bool,
    providers_file: Path,
    run_name: str,
    seed: Optional[int] = None,
    select_regex: Optional[str] = None,
    workers_per_provider: int = 2,
    provider_max_parallel: int = 0,
    max_tokens: int = 50000,
    request_timeout: int = 180,
    provider_qps: float = 0.0,
    sort_bars_desc: bool = False,
    # Graph finalize options (propagated to finalize_run_outputs)
    graph_provider_renames: Optional[Dict[str, str]] = None,
    graph_do_consensus: bool = False,
    graph_filtered_difficulties: Optional[List[int]] = None,
    graph_omit_run_name_in_titles: bool = False,
    graph_show_grading_type_in_titles: bool = False,
    graph_randomize_provider_line_colors: bool = False,
) -> None:
    import concurrent.futures as _fut
    import multiprocessing as _mp
    from collections import defaultdict, Counter

    algebra_dir = Path("algebraTest")
    specs = load_providers_config(providers_file)
    if select_regex:
        rx = re.compile(select_regex)
        specs = [s for s in specs if rx.search(f"{s.provider}:{s.model}:{s.reasoning_level}")]
    if not specs:
        print(f"[WARN] No providers matched selection '{select_regex}' in {providers_file}.")
        print("Nothing to benchmark; exiting.")
        # Still write an empty summary_all.json for consistency
        out_root = Path("algebraTest") / "benchmark_runs" / run_name
        ensure_dir(out_root)
        (out_root / "summary_all.json").write_text(json.dumps({}, ensure_ascii=False, indent=2), encoding="utf-8")
        return None

    out_root = algebra_dir / "benchmark_runs" / run_name
    ensure_dir(out_root)
    # Persist minimal run metadata for later finalize steps
    try:
        meta = {
            "base_name": base_name,
            "n_per_file": int(n_per_file),
            "acceptable_error_pct": float(acceptable_error_pct),
            "randomize": bool(randomize),
            "seed": int(seed) if seed is not None else None,
            "providers_file": str(providers_file),
            "run_name": run_name,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        (out_root / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    # Construct tasks grouped by provider
    tasks_by_provider: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    summary: Dict[str, Dict[str, Any]] = {}
    per_spec_files: Dict[str, Path] = {}
    per_spec_records: Dict[str, List[Dict[str, Any]]] = {}
    temp_map = {"low": 0.0, "medium": 0.3, "high": 0.7}
    # Determine if sampling is comparable across models
    comparable_sampling = (not randomize) or (randomize and seed is not None)
    spec_keys_in_run: List[str] = [f"{s.provider}:{s.model}:{s.reasoning_level}" for s in specs]
    # Map (difficulty, id) -> question/actual for logging
    item_info: Dict[Tuple[int, str], Dict[str, Any]] = {}
    # Aggregator for missed-by-all
    missed_agg: Dict[Tuple[int, str], Dict[str, Any]] = {}

    for spec in specs:
        key = f"{spec.provider}:{spec.model}:{spec.reasoning_level}"
        summary[key] = {
            "provider": spec.provider,
            "model": spec.model,
            "reasoning_level": spec.reasoning_level,
            "by_difficulty": {},
        }
        spec_slug = slugify(f"{spec.provider}_{spec.model}_{spec.reasoning_level}")
        p = out_root / f"results_{spec_slug}.jsonl"
        p.write_text("", encoding="utf-8")
        per_spec_files[key] = p
        per_spec_records[key] = []

        for diff in range(1, 11):
            infile = algebra_dir / f"{base_name}_diff{diff}_lite.json"
            if not infile.exists():
                print(f"[WARN] Missing input file: {infile}")
                summary[key]["by_difficulty"][str(diff)] = {"count": 0, "passed": 0, "pass_pct": 0.0}
                continue
            try:
                data = json.loads(infile.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[ERROR] Failed reading {infile}: {e}")
                summary[key]["by_difficulty"][str(diff)] = {"count": 0, "passed": 0, "pass_pct": 0.0}
                continue
            subset = pick_items(data, n_per_file, randomize=randomize, seed=(None if seed is None else seed + diff))
            for item in subset:
                actual = _coerce_float(item.get("deterministic_answer"))
                if actual is None:
                    continue
                if comparable_sampling:
                    qid = str(item.get("id"))
                    if qid:
                        item_info[(diff, qid)] = {
                            "question": item.get("question") or "",
                            "actual": actual,
                        }
                task = {
                    "spec_dict": {
                        "provider": spec.provider,
                        "model": spec.model,
                        "reasoning_level": spec.reasoning_level,
                        "api_key": spec.api_key,
                        "api_key_env": spec.api_key_env,
                        "verbosity": getattr(spec, 'verbosity', None),
                    },
                    "question": item.get("question") or "",
                    "actual": actual,
                    "acceptable_error_pct": acceptable_error_pct,
                    "max_tokens": max_tokens,
                    "temperature": temp_map.get(spec.reasoning_level, 0.2),
                    "difficulty": diff,
                    "qid": item.get("id"),
                    "request_timeout": request_timeout,
                    "provider_qps": provider_qps,
                }
                tasks_by_provider[spec.provider].append(task)

    # tqdm multi-bar or basic prints
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        tqdm = None  # type: ignore

    provider_names = list(tasks_by_provider.keys())
    if provider_max_parallel and provider_max_parallel > 0:
        provider_batches = [
            provider_names[i : i + provider_max_parallel] for i in range(0, len(provider_names), provider_max_parallel)
        ]
    else:
        provider_batches = [provider_names]

    def write_rec(spec_key: str, rec: Dict[str, Any]) -> None:
        # Determine spec key from rec
        key = spec_key
        # Write to file
        for spec in specs:
            if key == f"{spec.provider}:{spec.model}:{spec.reasoning_level}":
                p = per_spec_files[key]
                with p.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(_sanitize_for_json(rec), ensure_ascii=False) + "\n")
                break

    # Global aggregations across all batches
    from collections import defaultdict as _dd
    durations_by_provider_total: Dict[str, List[float]] = _dd(list)
    spec_dur_sum_total: Dict[str, float] = _dd(float)
    spec_dur_count_total: Dict[str, int] = _dd(int)

    # Execute in batches of providers to cap process count
    for batch in provider_batches:
        # Setup bars
        bars = {}
        # For environments without tqdm, keep lightweight textual progress
        totals: Dict[str, int] = {prov: len(tasks_by_provider.get(prov, [])) for prov in batch}
        counts: Dict[str, int] = {prov: 0 for prov in batch}
        last_pct: Dict[str, int] = {prov: -1 for prov in batch}
        if tqdm:
            for i, prov in enumerate(batch):
                bars[prov] = tqdm(total=len(tasks_by_provider.get(prov, [])), position=i, desc=f"{prov}", leave=True)
        else:
            for prov in batch:
                print(f"Starting provider {prov} with {len(tasks_by_provider.get(prov, []))} tasks...")

        executors = {}
        futures: Dict[Any, str] = {}
        for prov in batch:
            ex = _fut.ProcessPoolExecutor(max_workers=max(1, int(workers_per_provider)), mp_context=_mp.get_context("spawn"))
            executors[prov] = ex
            for t in tasks_by_provider.get(prov, []):
                fut = ex.submit(benchmark_worker, t)
                futures[fut] = prov

        # Counters for this batch (will be merged into summary)
        passed_counter: Dict[str, Counter] = {k: Counter() for k in summary.keys()}
        count_counter: Dict[str, Counter] = {k: Counter() for k in summary.keys()}

        for fut in _fut.as_completed(list(futures.keys())):
            prov = futures[fut]
            try:
                rec = fut.result()
            except Exception:
                # Update progress on failure
                if tqdm and prov in bars:
                    bars[prov].update(1)
                else:
                    # textual progress fallback
                    counts[prov] = counts.get(prov, 0) + 1
                    total = max(1, totals.get(prov, 1))
                    pct = int((counts[prov] / total) * 100)
                    if pct != last_pct.get(prov, -1):
                        last_pct[prov] = pct
                        print(f"[{prov}] {counts[prov]}/{total} ({pct}%)")
                continue
            if tqdm and prov in bars:
                bars[prov].update(1)
            else:
                # textual progress fallback
                counts[prov] = counts.get(prov, 0) + 1
                total = max(1, totals.get(prov, 1))
                pct = int((counts[prov] / total) * 100)
                if pct != last_pct.get(prov, -1):
                    last_pct[prov] = pct
                    print(f"[{prov}] {counts[prov]}/{total} ({pct}%)")
            spec_key = rec.get("spec_key")
            if spec_key and spec_key in summary:
                # Drop spec_key for persisted records
                rec_clean = {k: v for k, v in rec.items() if k != "spec_key"}
                write_rec(spec_key, rec_clean)
                try:
                    per_spec_records.setdefault(spec_key, []).append(_sanitize_for_json(rec_clean))
                except Exception:
                    pass
                d = rec.get("difficulty")
                if isinstance(d, int):
                    count_counter[spec_key][str(d)] += 1
                    if rec.get("matched"):
                        passed_counter[spec_key][str(d)] += 1
                # Aggregate durations into global totals
                dur = rec.get("duration_seconds")
                if isinstance(dur, (int, float)):
                    durations_by_provider_total[prov].append(float(dur))
                    spec_dur_sum_total[spec_key] += float(dur)
                    spec_dur_count_total[spec_key] += 1
                # Print notable errors to console for visibility
                if rec.get("error_reason") or rec.get("provider_error"):
                    print(f"[ERROR] {rec.get('provider')}:{rec.get('model')}:{rec.get('reasoning_level')} qid={rec.get('id')} diff={rec.get('difficulty')} reason={rec.get('error_reason')} provider_error={rec.get('provider_error')}")

                # Aggregate per-question answers for 'missed by all'
                if comparable_sampling:
                    qid = rec.get("id")
                    dval = rec.get("difficulty")
                    if qid is not None and isinstance(dval, int):
                        k = (int(dval), str(qid))
                        entry = missed_agg.get(k)
                        if not entry:
                            ii = item_info.get(k, {})
                            entry = {
                                "id": str(qid),
                                "difficulty": int(dval),
                                "question": ii.get("question", ""),
                                "actual": ii.get("actual", rec.get("actual")),
                                "answers_by_model": {},
                                "_matches": {},
                            }
                            missed_agg[k] = entry
                        entry["answers_by_model"][spec_key] = rec.get("proposed")
                        entry["_matches"][spec_key] = bool(rec.get("matched"))

        for ex in executors.values():
            ex.shutdown(wait=True)
        if tqdm:
            for bar in bars.values():
                bar.close()
        else:
            # final summary line per provider in batch
            for prov in batch:
                c = counts.get(prov, 0)
                t = totals.get(prov, 0)
                print(f"Completed provider {prov}: {c}/{t}")

        # Merge batch counters to summary
        for key in summary.keys():
            cc = count_counter.get(key)
            pc = passed_counter.get(key)
            if not cc:
                continue
            for d in range(1, 11):
                cnt = int(cc[str(d)]) if str(d) in cc else 0
                pas = int(pc[str(d)]) if str(d) in pc else 0
                prev = summary[key]["by_difficulty"].get(str(d), {"count": 0, "passed": 0})
                new_cnt = prev.get("count", 0) + cnt
                new_pas = prev.get("passed", 0) + pas
                new_pct = (100.0 * new_pas / new_cnt) if new_cnt else 0.0
                summary[key]["by_difficulty"][str(d)] = {"count": new_cnt, "passed": new_pas, "pass_pct": new_pct}

    # Write per-spec summaries and pretty results arrays
    for spec in specs:
        key = f"{spec.provider}:{spec.model}:{spec.reasoning_level}"
        spec_slug = slugify(f"{spec.provider}_{spec.model}_{spec.reasoning_level}")
        if spec_dur_count_total.get(key, 0) > 0:
            summary[key]["avg_duration_seconds"] = spec_dur_sum_total[key] / spec_dur_count_total[key]
            summary[key]["duration_count"] = spec_dur_count_total[key]
        (out_root / f"summary_{spec_slug}.json").write_text(json.dumps(summary[key], ensure_ascii=False, indent=2), encoding="utf-8")
        # Pretty JSON array alongside JSONL
        try:
            pretty_path = out_root / f"results_{spec_slug}.json"
            pretty_path.write_text(json.dumps(per_spec_records.get(key, []), ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[WARN] Failed to write pretty results for {spec_slug}: {e}")
    # Global summary with provider latency breakdown
    summary["__provider_latency__"] = {prov: _compute_latency_stats(durs) for prov, durs in durations_by_provider_total.items()}
    (out_root / "summary_all.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # If comparable sampling, emit missed_by_all.json and detailed companion
    if comparable_sampling and missed_agg:
        missed_list: List[Dict[str, Any]] = []
        for (_, _), entry in missed_agg.items():
            matches = entry.get("_matches", {})
            if all((k in matches and not matches[k]) for k in spec_keys_in_run):
                e = {k: v for k, v in entry.items() if k != "_matches"}
                missed_list.append(e)
        (out_root / "missed_by_all.json").write_text(json.dumps(missed_list, ensure_ascii=False, indent=2), encoding="utf-8")
        # Build a detailed version using dataset lookups
        try:
            det = _try_build_merged_missed_by_all_detailed(out_root)
            if det is not None:
                (out_root / "missed_by_all_detailed.json").write_text(json.dumps(det, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[WARN] Could not generate missed_by_all_detailed.json: {e}")

    # Optional graph
    if render_graph:
        try:
            import matplotlib.pyplot as plt  # type: ignore

            def _plot_items(items_kv, title: str, out_path: Path) -> None:
                # Choose layout based on number of series to avoid legend overlap
                n_series = len(items_kv)
                fig_w, fig_h = (10, 6) if n_series <= 8 else (12, 7)
                fig = plt.figure(figsize=(fig_w, fig_h))
                ax = fig.add_subplot(1, 1, 1)
                for key, info in items_kv:
                    xs = list(range(1, 11))
                    ys = [info.get("by_difficulty", {}).get(str(d), {}).get("pass_pct", 0.0) for d in xs]
                    ax.plot(xs, ys, marker="o", label=key)
                ax.set_xticks(list(range(1, 11)))
                ax.set_xticklabels([f"diff{d}" for d in range(1, 11)])
                ax.set_ylabel("% correct")
                ax.set_xlabel("Difficulty")
                ax.set_title(title)
                ax.grid(True, alpha=0.3)

                # Legend placement: put outside to avoid covering the plot
                if n_series > 15:
                    # Place to the right for very large legends
                    box = ax.get_position()
                    ax.set_position([box.x0, box.y0, box.width * 0.74, box.height])
                    ax.legend(
                        loc="center left",
                        bbox_to_anchor=(1.02, 0.5),
                        fontsize="small",
                        frameon=False,
                    )
                else:
                    # Place below with multiple columns
                    ncol = min(max(2, (n_series + 7) // 8), 6)
                    ax.legend(
                        loc="upper center",
                        bbox_to_anchor=(0.5, -0.12),
                        ncol=ncol,
                        fontsize="small",
                        frameon=False,
                    )
                    fig.subplots_adjust(bottom=0.22)

                fig.tight_layout()
                fig.savefig(out_path, bbox_inches="tight")
                plt.close(fig)

            # All models graph
            items = [(k, v) for k, v in summary.items() if isinstance(v, dict) and "by_difficulty" in v]
            if items:
                img_path = out_root / f"graph_{slugify(run_name)}.png"
                _plot_items(items, f"Benchmark pass rates by difficulty — {run_name}", img_path)
                print(f"Saved graph to {img_path}")

            # Per-provider graphs
            by_provider: Dict[str, list] = {}
            for key, info in items:
                provider = key.split(":", 1)[0]
                by_provider.setdefault(provider, []).append((key, info))
            for provider, kv in sorted(by_provider.items()):
                if not kv:
                    continue
                img_path = out_root / f"graph_{slugify(run_name)}_provider_{slugify(provider)}.png"
                _plot_items(kv, f"{provider} — pass rates by difficulty", img_path)
                print(f"Saved provider graph to {img_path}")

            # Per-provider per-model graphs (series = reasoning levels)
            for provider, kv in sorted(by_provider.items()):
                # Group by model within this provider
                by_model: Dict[str, list] = {}
                for key, info in kv:
                    try:
                        _, model, reasoning = key.split(":", 2)
                    except ValueError:
                        parts = key.split(":")
                        model = parts[1] if len(parts) > 1 else key
                        reasoning = parts[2] if len(parts) > 2 else ""
                    label = f"{model}:{reasoning}" if not reasoning else reasoning
                    by_model.setdefault(model, []).append((label, info))
                for model, series in sorted(by_model.items()):
                    img_path = out_root / f"graph_{slugify(run_name)}_provider_{slugify(provider)}_model_{slugify(model)}.png"
                    _plot_items(series, f"{provider} / {model} — pass rates by difficulty", img_path)
                    print(f"Saved provider-model graph to {img_path}")

            # Providers overall bar chart (aggregate across difficulties and models)
            prov_totals: Dict[str, Dict[str, float]] = {}
            for key, info in items:
                prov = key.split(":", 1)[0]
                bd = info.get("by_difficulty", {})
                passed = sum(float(bd.get(str(d), {}).get("passed", 0)) for d in range(1, 11))
                count = sum(float(bd.get(str(d), {}).get("count", 0)) for d in range(1, 11))
                t = prov_totals.setdefault(prov, {"passed": 0.0, "count": 0.0})
                t["passed"] += passed
                t["count"] += count

            if prov_totals:
                provs = list(prov_totals.keys())
                vals = [
                    (100.0 * prov_totals[p]["passed"] / prov_totals[p]["count"]) if prov_totals[p]["count"] > 0 else 0.0
                    for p in provs
                ]
                if sort_bars_desc and provs:
                    pv = sorted(zip(provs, vals), key=lambda x: (-x[1], str(x[0]).lower()))
                    provs, vals = [p for p, _ in pv], [v for _, v in pv]
                else:
                    provs = sorted(provs)
                    vals = [
                        (100.0 * prov_totals[p]["passed"] / prov_totals[p]["count"]) if prov_totals[p]["count"] > 0 else 0.0
                        for p in provs
                    ]
                fig_w, fig_h = (10, 6) if len(provs) <= 8 else (12, 7)
                fig = plt.figure(figsize=(fig_w, fig_h))
                ax = fig.add_subplot(1, 1, 1)
                bars = ax.bar(range(len(provs)), vals, color="#4c78a8")
                ax.set_xticks(range(len(provs)))
                ax.set_xticklabels([str(p) for p in provs], rotation=20, ha="right")
                ax.set_ylabel("% correct")
                ax.set_title(f"Overall % correct by provider — {run_name}")
                ax.set_ylim(0, max(100.0, (max(vals) if vals else 0.0) * 1.1))
                ax.grid(True, axis="y", alpha=0.3)
                for rect, v in zip(bars, vals):
                    ax.annotate(f"{v:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
                img_path = out_root / f"graph_{slugify(run_name)}_providers_overall.png"
                fig.tight_layout()
                fig.savefig(img_path, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved providers overall bar to {img_path}")

            # Provider/Model (aggregated across reasoning) overall bar chart
            prov_model_totals: Dict[str, Dict[str, float]] = {}
            for key, info in items:
                parts = key.split(":")
                prov = parts[0] if parts else key
                model = parts[1] if len(parts) > 1 else ""
                pm_key = f"{prov}:{model}"
                bd = info.get("by_difficulty", {})
                passed = sum(float(bd.get(str(d), {}).get("passed", 0)) for d in range(1, 11))
                count = sum(float(bd.get(str(d), {}).get("count", 0)) for d in range(1, 11))
                t = prov_model_totals.setdefault(pm_key, {"passed": 0.0, "count": 0.0})
                t["passed"] += passed
                t["count"] += count

            if prov_model_totals:
                labels = list(prov_model_totals.keys())
                vals = [
                    (100.0 * prov_model_totals[k]["passed"] / prov_model_totals[k]["count"]) if prov_model_totals[k]["count"] > 0 else 0.0
                    for k in labels
                ]
                if sort_bars_desc and labels:
                    lv = sorted(zip(labels, vals), key=lambda x: (-x[1], str(x[0]).lower()))
                    labels, vals = [k for k, _ in lv], [v for _, v in lv]
                else:
                    labels = sorted(labels)
                    vals = [
                        (100.0 * prov_model_totals[k]["passed"] / prov_model_totals[k]["count"]) if prov_model_totals[k]["count"] > 0 else 0.0
                        for k in labels
                    ]
                fig_w, fig_h = (12, 7) if len(labels) > 8 else (10, 6)
                fig = plt.figure(figsize=(fig_w, fig_h))
                ax = fig.add_subplot(1, 1, 1)
                bars = ax.bar(range(len(labels)), vals, color="#e45756")
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=35, ha="right")
                ax.set_ylabel("% correct")
                ax.set_title(f"Overall % correct by provider/model — {run_name}")
                ax.set_ylim(0, max(100.0, (max(vals) if vals else 0.0) * 1.1))
                ax.grid(True, axis="y", alpha=0.3)
                for rect, v in zip(bars, vals):
                    ax.annotate(f"{v:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
                img_path = out_root / f"graph_{slugify(run_name)}_providers_models_overall.png"
                fig.tight_layout()
                fig.savefig(img_path, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved provider/model overall bar to {img_path}")

            # Per-provider models overall bar chart (aggregate across difficulties per spec)
            for provider, kv in sorted(by_provider.items()):
                if not kv:
                    continue
                models = []
                vals = []
                for key, info in sorted(kv):
                    parts = key.split(":")
                    model = parts[1] if len(parts) > 1 else key
                    reasoning = parts[2] if len(parts) > 2 else ""
                    label = f"{model}:{reasoning}" if reasoning else model
                    bd = info.get("by_difficulty", {})
                    passed = sum(float(bd.get(str(d), {}).get("passed", 0)) for d in range(1, 11))
                    count = sum(float(bd.get(str(d), {}).get("count", 0)) for d in range(1, 11))
                    pct = (100.0 * passed / count) if count > 0 else 0.0
                    models.append(label)
                    vals.append(pct)
                if sort_bars_desc and models:
                    mv = sorted(zip(models, vals), key=lambda x: (-x[1], str(x[0]).lower()))
                    models, vals = [m for m, _ in mv], [v for _, v in mv]
                if not models:
                    continue
                fig_w, fig_h = (10, 6) if len(models) <= 8 else (12, 7)
                fig = plt.figure(figsize=(fig_w, fig_h))
                ax = fig.add_subplot(1, 1, 1)
                bars = ax.bar(range(len(models)), vals, color="#72b7b2")
                ax.set_xticks(range(len(models)))
                ax.set_xticklabels(models, rotation=25, ha="right")
                ax.set_ylabel("% correct")
                ax.set_title(f"Overall % correct by model — {provider} — {run_name}")
                ax.set_ylim(0, max(100.0, (max(vals) if vals else 0.0) * 1.1))
                ax.grid(True, axis="y", alpha=0.3)
                for rect, v in zip(bars, vals):
                    ax.annotate(f"{v:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
                img_path = out_root / f"graph_{slugify(run_name)}_provider_{slugify(provider)}_models_overall.png"
                fig.tight_layout()
                fig.savefig(img_path, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved provider models overall bar to {img_path}")

            
        except Exception as e:
            print(f"[WARN] Graph rendering failed or matplotlib missing: {e}")

    # After completing the run, merge prior summaries and refresh graphs (new, consistent charts)
    try:
        finalize_run_outputs(
            out_root,
            sort_bars_desc=sort_bars_desc,
            provider_renames=(graph_provider_renames or None),
            do_consensus=bool(graph_do_consensus),
            filtered_difficulties=(graph_filtered_difficulties or None),
            omit_run_name_in_titles=bool(graph_omit_run_name_in_titles),
            show_grading_type_in_titles=bool(graph_show_grading_type_in_titles),
            randomize_provider_line_colors=bool(graph_randomize_provider_line_colors),
        )
        print(f"Refreshed merged summaries and graphs under {out_root}")
    except Exception as e:
        print(f"[WARN] Post-run merge/refresh skipped: {e}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Algebra dataset benchmarking tool")
    p.add_argument("--n-per-file", type=int, default=25, help="Number of questions per difficulty file")
    p.add_argument("--error-pct", type=float, default=0.0, help="Acceptable error percentage for numeric answers")
    p.add_argument("--base-name", type=str, default="AGSM8K", help="Base name of files in algebraTest (e.g., AGSM8K)")
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Benchmark run name (default: timestamp_benchmark)",
    )
    p.add_argument("--random", action="store_true", help="Sample randomly (non-repeating) from each file")
    p.add_argument("--graph", action="store_true", help="Render graphs (lines by difficulty; provider and per-model bar charts)")
    p.add_argument(
        "--providers-file",
        type=str,
        default="defaultProvidersAndModels.json",
        help="Path to providers/models JSON list",
    )
    p.add_argument("--seed", type=int, default=None, help="Random seed when using --random")
    p.add_argument("--select", type=str, default=None, help="Regex to filter provider:model:reasoning")
    p.add_argument("--workers-per-provider", type=int, default=1, help="Parallel workers per provider (processes)")
    p.add_argument("--provider-max-parallel", type=int, default=0, help="Max providers processed concurrently (0=all)")
    p.add_argument("--max-tokens", type=int, default=50000, help="Per-call max_tokens override")
    p.add_argument("--request-timeout", type=int, default=300, help="Per-request timeout seconds")
    p.add_argument("--provider-qps", type=float, default=0.0, help="Approx per-process QPS cap per provider (0=off)")
    p.add_argument(
        "--sort-bars-desc",
        action="store_true",
        help="Order bar charts by descending values (left→right highest to lowest)",
    )
    # Graph customization options (applied via finalize_run_outputs)
    p.add_argument(
        "--graph-provider-rename",
        action="append",
        default=[],
        help="Rename providers in graph labels, e.g., 'ambient:zai'. Repeatable.",
    )
    p.add_argument(
        "--graph-omit-run-name-in-titles",
        action="store_true",
        help="Omit run directory name from graph titles.",
    )
    p.add_argument(
        "--graph-show-grading-type-in-titles",
        action="store_true",
        help="Include grading type (Ground truth or Consensus) in graph titles.",
    )
    p.add_argument(
        "--graph-randomize-provider-line-colors",
        action="store_true",
        help="Randomize line colors within each provider's per-model charts (deterministic per provider).",
    )
    p.add_argument(
        "--graph-filtered-difficulties",
        type=str,
        default=None,
        help="Comma-separated difficulties to include in grouped per-model bar charts (default: 1,10).",
    )
    p.add_argument(
        "--graph-consensus",
        action="store_true",
        help="Also compute consensus regrade and render _consensusGrade graphs.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_name = args.run_name or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_benchmark"
    # Parse graph customization options
    provider_renames: Dict[str, str] = {}
    for mapping in (args.graph_provider_rename or []):
        try:
            if not isinstance(mapping, str) or ":" not in mapping:
                continue
            a, b = mapping.split(":", 1)
            a = (a or "").strip()
            b = (b or "").strip()
            if a and b:
                provider_renames[a] = b
        except Exception:
            continue
    filtered_difficulties: Optional[List[int]] = None
    if getattr(args, "graph_filtered_difficulties", None):
        try:
            filtered_difficulties = [int(x) for x in str(args.graph_filtered_difficulties).split(",") if str(x).strip()]
        except Exception:
            filtered_difficulties = None
    try:
        run_benchmark(
            base_name=args.base_name,
            n_per_file=args.n_per_file,
            acceptable_error_pct=args.error_pct,
            randomize=bool(args.random),
            render_graph=bool(args.graph),
            providers_file=Path(args.providers_file),
            run_name=run_name,
            seed=args.seed,
            select_regex=args.select,
            workers_per_provider=int(args.workers_per_provider),
            provider_max_parallel=int(args.provider_max_parallel),
            max_tokens=int(args.max_tokens),
            request_timeout=int(args.request_timeout),
            provider_qps=float(args.provider_qps),
            sort_bars_desc=bool(args.sort_bars_desc),
            graph_provider_renames=(provider_renames or None),
            graph_do_consensus=bool(args.graph_consensus),
            graph_filtered_difficulties=filtered_difficulties,
            graph_omit_run_name_in_titles=bool(args.graph_omit_run_name_in_titles),
            graph_show_grading_type_in_titles=bool(args.graph_show_grading_type_in_titles),
            graph_randomize_provider_line_colors=bool(args.graph_randomize_provider_line_colors),
        )
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# --- Utilities to merge prior results in a run dir and refresh graphs ---
def _merge_summaries_in_dir(run_dir: Path, exclude: Optional[Union[str, List[str]]] = None) -> Dict[str, Dict[str, Any]]:
    """Scan run_dir for summary_*.json files and merge into a dict keyed by provider:model:reasoning.

    Returns the merged summary mapping that can be written to summary_all.json.
    """
    merged: Dict[str, Dict[str, Any]] = {}

    def _excluded(spec_key: str, model: str) -> bool:
        if not exclude:
            return False
        patterns: List[str]
        if isinstance(exclude, str):
            patterns = [exclude]
        else:
            patterns = [str(p) for p in (exclude or [])]
        for pat in patterns:
            try:
                rx = re.compile(pat)
            except Exception:
                rx = re.compile(re.escape(pat))
            if rx.search(spec_key) or rx.search(model):
                return True
        return False
    for p in run_dir.glob("summary_*.json"):
        if p.name == "summary_all.json":
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            prov = (data.get("provider") or "").strip()
            model = (data.get("model") or "").strip()
            rl = (data.get("reasoning_level") or "").strip()
            if not prov or not model or not rl:
                continue
            key = f"{prov}:{model}:{rl}"
            if _excluded(key, model):
                continue
            merged[key] = data
        except Exception:
            continue
    return merged


def _try_build_merged_missed_by_all(
    run_dir: Path,
    exclude: Optional[Union[str, List[str]]] = None,
    *,
    dataset_backfill: bool = True,
    prefer_lite: bool = True,
    dataset_base_name: Optional[str] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Attempt to compute a merged missed_by_all across all results_*.jsonl in run_dir.

    Returns a list of entries or None if comparability cannot be established.
    """
    # Load all results files
    results = list(run_dir.glob("results_*.jsonl"))
    # Exclude result files by inspecting their first valid record
    def _excluded(spec_key: str, model: str) -> bool:
        if not exclude:
            return False
        patterns: List[str]
        if isinstance(exclude, str):
            patterns = [exclude]
        else:
            patterns = [str(p) for p in (exclude or [])]
        for pat in patterns:
            try:
                rx = re.compile(pat)
            except Exception:
                rx = re.compile(re.escape(pat))
            if rx.search(spec_key) or rx.search(model):
                return True
        return False
    if not results:
        return None
    # Collect per-spec data
    per_spec: Dict[str, Dict[str, Any]] = {}
    per_spec_sets: Dict[str, Dict[int, set]] = {}
    for rp in results:
        try:
            # Infer spec key from filename: results_<provider>_<model>_<reasoning>.jsonl
            name = rp.stem  # results_<slug>
            slug = name[len("results_"):]
            # We'll prefer the spec_key embedded in each record when available
            recs: Dict[Tuple[int, str], Dict[str, Any]] = {}
            diff_sets: Dict[int, set] = {d: set() for d in range(1, 11)}
            with rp.open("r", encoding="utf-8") as f:
                # Determine exclusion for this file by peeking at the first valid line
                peeked_key = None
                for line in f:
                    try:
                        o = json.loads(line)
                    except Exception:
                        continue
                    if peeked_key is None:
                        prov = (o.get("provider") or "").strip()
                        model = (o.get("model") or "").strip()
                        rl = (o.get("reasoning_level") or "").strip()
                        peeked_key = f"{prov}:{model}:{rl}"
                        if _excluded(peeked_key, model):
                            # skip the entire file
                            recs = {}
                            diff_sets = {d: set() for d in range(1, 11)}
                            break
                    qid = o.get("id")
                    d = o.get("difficulty")
                    if qid is None or not isinstance(d, int):
                        continue
                    key = (d, str(qid))
                    recs[key] = {
                        "id": str(qid),
                        "difficulty": int(d),
                        "question": o.get("question"),
                        "actual": o.get("actual"),
                        "proposed": o.get("proposed"),
                        "matched": bool(o.get("matched")),
                        "provider": o.get("provider"),
                        "model": o.get("model"),
                        "reasoning_level": o.get("reasoning_level"),
                    }
                    diff_sets[int(d)].add(str(qid))
            if recs:
                per_spec[slug] = recs
                per_spec_sets[slug] = diff_sets
        except Exception:
            continue
    if not per_spec:
        return None
    # Verify comparability: all specs must have identical id sets per difficulty
    slugs = sorted(per_spec_sets.keys())
    ref = per_spec_sets[slugs[0]]
    for slug in slugs[1:]:
        ds = per_spec_sets[slug]
        for d in range(1, 11):
            if (ref.get(d) or set()) != (ds.get(d) or set()):
                return None
    # Streaming backfill helper: fetch only needed ids for a difficulty
    def _pick_fields(obj: Dict[str, Any], fields: tuple[str, ...]) -> Dict[str, Any]:
        rec = {k: obj.get(k) for k in fields}
        # Gather all sub-component details (lite or full schema)
        subs = obj.get("sub_components") or {}
        eq_flat: list = []
        eq_by_sub: Dict[str, Any] = {}
        sol_by_sub: Dict[str, Any] = {}
        if isinstance(subs, dict):
            for sub_key in sorted(subs.keys()):
                sub_obj = (subs.get(sub_key) or {})
                src = (sub_obj.get("source") or {})
                # eq_system_str list
                eqs = sub_obj.get("eq_system_str")
                if eqs is None:
                    eqs = src.get("eq_system_str")
                if eqs is not None:
                    try:
                        # ensure list for flatten
                        if isinstance(eqs, list):
                            eq_flat.extend(eqs)
                        else:
                            eq_flat.append(eqs)
                    except Exception:
                        pass
                    eq_by_sub[sub_key] = eqs
                # solution_eval per sub
                sev = sub_obj.get("solution_eval")
                if sev is None:
                    sev = src.get("solution_eval")
                if sev is not None:
                    sol_by_sub[sub_key] = sev
        if "eq_system_str" in fields:
            rec["eq_system_str"] = eq_flat if eq_flat else rec.get("eq_system_str")
        # Always include by-sub maps when available
        if eq_by_sub:
            rec["eq_system_str_by_sub"] = eq_by_sub
        if sol_by_sub:
            rec["solution_eval_by_sub"] = sol_by_sub
        return rec

    def _stream_backfill_for_diff(d: int, needed_ids: set[str], fields: tuple[str, ...]) -> Dict[str, Dict[str, Any]]:
        if not dataset_backfill or not needed_ids:
            return {}
        # Build candidate list: prefer lite
        from glob import glob as _glob
        base = Path("algebraTest")
        if dataset_base_name:
            lite = [base / f"{dataset_base_name}_diff{int(d)}_lite.json"]
            heavy = [base / f"{dataset_base_name}_diff{int(d)}.json"]
        else:
            lite = [Path(p) for p in _glob(str(base / f"*_diff{int(d)}_lite.json"))]
            heavy = [Path(p) for p in _glob(str(base / f"*_diff{int(d)}.json"))]
        candidates: List[Path] = sorted(lite) if (prefer_lite and lite) else (sorted(lite) if lite else sorted(heavy))
        result: Dict[str, Dict[str, Any]] = {}
        # Try streaming with ijson; if unavailable, fall back to safe small-file load
        try:
            import ijson  # type: ignore
        except Exception:
            for p in candidates:
                try:
                    if p.stat().st_size > 200 * 1024 * 1024:
                        # Avoid loading huge files without ijson
                        continue
                except Exception:
                    pass
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    continue
                for it in (data or []):
                    iid = str((it or {}).get("id"))
                    if iid in needed_ids:
                        result[iid] = _pick_fields(it, fields)
                        if len(result) == len(needed_ids):
                            return result
            return result
        # Streaming path
        for p in candidates:
            try:
                with p.open('rb') as f:
                    for obj in ijson.items(f, 'item'):
                        try:
                            iid = str((obj or {}).get('id'))
                        except Exception:
                            iid = None
                        if iid in needed_ids:
                            result[iid] = _pick_fields(obj, fields)
                            if len(result) == len(needed_ids):
                                return result
            except Exception:
                continue
        return result

    # Build missed-by-all list by iterating over reference spec (two-pass to batch backfill)
    temp: List[Dict[str, Any]] = []
    needed_by_diff: Dict[int, set[str]] = {}
    ref_recs = per_spec[slugs[0]]
    for (d, qid), r0 in ref_recs.items():
        all_attempted = True
        all_failed = True
        answers_by_model: Dict[str, Any] = {}
        for slug in slugs:
            rec = per_spec[slug].get((d, qid))
            if rec is None:
                all_attempted = False
                all_failed = False
                break
            spec_key = f"{rec.get('provider')}:{rec.get('model')}:{rec.get('reasoning_level')}"
            answers_by_model[spec_key] = rec.get("proposed")
            if rec.get("matched"):
                all_failed = False
        if all_attempted and all_failed:
            dq = r0.get("question")
            da = r0.get("actual")
            entry = {
                "id": str(qid),
                "difficulty": int(d),
                "question": dq,
                "actual": da,
                "answers_by_model": answers_by_model,
            }
            temp.append(entry)
            if dataset_backfill and (not dq or da is None):
                S = needed_by_diff.setdefault(int(d), set())
                S.add(str(qid))
    # Perform streaming backfill per difficulty
    if dataset_backfill and needed_by_diff:
        for d, ids in needed_by_diff.items():
            fetched = _stream_backfill_for_diff(d, ids, ("question", "deterministic_answer"))
            if not fetched:
                continue
            for e in temp:
                if int(e.get("difficulty", 0)) != int(d):
                    continue
                qid = str(e.get("id"))
                if qid in fetched:
                    if not e.get("question"):
                        e["question"] = fetched[qid].get("question")
                    if e.get("actual") is None:
                        e["actual"] = fetched[qid].get("deterministic_answer")
    return temp


def _try_build_merged_missed_by_all_detailed(
    run_dir: Path,
    exclude: Optional[Union[str, List[str]]] = None,
    *,
    prefer_lite: bool = True,
    dataset_base_name: Optional[str] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Like _try_build_merged_missed_by_all, but enrich each entry with
    equation_template, eq_system_str, solution_eval from the base dataset when available.
    Returns a list or None if comparability cannot be established.
    """
    base_list = _try_build_merged_missed_by_all(run_dir, exclude=exclude, dataset_backfill=False, prefer_lite=prefer_lite)
    if base_list is None:
        return None
    # Collect needed ids by difficulty
    needed_by_diff: Dict[int, set[str]] = {}
    for entry in base_list:
        d = int(entry.get("difficulty") or 0)
        qid = str(entry.get("id") or "")
        if qid:
            needed_by_diff.setdefault(d, set()).add(qid)
    # Stream backfill per diff
    all_details: Dict[tuple[int, str], Dict[str, Any]] = {}
    fields = ("equation_template", "eq_system_str", "solution_eval")
    def_fields = ("question", "deterministic_answer") + fields
    def _pick_fields(obj: Dict[str, Any], fields: tuple[str, ...]) -> Dict[str, Any]:
        rec = {k: obj.get(k) for k in fields}
        subs = obj.get("sub_components") or {}
        eq_flat: list = []
        eq_by_sub: Dict[str, Any] = {}
        sol_by_sub: Dict[str, Any] = {}
        if isinstance(subs, dict):
            for sub_key in sorted(subs.keys()):
                sub_obj = (subs.get(sub_key) or {})
                src = (sub_obj.get("source") or {})
                eqs = sub_obj.get("eq_system_str")
                if eqs is None:
                    eqs = src.get("eq_system_str")
                if eqs is not None:
                    if isinstance(eqs, list):
                        eq_flat.extend(eqs)
                    else:
                        eq_flat.append(eqs)
                    eq_by_sub[sub_key] = eqs
                sev = sub_obj.get("solution_eval")
                if sev is None:
                    sev = src.get("solution_eval")
                if sev is not None:
                    sol_by_sub[sub_key] = sev
        if "eq_system_str" in fields:
            rec["eq_system_str"] = eq_flat if eq_flat else rec.get("eq_system_str")
        if eq_by_sub:
            rec["eq_system_str_by_sub"] = eq_by_sub
        if sol_by_sub:
            rec["solution_eval_by_sub"] = sol_by_sub
        return rec

    for d, ids in needed_by_diff.items():
        if not ids:
            continue
        # Build candidates limited to base name when provided
        from glob import glob as _glob
        base_dir = Path("algebraTest")
        if dataset_base_name:
            lite = [base_dir / f"{dataset_base_name}_diff{int(d)}_lite.json"]
            heavy = [base_dir / f"{dataset_base_name}_diff{int(d)}.json"]
        else:
            lite = [Path(p) for p in _glob(str(base_dir / f"*_diff{int(d)}_lite.json"))]
            heavy = [Path(p) for p in _glob(str(base_dir / f"*_diff{int(d)}.json"))]
        candidates = sorted(lite) if (prefer_lite and lite) else (sorted(lite) if lite else sorted(heavy))
        fetched_for_d: Dict[str, Dict[str, Any]] = {}
        # Try streaming first
        try:
            import ijson  # type: ignore
        except Exception:
            ijson = None  # type: ignore
        for p in candidates:
            try:
                if not p.exists():
                    continue
                if ijson is not None:
                    with p.open('rb') as f:
                        for obj in ijson.items(f, 'item'):
                            iid = str((obj or {}).get('id'))
                            if iid in ids:
                                fetched_for_d[iid] = _pick_fields(obj, def_fields)
                                if len(fetched_for_d) == len(ids):
                                    break
                else:
                    # Only load small files without ijson
                    if p.stat().st_size > 200 * 1024 * 1024:
                        continue
                    data = json.loads(p.read_text(encoding="utf-8"))
                    for it in (data or []):
                        iid = str((it or {}).get("id"))
                        if iid in ids:
                            fetched_for_d[iid] = _pick_fields(it, def_fields)
                            if len(fetched_for_d) == len(ids):
                                break
            except Exception:
                continue
            if len(fetched_for_d) == len(ids):
                break
        for iid, rec in fetched_for_d.items():
            all_details[(d, str(iid))] = rec
        # Fallback pass: fill missing equation_template from full dataset if lite lacked it
        missing_tpl = {iid for iid, rec in fetched_for_d.items() if rec.get("equation_template") in (None, "")}
        if missing_tpl:
            # Try heavy file for equation_template only
            heavy_candidates = []
            if dataset_base_name:
                heavy_candidates = [base_dir / f"{dataset_base_name}_diff{int(d)}.json"]
            else:
                from glob import glob as _glob
                heavy_candidates = [Path(p) for p in _glob(str(base_dir / f"*_diff{int(d)}.json"))]
            for hp in heavy_candidates:
                try:
                    import ijson  # type: ignore
                    with hp.open('rb') as f:
                        for obj in ijson.items(f, 'item'):
                            iid = str((obj or {}).get('id'))
                            if iid in missing_tpl:
                                et = (obj or {}).get('equation_template')
                                if et:
                                    all_details[(d, iid)]["equation_template"] = et
                                    missing_tpl.discard(iid)
                            if not missing_tpl:
                                break
                except Exception:
                    try:
                        # Small file fallback
                        data = json.loads(hp.read_text(encoding='utf-8'))
                        for it in (data or []):
                            iid = str((it or {}).get('id'))
                            if iid in missing_tpl and (it or {}).get('equation_template'):
                                all_details[(d, iid)]["equation_template"] = it.get('equation_template')
                                missing_tpl.discard(iid)
                            if not missing_tpl:
                                break
                    except Exception:
                        continue
                if not missing_tpl:
                    break
    # Build detailed list
    detailed: List[Dict[str, Any]] = []
    for entry in base_list:
        d = int(entry.get("difficulty") or 0)
        qid = str(entry.get("id") or "")
        ds = all_details.get((d, qid), {})
        detailed.append({
            **entry,
            "equation_template": ds.get("equation_template"),
            # Flattened across sub-components
            "eq_system_str": ds.get("eq_system_str"),
            # Detailed per sub-component
            "eq_system_str_by_sub": ds.get("eq_system_str_by_sub"),
            "solution_eval_by_sub": ds.get("solution_eval_by_sub"),
            # Keep legacy solution_eval if present (may be from first sub)
            "solution_eval": ds.get("solution_eval"),
            # Also backfill question/actual if still missing
            "question": entry.get("question") or ds.get("question"),
            "actual": entry.get("actual") if entry.get("actual") is not None else ds.get("deterministic_answer"),
        })
    return detailed


def _enrich_entries_with_dataset_fields(
    entries: List[Dict[str, Any]],
    *,
    prefer_lite: bool = True,
    dataset_base_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Given a list of entries with at least {id, difficulty}, return a new list with
    dataset-derived fields attached when available: question (if missing),
    equation_template, eq_system_str, solution_eval, and actual (if missing).
    """
    from glob import glob as _glob
    # Collect ids needed by difficulty
    needed_by_diff: Dict[int, set[str]] = {}
    for e in entries or []:
        d = int(e.get("difficulty") or 0)
        qid = str(e.get("id") or "")
        if qid:
            needed_by_diff.setdefault(d, set()).add(qid)
    # Stream backfill per difficulty
    backfills: Dict[tuple[int, str], Dict[str, Any]] = {}
    fields = ("question", "deterministic_answer", "equation_template", "eq_system_str", "solution_eval")
    def _pick_fields(obj: Dict[str, Any], fields: tuple[str, ...]) -> Dict[str, Any]:
        rec = {k: obj.get(k) for k in fields}
        subs = obj.get("sub_components") or {}
        eq_flat: list = []
        eq_by_sub: Dict[str, Any] = {}
        sol_by_sub: Dict[str, Any] = {}
        if isinstance(subs, dict):
            for sub_key in sorted(subs.keys()):
                sub_obj = (subs.get(sub_key) or {})
                src = (sub_obj.get("source") or {})
                # eq_system_str may be on sub_obj (lite) or under source (full)
                eqs = sub_obj.get("eq_system_str")
                if eqs is None:
                    eqs = src.get("eq_system_str")
                if eqs is not None:
                    if isinstance(eqs, list):
                        eq_flat.extend(eqs)
                    else:
                        eq_flat.append(eqs)
                    eq_by_sub[sub_key] = eqs
                # solution_eval may be on sub_obj (lite) or under source (full)
                sev = sub_obj.get("solution_eval")
                if sev is None:
                    sev = src.get("solution_eval")
                if sev is not None:
                    sol_by_sub[sub_key] = sev
        if "eq_system_str" in fields:
            rec["eq_system_str"] = eq_flat if eq_flat else rec.get("eq_system_str")
        if eq_by_sub:
            rec["eq_system_str_by_sub"] = eq_by_sub
        if sol_by_sub:
            rec["solution_eval_by_sub"] = sol_by_sub
        return rec
    from glob import glob as _glob
    import os as _os
    base = Path("algebraTest")
    # Try streaming via ijson; fall back to small-file loads
    try:
        import ijson  # type: ignore
    except Exception:
        ijson = None  # type: ignore
    for d, ids in needed_by_diff.items():
        if not ids:
            continue
        if dataset_base_name:
            lite = [base / f"{dataset_base_name}_diff{int(d)}_lite.json"]
            heavy = [base / f"{dataset_base_name}_diff{int(d)}.json"]
        else:
            lite = [Path(p) for p in _glob(str(base / f"*_diff{int(d)}_lite.json"))]
            heavy = [Path(p) for p in _glob(str(base / f"*_diff{int(d)}.json"))]
        candidates = sorted(lite) if (prefer_lite and lite) else (sorted(lite) if lite else sorted(heavy))
        fetched_for_d: Dict[str, Dict[str, Any]] = {}
        for p in candidates:
            try:
                if not p.exists():
                    continue
                if ijson is not None:
                    with p.open('rb') as f:
                        for obj in ijson.items(f, 'item'):
                            try:
                                iid = str((obj or {}).get('id'))
                            except Exception:
                                iid = None
                            if iid in ids:
                                fetched_for_d[iid] = _pick_fields(obj, fields)
                                if len(fetched_for_d) == len(ids):
                                    break
                else:
                    # Only load small files without ijson
                    if p.stat().st_size > 200 * 1024 * 1024:
                        continue
                    data = json.loads(p.read_text(encoding="utf-8"))
                    for it in (data or []):
                        iid = str((it or {}).get("id"))
                        if iid in ids:
                            fetched_for_d[iid] = _pick_fields(it, fields)
                            if len(fetched_for_d) == len(ids):
                                break
            except Exception:
                continue
            if len(fetched_for_d) == len(ids):
                break
        for iid, rec in fetched_for_d.items():
            backfills[(int(d), str(iid))] = rec

    enriched: List[Dict[str, Any]] = []
    for e in entries or []:
        d = int(e.get("difficulty") or 0)
        qid = str(e.get("id") or "")
        ds = backfills.get((d, qid), {})
        out = dict(e)
        if not out.get("question"):
            q = ds.get("question")
            if q:
                out["question"] = q
        if out.get("actual") is None and ds.get("deterministic_answer") is not None:
            out["actual"] = ds.get("deterministic_answer")
        out["equation_template"] = ds.get("equation_template")
        # Flattened and per-sub variants (if present in backfill)
        out["eq_system_str"] = ds.get("eq_system_str")
        if ds.get("eq_system_str_by_sub") is not None:
            out["eq_system_str_by_sub"] = ds.get("eq_system_str_by_sub")
        if ds.get("solution_eval_by_sub") is not None:
            out["solution_eval_by_sub"] = ds.get("solution_eval_by_sub")
        # Keep legacy solution_eval if present
        if ds.get("solution_eval") is not None:
            out["solution_eval"] = ds.get("solution_eval")
        enriched.append(out)
    return enriched


def finalize_run_outputs(
    run_dir: str | Path,
    exclude: Optional[Union[str, List[str]]] = None,
    sort_bars_desc: bool = False,
    provider_renames: Optional[Dict[str, str]] = None,
    # Consensus analysis options (optional)
    do_consensus: bool = False,
    consensus_pct: float = 0.5,
    consensus_min_votes: int = 2,
    consensus_error_pct: float = 0.5,
    fallback_mode: str = "ground_truth",
    filtered_difficulties: Optional[List[int]] = None,
    omit_run_name_in_titles: bool = False,
    show_grading_type_in_titles: bool = False,
    randomize_provider_line_colors: bool = False,
    # New: control heavy file construction and dataset backfilling
    build_missed: bool = False,
    build_missed_detailed: bool = False,
    dataset_backfill: bool = True,
    prefer_lite: bool = True,
    dataset_base_name: Optional[str] = None,
) -> None:
    """Merge summaries in run_dir, refresh graphs, and recreate missed_by_all if comparable.

    - When `exclude` is provided, excluded specs are omitted from regenerated graphs.
      `exclude` may be a regex string or a list of regex strings. Patterns are tested
      against the spec key `provider:model:reasoning` and against the plain `model`.
    - `provider_renames` (optional): mapping applied to provider names for labels/titles in
      regenerated graphs only (e.g., {"ambient": "zai"}). Raw files and JSON remain unchanged.
    """
    runp = Path(run_dir)
    # Merge summaries
    merged = _merge_summaries_in_dir(runp, exclude=exclude)
    if merged:
        (runp / "summary_all.json").write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    # Try to load dataset base from run metadata if not supplied
    if dataset_base_name is None:
        try:
            meta_path = runp / "run_meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                dbn = str(meta.get("base_name") or "").strip()
                if dbn:
                    dataset_base_name = dbn
        except Exception:
            pass

    # Optionally create missed_by_all files (now opt-in to avoid OOM)
    if build_missed:
        try:
            mba = _try_build_merged_missed_by_all(
                runp,
                exclude=exclude,
                dataset_backfill=dataset_backfill,
                prefer_lite=prefer_lite,
                dataset_base_name=dataset_base_name,
            )
        except Exception as e:
            print(f"[WARN] Unable to build missed_by_all.json: {e}")
            mba = None
        if mba is not None:
            (runp / "missed_by_all.json").write_text(json.dumps(_sanitize_for_json(mba), ensure_ascii=False, indent=2), encoding="utf-8")
            if build_missed_detailed:
                try:
                    mba_det = _try_build_merged_missed_by_all_detailed(
                        runp,
                        exclude=exclude,
                        prefer_lite=prefer_lite,
                        dataset_base_name=dataset_base_name,
                    )
                    if mba_det is not None:
                        (runp / "missed_by_all_detailed.json").write_text(json.dumps(_sanitize_for_json(mba_det), ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception as e:
                    print(f"[WARN] Unable to build missed_by_all_detailed.json: {e}")
    # Regenerate graphs (original grading)
    try:
        regenerate_graphs_for_run(
            runp,
            exclude=exclude,
            sort_bars_desc=sort_bars_desc,
            provider_renames=provider_renames,
            summary_filename="summary_all.json",
            file_suffix="",
            missed_by_all_filename="missed_by_all.json",
            filtered_difficulties=filtered_difficulties,
            omit_run_name_in_titles=omit_run_name_in_titles,
            show_grading_type_in_titles=show_grading_type_in_titles,
            randomize_provider_line_colors=randomize_provider_line_colors,
        )
    except Exception as e:
        print(f"[WARN] Unable to regenerate graphs: {e}")

    # Optional: consensus-based regrade and graphs
    if do_consensus:
        try:
            _consensus_regrade_and_graphs(
                runp,
                exclude=exclude,
                sort_bars_desc=sort_bars_desc,
                provider_renames=provider_renames,
                consensus_pct=float(consensus_pct),
                consensus_min_votes=int(consensus_min_votes),
                consensus_error_pct=float(consensus_error_pct),
                fallback_mode=str(fallback_mode or "ground_truth").strip().lower(),
                filtered_difficulties=filtered_difficulties,
                omit_run_name_in_titles=omit_run_name_in_titles,
                show_grading_type_in_titles=show_grading_type_in_titles,
                randomize_provider_line_colors=randomize_provider_line_colors,
                dataset_base_name=dataset_base_name,
            )
        except Exception as e:
            print(f"[WARN] Consensus regrade failed: {e}")


# Utility: regenerate graphs for an existing run directory from summary_all.json
def regenerate_graphs_for_run(
    run_dir: str | Path,
    exclude: Optional[Union[str, List[str]]] = None,
    sort_bars_desc: bool = False,
    provider_renames: Optional[Dict[str, str]] = None,
    summary_filename: str = "summary_all.json",
    file_suffix: str = "",
    missed_by_all_filename: str = "missed_by_all.json",
    filtered_difficulties: Optional[List[int]] = None,
    omit_run_name_in_titles: bool = False,
    show_grading_type_in_titles: bool = False,
    randomize_provider_line_colors: bool = False,
) -> List[Path]:
    """Regenerate graphs for a completed run: all-model, per-provider, per-provider per-model,
    and providers overall bar chart.

    Expects `summary_all.json` to be present under `run_dir`.
    Returns a list of Paths to the saved image files.
    """
    from typing import Any, Dict, List, Tuple

    out_root = Path(run_dir)
    summ_path = out_root / summary_filename
    if not summ_path.exists():
        raise FileNotFoundError(f"{summ_path.name} not found under {out_root}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        raise RuntimeError(f"matplotlib not available: {e}")

    summary = json.loads(summ_path.read_text(encoding="utf-8"))
    items = [(k, v) for k, v in summary.items() if isinstance(v, dict) and "by_difficulty" in v]

    # Optional provider label remapping for display
    provider_renames = provider_renames or {}

    def _disp_provider(p: str) -> str:
        try:
            return provider_renames.get(p, p)
        except Exception:
            return p

    def _disp_key(key: str) -> str:
        # Replace provider prefix in a spec key like "provider:model:reasoning"
        if not isinstance(key, str):
            return str(key)
        parts = key.split(":", 1)
        if not parts:
            return key
        prov = parts[0]
        rest = parts[1] if len(parts) > 1 else ""
        new_prov = _disp_provider(prov)
        return f"{new_prov}:{rest}" if rest else new_prov

    # ---------- Stable color mapping (provider base + model shade by diff10) ----------
    # Build provider list deterministically (sorted by name) for base color assignment
    providers_sorted = sorted({k.split(":", 1)[0] for k, _ in items}, key=lambda s: s.lower())
    base_cmap = plt.get_cmap("tab10")
    provider_base_color = {p: base_cmap(i % base_cmap.N) for i, p in enumerate(providers_sorted)}

    # Compute per (provider, model) performance at difficulty 10 aggregated across reasoning
    pm_counts_d10: Dict[tuple, Dict[str, float]] = {}
    for key, info in items:
        parts = key.split(":")
        prov = parts[0] if parts else key
        model = parts[1] if len(parts) > 1 else ""
        d10 = (info.get("by_difficulty", {}) or {}).get("10", {})
        try:
            c = float(d10.get("count", 0) or 0)
            p = float(d10.get("passed", 0) or 0)
        except Exception:
            c, p = 0.0, 0.0
        pmk = (prov, model)
        acc = pm_counts_d10.setdefault(pmk, {"count": 0.0, "passed": 0.0})
        acc["count"] += c
        acc["passed"] += p

    # For each provider, rank its models by diff10 pass rate desc; assign darkest shade to best
    def _linspace(a: float, b: float, n: int) -> list:
        if n <= 1:
            return [b]
        step = (b - a) / float(n - 1)
        return [a + i * step for i in range(n)]

    from matplotlib.colors import to_rgb
    def _mix_with_white(rgb, alpha: float):
        r, g, b = to_rgb(rgb)
        return (1 - alpha) * 1.0 + alpha * r, (1 - alpha) * 1.0 + alpha * g, (1 - alpha) * 1.0 + alpha * b

    # Build model color mapping
    model_color: Dict[tuple, tuple] = {}
    top_model_by_provider: Dict[str, str] = {}
    # Collect models per provider
    models_by_provider: Dict[str, set] = {}
    for (prov, model) in pm_counts_d10.keys():
        models_by_provider.setdefault(prov, set()).add(model)
    for prov in providers_sorted:
        models = sorted(list(models_by_provider.get(prov, [])))
        # Sort models by diff10 perf desc
        def _perf(m: str) -> float:
            acc = pm_counts_d10.get((prov, m), {"count": 0.0, "passed": 0.0})
            return (acc["passed"] / acc["count"]) if acc["count"] > 0 else 0.0
        models_sorted_desc = sorted(models, key=lambda m: (-_perf(m), m.lower()))
        if models_sorted_desc:
            top_model_by_provider[prov] = models_sorted_desc[0]
        shades = _linspace(0.55, 0.95, max(1, len(models_sorted_desc)))
        # Darkest shade (highest alpha) to best model
        for shade, m in zip(reversed(shades), models_sorted_desc):
            model_color[(prov, m)] = _mix_with_white(provider_base_color[prov], shade)

    def _color_for_spec_key(spec_key: str) -> tuple:
        parts = spec_key.split(":")
        prov = parts[0] if parts else spec_key
        model = parts[1] if len(parts) > 1 else ""
        return model_color.get((prov, model), provider_base_color.get(prov, (0.5, 0.5, 0.5)))

    # Apply exclusion filter if provided
    def _excluded(spec_key: str) -> bool:
        if not exclude:
            return False
        patterns: List[str]
        if isinstance(exclude, str):
            patterns = [exclude]
        else:
            patterns = [str(p) for p in (exclude or [])]
        key = spec_key
        # Also check model-only part for convenience
        parts = key.split(":")
        model = parts[1] if len(parts) > 1 else key
        for pat in patterns:
            try:
                rx = re.compile(pat)
            except Exception:
                # treat as literal
                rx = re.compile(re.escape(pat))
            if rx.search(key) or rx.search(model):
                return True
        return False

    if exclude:
        items = [(k, v) for (k, v) in items if not _excluded(k)]

    # Title helper: include grading type and optionally run name
    def _title(base: str) -> str:
        parts = [base]
        if show_grading_type_in_titles:
            gt = "Consensus grading" if ("consensus" in summary_filename.lower() or "consensus" in file_suffix.lower()) else "Ground truth grading"
            parts.append(gt)
        if not omit_run_name_in_titles:
            parts.append(out_root.name)
        return " — ".join(parts)

    def _plot_items(items_kv: List[Tuple[str, Dict[str, Any]]], title: str, out_path: Path, colors_by_key: Optional[Dict[str, tuple]] = None) -> None:
        n_series = len(items_kv)
        fig_w, fig_h = (10, 6) if n_series <= 8 else (12, 7)
        fig = plt.figure(figsize=(fig_w, fig_h))
        ax = fig.add_subplot(1, 1, 1)
        # Fixed, deterministic color per series based on provider+model shade mapping
        for idx, (key, info) in enumerate(items_kv):
            xs = list(range(1, 11))
            ys = [info.get("by_difficulty", {}).get(str(d), {}).get("pass_pct", 0.0) for d in xs]
            col = _color_for_spec_key(key) if not colors_by_key else colors_by_key.get(key, _color_for_spec_key(key))
            ax.plot(xs, ys, marker="o", label=_disp_key(key), color=col)
        ax.set_xticks(list(range(1, 11)))
        ax.set_xticklabels([f"diff{d}" for d in range(1, 11)])
        ax.set_ylabel("% correct")
        ax.set_xlabel("Difficulty")
        ax.set_title(_title(title))
        ax.grid(True, alpha=0.3)

        if n_series > 15:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.74, box.height])
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize="small", frameon=False)
        else:
            ncol = min(max(2, (n_series + 7) // 8), 6)
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=ncol, fontsize="small", frameon=False)
            fig.subplots_adjust(bottom=0.22)

        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        try:
            import gc as _gc
            _gc.collect()
        except Exception:
            pass

    def _plot_items_with_reference(
        items_kv: List[Tuple[str, Dict[str, Any]]],
        naive_by_label: Dict[str, List[float]],
        title: str,
        out_path: Path,
        colors_by_key: Optional[Dict[str, tuple]] = None,
        fit_by_label: Optional[Dict[str, List[float]]] = None,
        fit_ci_by_label: Optional[Dict[str, Tuple[List[float], List[float]]]] = None,
    ) -> None:
        # Plot actual series (consistent color) and overlay two reference lines per series when available:
        #  - naive (from diff1): slate gray dashed
        #  - fit(all d): orange dashed (log-linear fit of p from all difficulties)
        n_series = len(items_kv)
        fig_w, fig_h = (10, 6) if n_series <= 8 else (12, 7)
        fig = plt.figure(figsize=(fig_w, fig_h))
        ax = fig.add_subplot(1, 1, 1)
        xs = list(range(1, 11))
        for key, info in items_kv:
            ys = [info.get("by_difficulty", {}).get(str(d), {}).get("pass_pct", 0.0) for d in xs]
            col = _color_for_spec_key(key) if not colors_by_key else colors_by_key.get(key, _color_for_spec_key(key))
            ax.plot(xs, ys, marker="o", label=_disp_key(key), color=col)
        ref_color = (0.44, 0.5, 0.56)  # slate gray approx (#708090)
        fit_color = (1.0, 0.5, 0.0)    # orange-like (#ff7f0e)
        for key, _ in items_kv:
            ys_ref = naive_by_label.get(key)
            if not ys_ref:
                continue
            ax.plot(xs, ys_ref, linestyle="--", color=ref_color, alpha=0.9, label=f"naive({_disp_key(key)})")
        if fit_by_label:
            for key, _ in items_kv:
                ys_fit = fit_by_label.get(key)
                if not ys_fit:
                    continue
                ax.plot(xs, ys_fit, linestyle=(0, (5, 3)), color=fit_color, alpha=0.9, label=f"fit({_disp_key(key)})")
                # Optional confidence band
                if fit_ci_by_label and fit_ci_by_label.get(key):
                    lo, hi = fit_ci_by_label[key]
                    try:
                        ax.fill_between(xs, lo, hi, color=fit_color, alpha=0.15, linewidth=0)
                    except Exception:
                        pass
        ax.set_xticks(xs)
        ax.set_xticklabels([f"diff{d}" for d in xs])
        ax.set_ylabel("% correct")
        ax.set_xlabel("Difficulty")
        ax.set_title(_title(title))
        ax.grid(True, alpha=0.3)
        n_series_total = (
            len(items_kv)
            + sum(1 for k,_ in items_kv if naive_by_label.get(k))
            + (sum(1 for k,_ in items_kv if fit_by_label and fit_by_label.get(k)) if fit_by_label else 0)
        )
        if n_series_total > 15:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.74, box.height])
            leg = ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize="small", frameon=False)
            # For right-side legends, keep caption at a fixed bottom position
            caption_y = 0.02
        else:
            ncol = min(max(2, (n_series_total + 7) // 8), 6)
            # Place legend below the plot; we'll measure its true size and insert caption underneath deterministically
            leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=ncol, fontsize="small", frameon=False)
            # Force a draw to compute precise legend bounds, then place caption below it
            try:
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                bbox_px = leg.get_window_extent(renderer=renderer)
                bbox_fig = bbox_px.transformed(fig.transFigure.inverted())
                # Put caption 0.02 figure units below the legend bottom, but not below y=0.005
                caption_y = max(0.005, bbox_fig.y0 - 0.02)
            except Exception:
                caption_y = 0.015
        # Short caption explaining baselines
        try:
            caption = (
                "Baselines: naive(d1) pass(d)=100·(pass1^d); fit(all d) via WLS on log(Pd) with smoothing, "
                "95% band by delta method."
            )
            fig.text(0.5, caption_y, caption, ha="center", va="bottom", fontsize=8, color=(0.25, 0.25, 0.25))
        except Exception:
            pass
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        try:
            import gc as _gc
            _gc.collect()
        except Exception:
            pass

    saved: List[Path] = []
    if items:
        img_path = out_root / f"graph_{slugify(out_root.name)}{file_suffix}.png"
        _plot_items(items, "Benchmark pass rates by difficulty", img_path)
        saved.append(img_path)

    # Per-provider graphs
    by_provider: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for key, info in items:
        provider = key.split(":", 1)[0]
        by_provider.setdefault(provider, []).append((key, info))
    for provider, kv in sorted(by_provider.items()):
        if not kv:
            continue
        disp_provider = _disp_provider(provider)
        img_path = out_root / f"graph_{slugify(out_root.name)}_provider_{slugify(disp_provider)}{file_suffix}.png"
        colors_override = None
        if randomize_provider_line_colors:
            # Randomize model colors within this provider except keep top d10 model darkest provider shade
            import hashlib, random as _rnd
            # Group by model within this provider
            by_model: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
            for k, info in kv:
                parts = k.split(":")
                model = parts[1] if len(parts) > 1 else ""
                by_model.setdefault(model, []).append((k, info))
            # Identify top model from fixed mapping to keep consistency across charts
            top_model = top_model_by_provider.get(provider)
            # Seeded RNG per provider for reproducibility
            seed = int(hashlib.md5(provider.encode("utf-8")).hexdigest(), 16) & 0xffffffff
            rng = _rnd.Random(seed)
            # Build color overrides per key
            colors_override = {}
            # Build a distinct color palette, avoiding provider base family hue
            import math
            base_rgb = provider_base_color.get(provider, (0.5, 0.5, 0.5))
            cmap20 = plt.get_cmap("tab20")
            palette = [cmap20(i) for i in range(cmap20.N)]
            def _dist(c1, c2):
                return math.sqrt(sum((a-b)*(a-b) for a,b in zip(c1[:3], c2[:3])))
            # Filter out colors too close to provider base
            palette = [c for c in palette if _dist(c, base_rgb) > 0.25]
            rng.shuffle(palette)
            pal_idx = 0
            for m, entries in by_model.items():
                if m == top_model:
                    # Keep fixed provider/model shade
                    col = model_color.get((provider, m), provider_base_color.get(provider, (0.5, 0.5, 0.5)))
                else:
                    # Assign next palette color; fallback to HSV if exhausted
                    if pal_idx < len(palette):
                        col = palette[pal_idx]
                        pal_idx += 1
                    else:
                        import colorsys
                        h = rng.random(); s = 0.8; v = 0.9
                        col = colorsys.hsv_to_rgb(h, s, v)
                for k, _info in entries:
                    colors_override[k] = col
        _plot_items(kv, f"{disp_provider} — pass rates by difficulty", img_path, colors_by_key=colors_override)
        saved.append(img_path)

    # Per-provider per-model graphs with naive reference overlay (series = reasoning levels)
    by_provider_model: Dict[Tuple[str, str], List[Tuple[str, Dict[str, Any]]]] = {}
    for key, info in items:
        parts = key.split(":")
        prov = parts[0] if parts else key
        model = parts[1] if len(parts) > 1 else ""
        # Label series by reasoning (fallback to full key)
        reasoning = parts[2] if len(parts) > 2 else ""
        label = f"{model}:{reasoning}" if reasoning else reasoning or key
        by_provider_model.setdefault((prov, model), []).append((key, info))
    # Helper to compute fitted pass line using all difficulties (log-linear fit p from Pd ≈ p^d)
    def _fit_all_d_line(bd_map: Dict[str, Any]) -> Optional[Tuple[List[float], List[float], List[float]]]:
        import math as _math
        num = 0.0
        den = 0.0
        has_any = False
        # Collect to estimate residual variance
        ys: List[Tuple[float, float]] = []  # (d, log(Pd)) duplicated by weight chunks
        w_sum = 0.0
        for d in range(1, 11):
            dmap = bd_map.get(str(d), {}) or {}
            try:
                cnt = float(dmap.get("count", 0) or 0)
                pas = float(dmap.get("passed", 0) or 0)
            except Exception:
                cnt, pas = 0.0, 0.0
            if cnt <= 0:
                # try pass_pct fallback
                try:
                    pct = float(dmap.get("pass_pct", 0.0) or 0.0)
                    Pd = max(0.0, min(1.0, pct / 100.0))
                    w = 1.0
                except Exception:
                    continue
            else:
                # smoothed proportion to avoid log(0)
                Pd = (pas + 0.5) / (cnt + 1.0)
                w = cnt
            Pd = max(1e-9, min(1.0, Pd))
            has_any = True
            num += w * d * _math.log(Pd)
            den += w * (d * d)
            w_sum += w
            # Store weighted contributions for residual variance estimate
            # We won’t duplicate rows; we’ll use weights explicitly in residual calc
            ys.append((float(d), _math.log(Pd)))
        if not has_any or den <= 0:
            return None
        slope = num / den
        p_hat = _math.exp(slope)
        p_hat = max(0.0, min(1.0, p_hat))
        # Approximate residual variance (weighted): RSS = sum(w*(y - s*d)^2)
        try:
            rss = 0.0
            for d, y in ys:
                # Recover weight for this d
                dmap = bd_map.get(str(int(d)), {}) or {}
                try:
                    w = float(dmap.get("count", 0) or 0)
                except Exception:
                    w = 1.0
                if w <= 0:
                    w = 1.0
                rss += w * (y - slope * d) ** 2
            dof = max(1.0, (w_sum if w_sum > 0 else len(ys)) - 1.0)
            sigma2 = rss / dof
            var_slope = sigma2 / den
            se_slope = _math.sqrt(max(0.0, var_slope))
        except Exception:
            se_slope = 0.0
        # Build 95% CI for pass(d) = 100 * exp(d*slope)
        ys_fit = []
        ci_lo = []
        ci_hi = []
        for d in range(1, 11):
            yhat = d * slope
            # Delta method: Var(exp(yhat)) ≈ exp(2*yhat) * Var(yhat); with Var(yhat)=d^2 Var(slope)
            var_yhat = (d * se_slope) ** 2
            Phat = _math.exp(yhat)
            se_P = Phat * _math.sqrt(max(0.0, var_yhat))
            lo = max(0.0, min(1.0, Phat - 1.96 * se_P))
            hi = max(0.0, min(1.0, Phat + 1.96 * se_P))
            ys_fit.append(Phat * 100.0)
            ci_lo.append(lo * 100.0)
            ci_hi.append(hi * 100.0)
        return ys_fit, ci_lo, ci_hi

    for (prov, model), entries in sorted(by_provider_model.items()):
        # Build series list with display labels (reasoning) and info
        series: List[Tuple[str, Dict[str, Any]]] = []
        for spec_key, info in entries:
            parts = spec_key.split(":")
            reasoning = parts[2] if len(parts) > 2 else ""
            label = f"{model}:{reasoning}" if not reasoning else reasoning
            series.append((spec_key, info))  # keep spec_key for color; display uses _disp_key inside plotter
        # Naive expected pass per difficulty using diff1 failure per series
        naive_by_label: Dict[str, List[float]] = {}
        fit_by_label: Dict[str, List[float]] = {}
        fit_ci_by_label: Dict[str, Tuple[List[float], List[float]]] = {}
        for spec_key, info in series:
            bd = info.get("by_difficulty", {}) or {}
            d1 = bd.get("1", {})
            try:
                c1 = float(d1.get("count", 0) or 0)
                p1 = float(d1.get("passed", 0) or 0)
            except Exception:
                c1, p1 = 0.0, 0.0
            pass1 = None
            if c1 > 0:
                pass1 = max(0.0, min(1.0, (p1 / c1)))
            else:
                # Fallback: use recorded pass_pct for difficulty 1 if counts are missing
                try:
                    pct = float(d1.get("pass_pct", 0.0) or 0.0)
                    pass1 = max(0.0, min(1.0, pct / 100.0))
                except Exception:
                    pass1 = None
            if pass1 is None:
                # still attempt fit
                pass
            # Naive expected pass at difficulty d assuming independence: pass1^d
            if pass1 is not None:
                naive_by_label[spec_key] = [(pass1 ** d) * 100.0 for d in range(1, 11)]
            # Fit using all difficulties
            fit_tuple = _fit_all_d_line(bd)
            if fit_tuple is not None:
                ys_fit, lo, hi = fit_tuple
                fit_by_label[spec_key] = ys_fit
                fit_ci_by_label[spec_key] = (lo, hi)
        disp_provider = _disp_provider(prov)
        ref_img_path = out_root / f"graph_{slugify(out_root.name)}_provider_{slugify(disp_provider)}_model_{slugify(model)}_ref{file_suffix}.png"
        _plot_items_with_reference(series, naive_by_label, f"{disp_provider} / {model} — pass vs naive", ref_img_path, fit_by_label=fit_by_label, fit_ci_by_label=fit_ci_by_label)
        saved.append(ref_img_path)

    # Per-provider per-model per-reasoning graphs (single-series) with naive reference
    for key, info in items:
        parts = key.split(":")
        prov = parts[0] if parts else key
        model = parts[1] if len(parts) > 1 else ""
        reasoning = parts[2] if len(parts) > 2 else ""
        disp_provider = _disp_provider(prov)
        series = [(key, info)]
        # Compute naive baseline from diff1 for this series
        bd = info.get("by_difficulty", {}) or {}
        d1 = bd.get("1", {})
        pass1 = None
        try:
            c1 = float(d1.get("count", 0) or 0)
            p1 = float(d1.get("passed", 0) or 0)
            if c1 > 0:
                pass1 = max(0.0, min(1.0, (p1 / c1)))
        except Exception:
            pass
        if pass1 is None:
            try:
                pct = float(d1.get("pass_pct", 0.0) or 0.0)
                pass1 = max(0.0, min(1.0, pct / 100.0))
            except Exception:
                pass1 = None
        naive_by_label = {}
        if pass1 is not None:
            naive_by_label[key] = [(pass1 ** d) * 100.0 for d in range(1, 11)]
        # Fit using all difficulties for single series
        fit_by_label = {}
        fit_ci_by_label = {}
        fit_tuple = _fit_all_d_line(bd)
        if fit_tuple is not None:
            ys_fit, lo, hi = fit_tuple
            fit_by_label[key] = ys_fit
            fit_ci_by_label[key] = (lo, hi)
        # Save normal single-series graph
        single_img_path = out_root / f"graph_{slugify(out_root.name)}_provider_{slugify(disp_provider)}_model_{slugify(model)}_reasoning_{slugify(reasoning or 'unknown')}{file_suffix}.png"
        _plot_items(series, f"{disp_provider} / {model} / {reasoning or 'unknown'} — pass rates by difficulty", single_img_path)
        saved.append(single_img_path)
        # Save ref overlay single-series graph
        ref_single_img_path = out_root / f"graph_{slugify(out_root.name)}_provider_{slugify(disp_provider)}_model_{slugify(model)}_reasoning_{slugify(reasoning or 'unknown')}_ref{file_suffix}.png"
        _plot_items_with_reference(series, naive_by_label, f"{disp_provider} / {model} / {reasoning or 'unknown'} — pass vs naive", ref_single_img_path, fit_by_label=fit_by_label, fit_ci_by_label=fit_ci_by_label)
        saved.append(ref_single_img_path)

    # Providers overall bar chart (aggregate across difficulties and models)
    prov_totals: Dict[str, Dict[str, float]] = {}
    for key, info in items:
        prov = key.split(":", 1)[0]
        bd = info.get("by_difficulty", {})
        passed = sum(float(bd.get(str(d), {}).get("passed", 0)) for d in range(1, 11))
        count = sum(float(bd.get(str(d), {}).get("count", 0)) for d in range(1, 11))
        t = prov_totals.setdefault(prov, {"passed": 0.0, "count": 0.0})
        t["passed"] += passed
        t["count"] += count
    if prov_totals:
        provs = list(prov_totals.keys())
        vals = [
            (100.0 * prov_totals[p]["passed"] / prov_totals[p]["count"]) if prov_totals[p]["count"] > 0 else 0.0
            for p in provs
        ]
        if sort_bars_desc and provs:
            pv = sorted(zip(provs, vals), key=lambda x: (-x[1], str(x[0]).lower()))
            provs, vals = [p for p, _ in pv], [v for _, v in pv]
        else:
            provs = sorted(provs)
            vals = [
                (100.0 * prov_totals[p]["passed"] / prov_totals[p]["count"]) if prov_totals[p]["count"] > 0 else 0.0
                for p in provs
            ]
        fig_w, fig_h = (10, 6) if len(provs) <= 8 else (12, 7)
        fig = plt.figure(figsize=(fig_w, fig_h))
        ax = fig.add_subplot(1, 1, 1)
        # Use fixed base color per provider
        bars = ax.bar(range(len(provs)), vals, color=[provider_base_color.get(p, (0.5,0.5,0.5)) for p in provs])
        ax.set_xticks(range(len(provs)))
        ax.set_xticklabels([_disp_provider(str(p)) for p in provs], rotation=20, ha="right")
        ax.set_ylabel("% correct")
        ax.set_title(_title("Overall % correct by provider"))
        ax.set_ylim(0, max(100.0, (max(vals) if vals else 0.0) * 1.1))
        ax.grid(True, axis="y", alpha=0.3)
        for rect, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
        bar_path = out_root / f"graph_{slugify(out_root.name)}_providers_overall{file_suffix}.png"
        fig.tight_layout()
        fig.savefig(bar_path, bbox_inches="tight")
        plt.close(fig)
        try:
            import gc as _gc
            _gc.collect()
        except Exception:
            pass
        saved.append(bar_path)

    # Provider/Model overall bar chart (aggregate across reasoning levels)
    prov_model_totals: Dict[str, Dict[str, float]] = {}
    for key, info in items:
        parts = key.split(":")
        prov = parts[0] if parts else key
        model = parts[1] if len(parts) > 1 else ""
        pm_key = f"{prov}:{model}"
        bd = info.get("by_difficulty", {})
        passed = sum(float(bd.get(str(d), {}).get("passed", 0)) for d in range(1, 11))
        count = sum(float(bd.get(str(d), {}).get("count", 0)) for d in range(1, 11))
        t = prov_model_totals.setdefault(pm_key, {"passed": 0.0, "count": 0.0})
        t["passed"] += passed
        t["count"] += count
    if prov_model_totals:
        labels = list(prov_model_totals.keys())
        vals = [
            (100.0 * prov_model_totals[k]["passed"] / prov_model_totals[k]["count"]) if prov_model_totals[k]["count"] > 0 else 0.0
            for k in labels
        ]
        if sort_bars_desc and labels:
            lv = sorted(zip(labels, vals), key=lambda x: (-x[1], str(x[0]).lower()))
            labels, vals = [k for k, _ in lv], [v for _, v in lv]
        else:
            labels = sorted(labels)
            vals = [
                (100.0 * prov_model_totals[k]["passed"] / prov_model_totals[k]["count"]) if prov_model_totals[k]["count"] > 0 else 0.0
                for k in labels
            ]
        # Build display labels and compute provider-aligned colors with model variations
        provs = []
        models = []
        for k in labels:
            p, m = (k.split(":", 1) + [""])[:2]
            provs.append(p)
            models.append(m)
        # Map provider -> list of indices for its models in labels order
        prov_to_indices: Dict[str, List[int]] = {}
        for i, p in enumerate(provs):
            prov_to_indices.setdefault(p, []).append(i)
        # Use fixed mapping for provider+model colors
        colors: List[tuple] = [model_color.get((provs[i], models[i]), provider_base_color.get(provs[i], (0.5,0.5,0.5))) for i in range(len(labels))]
        # Fallback for any None (shouldn't happen)
        for i, c in enumerate(colors):
            if c is None:
                colors[i] = base_cmap(i % base_cmap.N)
        # Plot
        fig_w, fig_h = (12, 7) if len(labels) > 8 else (10, 6)
        fig = plt.figure(figsize=(fig_w, fig_h))
        ax = fig.add_subplot(1, 1, 1)
        bars = ax.bar(range(len(labels)), vals, color=colors, edgecolor='black', linewidth=0.2)
        ax.set_xticks(range(len(labels)))
        # Replace provider prefix in labels for display
        disp_labels = []
        for k in labels:
            parts = k.split(":", 1)
            if len(parts) == 2:
                disp_labels.append(f"{_disp_provider(parts[0])}:{parts[1]}")
            else:
                disp_labels.append(_disp_provider(k))
        ax.set_xticklabels(disp_labels, rotation=35, ha="right")
        ax.set_ylabel("% correct")
        ax.set_title(_title("Overall % correct by provider/model"))
        ax.set_ylim(0, max(100.0, (max(vals) if vals else 0.0) * 1.1))
        ax.grid(True, axis="y", alpha=0.3)
        for rect, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
        pm_path = out_root / f"graph_{slugify(out_root.name)}_providers_models_overall{file_suffix}.png"
        fig.tight_layout()
        fig.savefig(pm_path, bbox_inches="tight")
        plt.close(fig)
        try:
            import gc as _gc
            _gc.collect()
        except Exception:
            pass
        saved.append(pm_path)

    # Provider/Model/Reasoning bars (one bar per triple)
    spec_totals: Dict[str, Dict[str, float]] = {}
    for key, info in items:
        bd = info.get("by_difficulty", {})
        passed = sum(float(bd.get(str(d), {}).get("passed", 0)) for d in range(1, 11))
        count = sum(float(bd.get(str(d), {}).get("count", 0)) for d in range(1, 11))
        spec_totals[key] = {"passed": passed, "count": count}
    if spec_totals:
        def _key_order(k: str):
            parts = k.split(":")
            prov = parts[0] if parts else ""
            model = parts[1] if len(parts) > 1 else ""
            reasoning = parts[2] if len(parts) > 2 else ""
            rord = {"low": 0, "medium": 1, "high": 2}.get(reasoning, 3)
            return (prov, model, rord, reasoning)
        labels = sorted(spec_totals.keys(), key=_key_order)
        vals = [
            (100.0 * spec_totals[k]["passed"] / spec_totals[k]["count"]) if spec_totals[k]["count"] > 0 else 0.0
            for k in labels
        ]
        # Colors from provider/model mapping; add hatch patterns for reasoning levels
        bar_colors: List[tuple] = []
        bar_reasonings: List[str] = []
        for k in labels:
            parts = k.split(":")
            prov = parts[0] if parts else ""
            model = parts[1] if len(parts) > 1 else ""
            reasoning = parts[2] if len(parts) > 2 else ""
            bar_colors.append(model_color.get((prov, model), provider_base_color.get(prov, (0.5, 0.5, 0.5))))
            bar_reasonings.append(reasoning)
        fig_w, fig_h = (max(12, len(labels) * 0.35), 7)
        fig = plt.figure(figsize=(fig_w, fig_h))
        ax = fig.add_subplot(1, 1, 1)
        xs = range(len(labels))
        bars = ax.bar(xs, vals, color=bar_colors, edgecolor='black', linewidth=0.3)
        hatch_map = {"low": '/', "medium": '\\', "high": 'x'}
        for i, b in enumerate(bars):
            b.set_hatch(hatch_map.get(bar_reasonings[i], ''))
        ax.set_xticks(list(xs))
        ax.set_xticklabels([_disp_key(k) for k in labels], rotation=40, ha="right")
        ax.set_ylabel("% correct")
        ax.set_title(_title("Overall % correct by provider/model/reasoning"))
        ax.set_ylim(0, max(100.0, (max(vals) if vals else 0.0) * 1.1))
        ax.grid(True, axis="y", alpha=0.3)
        from matplotlib.patches import Patch
        legend_elems = [Patch(facecolor='white', edgecolor='black', hatch=hatch_map[rl], label=rl) for rl in ("low", "medium", "high")]
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.78, box.height])
        ax.legend(handles=legend_elems, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="reasoning")
        for rect, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        pmr_path = out_root / f"graph_{slugify(out_root.name)}_providers_models_reasoning_overall{file_suffix}.png"
        fig.subplots_adjust(bottom=0.28)
        fig.tight_layout()
        fig.savefig(pmr_path, bbox_inches="tight")
        plt.close(fig)
        try:
            import gc as _gc
            _gc.collect()
        except Exception:
            pass
        saved.append(pmr_path)

    # Per-provider models overall bar charts
    for provider, kv in sorted(by_provider.items()):
        if not kv:
            continue
        models: list[str] = []
        vals: list[float] = []
        for key, info in sorted(kv):
            parts = key.split(":")
            model = parts[1] if len(parts) > 1 else key
            reasoning = parts[2] if len(parts) > 2 else ""
            label = f"{model}:{reasoning}" if reasoning else model
            bd = info.get("by_difficulty", {})
            passed = sum(float(bd.get(str(d), {}).get("passed", 0)) for d in range(1, 11))
            count = sum(float(bd.get(str(d), {}).get("count", 0)) for d in range(1, 11))
            pct = (100.0 * passed / count) if count > 0 else 0.0
            models.append(label)
            vals.append(pct)
        if sort_bars_desc and models:
            mv = sorted(zip(models, vals), key=lambda x: (-x[1], str(x[0]).lower()))
            models, vals = [m for m, _ in mv], [v for _, v in mv]
        if not models:
            continue
        fig_w, fig_h = (10, 6) if len(models) <= 8 else (12, 7)
        fig = plt.figure(figsize=(fig_w, fig_h))
        ax = fig.add_subplot(1, 1, 1)
        # Use fixed mapping by provider+model
        color_by_idx = [model_color.get((provider, m.split(":")[0]), provider_base_color.get(provider, (0.5,0.5,0.5))) for m in models]
        bars = ax.bar(range(len(models)), vals, color=color_by_idx, edgecolor='black', linewidth=0.2)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=25, ha="right")
        ax.set_ylabel("% correct")
        ax.set_title(_title(f"Overall % correct by model — {_disp_provider(provider)}"))
        ax.set_ylim(0, max(100.0, (max(vals) if vals else 0.0) * 1.1))
        ax.grid(True, axis="y", alpha=0.3)
        for rect, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
        img_path = out_root / f"graph_{slugify(out_root.name)}_provider_{slugify(_disp_provider(provider))}_models_overall{file_suffix}.png"
        fig.tight_layout()
        fig.savefig(img_path, bbox_inches="tight")
        plt.close(fig)
        try:
            import gc as _gc
            _gc.collect()
        except Exception:
            pass
        saved.append(img_path)

    # Provider/Model filtered difficulties grouped bar chart (one bar per difficulty per model; model color persistent)
    levels = [int(d) for d in (filtered_difficulties or [1, 10]) if isinstance(d, (int,)) and 1 <= int(d) <= 10]
    levels = sorted(set(levels))
    if levels:
        # Build per provider->model metrics per difficulty (aggregate across reasoning levels)
        # Also compute sorting key based on difficulty 10 (or 0 if none)
        pm_by_provider: Dict[str, Dict[str, Dict[int, float]]] = {}
        pm_counts: Dict[str, Dict[str, Dict[int, float]]] = {}
        for key, info in items:
            parts = key.split(":")
            prov = parts[0] if parts else key
            model = parts[1] if len(parts) > 1 else ""
            dmap = info.get("by_difficulty", {})
            # Merge into provider->model
            P = pm_by_provider.setdefault(prov, {})
            C = pm_counts.setdefault(prov, {})
            mp = P.setdefault(model, {})
            mc = C.setdefault(model, {})
            for d in range(1, 11):
                dm = dmap.get(str(d), {})
                try:
                    cnt = float(dm.get("count", 0) or 0)
                    pas = float(dm.get("passed", 0) or 0)
                except Exception:
                    cnt = 0.0; pas = 0.0
                mp[d] = mp.get(d, 0.0) + pas
                mc[d] = mc.get(d, 0.0) + cnt
        # Compute pass pct per difficulty
        pm_pct: Dict[str, Dict[str, Dict[int, float]]] = {}
        for prov, models in pm_by_provider.items():
            pd = pm_pct.setdefault(prov, {})
            for model, per_d in models.items():
                pd[model] = {}
                for d in levels + [10]:
                    cnt = pm_counts.get(prov, {}).get(model, {}).get(d, 0.0)
                    pas = per_d.get(d, 0.0)
                    pct = (100.0 * pas / cnt) if cnt > 0 else 0.0
                    pd[model][d] = pct
        # Sorting: providers by best model at difficulty 10 (desc); models by their difficulty 10 (desc)
        def prov_sort_key(prov: str) -> float:
            md = pm_pct.get(prov, {})
            best = 0.0
            for m, dd in md.items():
                best = max(best, dd.get(10, 0.0))
            return -best
        prov_order = sorted(pm_pct.keys(), key=prov_sort_key)
        # Compose categories: list of (prov, model)
        categories: List[tuple] = []
        for prov in prov_order:
            md = pm_pct.get(prov, {})
            models_sorted = sorted(md.keys(), key=lambda m: -md.get(m, {}).get(10, 0.0))
            for m in models_sorted:
                categories.append((prov, m))
        if categories:
            # Use fixed mapping for categories
            # Hatches per difficulty to distinguish within same color
            hatches = ['/', '\\', 'x', '.', 'o', '*', '+']
            # Build bars
            fig_w = max(12, int(len(categories) * (0.35 + 0.08 * max(0, len(levels)-2))))
            fig_h = 7
            fig = plt.figure(figsize=(fig_w, fig_h))
            ax = fig.add_subplot(1, 1, 1)
            # x positions
            idxs = list(range(len(categories)))
            group_width = min(0.8, 0.5 + 0.1 * len(levels))
            bar_w = group_width / max(1, len(levels))
            # Plot each difficulty
            for di, d in enumerate(levels):
                xs = [i - group_width/2 + di*bar_w + bar_w/2 for i in idxs]
                vals = []
                cols = []
                for ci, (prov, m) in enumerate(categories):
                    v = pm_pct.get(prov, {}).get(m, {}).get(d, 0.0)
                    vals.append(v)
                    cols.append(model_color.get((prov, m), provider_base_color.get(prov, (0.5,0.5,0.5))))
                bars = ax.bar(xs, vals, width=bar_w, color=cols, hatch=hatches[di % len(hatches)], edgecolor='black', linewidth=0.3)
            # X tick labels: provider:model with renamed provider
            disp_labels = []
            for prov, m in categories:
                disp_labels.append(f"{_disp_provider(prov)}:{m}")
            ax.set_xticks(idxs)
            ax.set_xticklabels(disp_labels, rotation=35, ha="right")
            ax.set_ylabel("% correct")
            lev_str = ", ".join([str(d) for d in levels])
            ax.set_title(_title(f"Pass rates by provider/model — per difficulty ({lev_str})"))
            # Add legend for difficulties (hatches)
            from matplotlib.patches import Patch
            diff_legend = [Patch(facecolor='white', edgecolor='black', hatch=hatches[i % len(hatches)], label=f"d{levels[i]}") for i in range(len(levels))]
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
            ax.legend(handles=diff_legend, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="difficulty")
            # Y limits/annotations
            maxv = 0.0
            for prov, m in categories:
                for d in levels:
                    maxv = max(maxv, pm_pct.get(prov, {}).get(m, {}).get(d, 0.0))
            ax.set_ylim(0, max(100.0, (maxv if categories else 0.0) * 1.1))
            ax.grid(True, axis="y", alpha=0.3)
            # File name
            lev_suffix = "_".join([f"d{d}" for d in levels])
            img_path = out_root / f"graph_{slugify(out_root.name)}_providers_models_levels_grouped_{lev_suffix}{file_suffix}.png"
            fig.tight_layout()
            fig.savefig(img_path, bbox_inches="tight")
            plt.close(fig)
            saved.append(img_path)

    # Zero-incorrect rate by difficulty (per spec_key line chart)
    try:
        results = list(out_root.glob("results_*.jsonl"))
        zero_data: Dict[str, Dict[int, Dict[str, float]]] = {}
        for rp in results:
            # Determine if file should be excluded by checking first valid record
            with rp.open("r", encoding="utf-8") as f:
                first_valid = None
                for line in f:
                    try:
                        o = json.loads(line)
                    except Exception:
                        continue
                    prov = (o.get("provider") or "").strip()
                    model = (o.get("model") or "").strip()
                    rl = (o.get("reasoning_level") or "").strip()
                    spec_key = f"{prov}:{model}:{rl}"
                    first_valid = spec_key
                    break
            if not first_valid or _excluded(first_valid):
                continue
            # Scan file fully
            with rp.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        o = json.loads(line)
                    except Exception:
                        continue
                    prov = (o.get("provider") or "").strip()
                    model = (o.get("model") or "").strip()
                    rl = (o.get("reasoning_level") or "").strip()
                    spec_key = f"{prov}:{model}:{rl}"
                    if _excluded(spec_key):
                        continue
                    d = o.get("difficulty")
                    try:
                        d = int(d)
                    except Exception:
                        continue
                    if not (1 <= d <= 10):
                        continue
                    proposed = o.get("proposed")
                    matched = bool(o.get("matched"))
                    stats = zero_data.setdefault(spec_key, {}).setdefault(d, {"count": 0.0, "zero_incorrect": 0.0})
                    stats["count"] += 1.0
                    is_zero = False
                    try:
                        if proposed is None:
                            is_zero = False
                        else:
                            v = float(proposed)
                            is_zero = abs(v) <= 1e-12
                    except Exception:
                        is_zero = False
                    if is_zero and (not matched):
                        stats["zero_incorrect"] += 1.0
        # Convert to items-like for plotting with existing helper
        zero_items: List[Tuple[str, Dict[str, Any]]] = []
        for key, per_d in zero_data.items():
            bd = {}
            for d in range(1, 11):
                s = per_d.get(d) or {"count": 0.0, "zero_incorrect": 0.0}
                cnt = float(s.get("count", 0.0) or 0.0)
                zi = float(s.get("zero_incorrect", 0.0) or 0.0)
                pct = (100.0 * zi / cnt) if cnt > 0 else 0.0
                bd[str(d)] = {"pass_pct": pct}
            zero_items.append((key, {"by_difficulty": bd}))
        if zero_items:
            img_path = out_root / f"graph_{slugify(out_root.name)}_zero_incorrect_by_difficulty{file_suffix}.png"
            _plot_items(zero_items, "Answer zero and incorrect", img_path)
            saved.append(img_path)
    except Exception:
        pass

    # Missed-by-all line chart (requires missed_by_all.json)
    mba_path = out_root / missed_by_all_filename
    if mba_path.exists() and items:
        try:
            mba = json.loads(mba_path.read_text(encoding="utf-8"))
            missed_counts = {d: 0 for d in range(1, 11)}
            for e in mba:
                try:
                    d = int(e.get("difficulty") or 0)
                except Exception:
                    d = 0
                if 1 <= d <= 10:
                    missed_counts[d] += 1
            denom = {d: 0 for d in range(1, 11)}
            if items:
                _, first = items[0]
                for d in range(1, 11):
                    denom[d] = int(first.get("by_difficulty", {}).get(str(d), {}).get("count", 0))
            xs = list(range(1, 11))
            ys = [((100.0 * missed_counts[d] / denom[d]) if denom[d] else 0.0) for d in xs]
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(xs, ys, marker="o", color="#d62728")
            ax.set_xticks(xs)
            ax.set_xticklabels([f"diff{d}" for d in xs])
            ax.set_ylabel("% unanswered (all models)")
            ax.set_xlabel("Difficulty")
            ax.set_title(_title("Missed by all"))
            ax.grid(True, alpha=0.3)
            mba_img = out_root / f"graph_{slugify(out_root.name)}_missed_by_all{file_suffix}.png"
            fig.tight_layout()
            fig.savefig(mba_img, bbox_inches="tight")
            plt.close(fig)
            saved.append(mba_img)
        except Exception:
            pass

    return saved


# ----------------- Consensus-based regrade utilities -----------------

def _sym_percent_diff(a: float, b: float, eps: float = 1e-12) -> float:
    """Symmetric percent difference between two numbers (0..inf).
    0 if both ~0; else 200*|a-b|/(|a|+|b|) in percent.
    """
    da = abs(float(a)); db = abs(float(b))
    denom = da + db
    if denom < eps:
        return 0.0
    return 200.0 * abs(float(a) - float(b)) / denom


def _load_all_results(run_dir: Path, exclude: Optional[Union[str, List[str]]] = None):
    """Load all results_*.jsonl under run_dir, returning:
    - recs_by_spec: spec_key -> {(d,id): record}
    - meta_by_spec: spec_key -> {provider, model, reasoning_level}
    - questions: {(d,id): question_text}
    """
    results = list(run_dir.glob("results_*.jsonl"))

    def _excluded(spec_key: str, model: str) -> bool:
        if not exclude:
            return False
        patterns: List[str]
        if isinstance(exclude, str):
            patterns = [exclude]
        else:
            patterns = [str(p) for p in (exclude or [])]
        for pat in patterns:
            try:
                rx = re.compile(pat)
            except Exception:
                rx = re.compile(re.escape(pat))
            if rx.search(spec_key) or rx.search(model):
                return True
        return False

    recs_by_spec: Dict[str, Dict[Tuple[int, str], Dict[str, Any]]] = {}
    meta_by_spec: Dict[str, Dict[str, Any]] = {}
    questions: Dict[Tuple[int, str], str] = {}

    for rp in results:
        try:
            with rp.open("r", encoding="utf-8") as f:
                peeked_key = None
                model_name = None
                for line in f:
                    try:
                        o = json.loads(line)
                    except Exception:
                        continue
                    prov = (o.get("provider") or "").strip()
                    model_name = (o.get("model") or "").strip()
                    rl = (o.get("reasoning_level") or "").strip()
                    spec_key = f"{prov}:{model_name}:{rl}"
                    if peeked_key is None:
                        if _excluded(spec_key, model_name):
                            break
                        peeked_key = spec_key
                        meta_by_spec[spec_key] = {"provider": prov, "model": model_name, "reasoning_level": rl}
                        recs_by_spec.setdefault(spec_key, {})
                    qid = o.get("id")
                    d = o.get("difficulty")
                    if qid is None or not isinstance(d, int):
                        continue
                    key = (int(d), str(qid))
                    recs_by_spec[spec_key][key] = o
                    if "question" in o and o.get("question") and key not in questions:
                        try:
                            questions[key] = str(o.get("question"))
                        except Exception:
                            pass
        except Exception:
            continue
    return recs_by_spec, meta_by_spec, questions


def _build_consensus_key(
    recs_by_spec: Dict[str, Dict[Tuple[int, str], Dict[str, Any]]],
    consensus_pct: float,
    min_votes: int,
):
    """Compute consensus per (difficulty,id). Returns:
    - consensus_map: {(d,id): {"consensus": value, "votes": int, "members": [(spec_key, proposed), ...]}}
    """
    # Gather proposals by problem
    by_problem: Dict[Tuple[int, str], List[Tuple[str, float]]] = {}
    for spec_key, recs in recs_by_spec.items():
        for key, r in recs.items():
            proposed = r.get("proposed")
            try:
                if proposed is None:
                    continue
                val = float(proposed)
            except Exception:
                continue
            by_problem.setdefault(key, []).append((spec_key, val))

    consensus_map: Dict[Tuple[int, str], Dict[str, Any]] = {}
    thr = float(consensus_pct)
    for key, pairs in by_problem.items():
        if not pairs:
            continue
        vals = [v for _, v in pairs]
        # Build clusters by symmetric percent diff
        clusters: List[List[int]] = []  # indices into pairs
        for i, vi in enumerate(vals):
            found = False
            for cl in clusters:
                # Check closeness to cluster centroid (use median of cluster)
                cv = sorted([vals[j] for j in cl])
                centroid = cv[len(cv)//2]
                if _sym_percent_diff(vi, centroid) <= thr:
                    cl.append(i)
                    found = True
                    break
            if not found:
                clusters.append([i])
        # Expand clusters by pairwise inclusion to avoid centroid sensitivity
        # (simple refinement pass)
        refined: List[List[int]] = []
        for cl in clusters:
            acc = list(cl)
            changed = True
            while changed:
                changed = False
                for i in range(len(vals)):
                    if i in acc:
                        continue
                    if any(_sym_percent_diff(vals[i], vals[j]) <= thr for j in acc):
                        acc.append(i)
                        changed = True
            refined.append(sorted(set(acc)))
        # Pick the largest cluster
        refined.sort(key=lambda c: (-len(c), c))
        top = refined[0] if refined else []
        if len(top) >= int(min_votes):
            members = [(pairs[i][0], vals[i]) for i in top]
            # consensus value: median of member values
            mvals = sorted([vals[i] for i in top])
            consensus_val = mvals[len(mvals)//2]
            consensus_map[key] = {"consensus": consensus_val, "votes": len(top), "members": members}
    return consensus_map


def _write_consensus_files(run_dir: Path, consensus_map: Dict[Tuple[int, str], Dict[str, Any]]) -> None:
    by_diff: Dict[int, List[Dict[str, Any]]] = {}
    for (d, qid), info in consensus_map.items():
        by_diff.setdefault(int(d), []).append({
            "id": str(qid),
            "consensus": info.get("consensus"),
            "votes": int(info.get("votes") or 0),
        })
    # Write per-difficulty files
    for d, arr in by_diff.items():
        path = run_dir / f"consensus_diff{int(d)}.json"
        path.write_text(json.dumps(arr, ensure_ascii=False, indent=2), encoding="utf-8")
    # Write a combined file
    all_arr: List[Dict[str, Any]] = []
    for d in sorted(by_diff.keys()):
        for it in by_diff[d]:
            all_arr.append({"difficulty": int(d), **it})
    (run_dir / "consensus_key.json").write_text(json.dumps(all_arr, ensure_ascii=False, indent=2), encoding="utf-8")


def _regrade_with_consensus(
    recs_by_spec: Dict[str, Dict[Tuple[int, str], Dict[str, Any]]],
    meta_by_spec: Dict[str, Dict[str, Any]],
    consensus_map: Dict[Tuple[int, str], Dict[str, Any]],
    consensus_error_pct: float,
    fallback_mode: str,
) -> Dict[str, Dict[str, Any]]:
    """Build a summary dict like summary_all.json but graded vs consensus.
    Returns mapping spec_key -> summary entry.
    """
    thresh = float(consensus_error_pct)
    # Pre-init summaries
    summary: Dict[str, Dict[str, Any]] = {}
    for spec_key, meta in meta_by_spec.items():
        summary[spec_key] = {
            "provider": meta.get("provider"),
            "model": meta.get("model"),
            "reasoning_level": meta.get("reasoning_level"),
            "by_difficulty": {},
        }
    # For each spec and difficulty, compute pass counts only on items with consensus
    for spec_key, recs in recs_by_spec.items():
        counts: Dict[int, Dict[str, float]] = {}
        # Build the set of problems to consider for this spec
        # If fallback_mode == 'exclude': only consensus problems
        # Else include all problems this spec attempted
        if str(fallback_mode).lower() == "exclude":
            keys_to_check = list(consensus_map.keys())
        else:
            keys_to_check = list(recs.keys())
        for (d, qid) in keys_to_check:
            d_int = int(d)
            e = counts.setdefault(d_int, {"count": 0, "passed": 0})
            r = recs.get((d_int, qid))
            if not r:
                # Not attempted by this spec; do not count
                continue
            # Determine target value: consensus if present; else ground truth if allowed
            target = None
            if (d_int, qid) in consensus_map:
                try:
                    target = float(consensus_map[(d_int, qid)]["consensus"])  # type: ignore
                except Exception:
                    target = None
            if target is None and str(fallback_mode).lower() != "exclude":
                try:
                    target = float(r.get("actual")) if r.get("actual") is not None else None
                except Exception:
                    target = None
            if target is None:
                # No target available; skip this problem
                continue
            # Increment count; treat missing/invalid proposed as incorrect
            proposed = r.get("proposed")
            try:
                pv = float(proposed)
            except Exception:
                pv = None
            e["count"] += 1
            if pv is None:
                continue
            err_pct = _percent_error(target, pv)
            if err_pct <= thresh:
                e["passed"] += 1
        # Emit by_difficulty
        for d in range(1, 11):
            c = counts.get(d, None)
            if not c:
                summary[spec_key]["by_difficulty"][str(d)] = {"count": 0, "passed": 0, "pass_pct": 0.0}
            else:
                cnt = int(c["count"]) or 0
                pas = int(c["passed"]) or 0
                pct = (100.0 * pas / cnt) if cnt else 0.0
                summary[spec_key]["by_difficulty"][str(d)] = {"count": cnt, "passed": pas, "pass_pct": pct}
    return summary


def _build_missed_by_all_consensus(
    recs_by_spec: Dict[str, Dict[Tuple[int, str], Dict[str, Any]]],
    consensus_map: Dict[Tuple[int, str], Dict[str, Any]],
    consensus_error_pct: float,
    fallback_mode: str,
) -> List[Dict[str, Any]]:
    """Items where all specs attempted and none matched consensus."""
    out: List[Dict[str, Any]] = []
    # Collect all spec keys
    spec_keys = sorted(recs_by_spec.keys())
    # Build the set of problems to evaluate
    if str(fallback_mode).lower() == "exclude":
        problem_keys = list(consensus_map.keys())
    else:
        # Union of all problems across specs
        u: set = set()
        for recs in recs_by_spec.values():
            u.update(recs.keys())
        problem_keys = list(u)
    for (d, qid) in problem_keys:
        all_attempted = True
        all_failed = True
        answers_by_model: Dict[str, Any] = {}
        for spec_key in spec_keys:
            rec = recs_by_spec[spec_key].get((d, qid))
            if rec is None:
                all_attempted = False
                all_failed = False
                break
            proposed = rec.get("proposed")
            answers_by_model[spec_key] = proposed
            try:
                pv = float(proposed)
            except Exception:
                pv = None
            if pv is None:
                continue
            # Determine target: consensus or ground truth fallback
            target = None
            if (d, qid) in consensus_map:
                try:
                    target = float(consensus_map[(d, qid)]["consensus"])  # type: ignore
                except Exception:
                    target = None
            if target is None and str(fallback_mode).lower() != "exclude":
                try:
                    target = float(rec.get("actual")) if rec.get("actual") is not None else None
                except Exception:
                    target = None
            if target is None:
                all_attempted = False
                all_failed = False
                break
            if _percent_error(target, pv) <= float(consensus_error_pct):
                all_failed = False
        if all_attempted and all_failed:
            out.append({
                "id": str(qid),
                "difficulty": int(d),
                "actual": float(consensus_map[(d, qid)]['consensus']) if (d, qid) in consensus_map else None,
                "answers_by_model": answers_by_model,
            })
    return out


def _consensus_regrade_and_graphs(
    run_dir: Path,
    exclude: Optional[Union[str, List[str]]],
    sort_bars_desc: bool,
    provider_renames: Optional[Dict[str, str]],
    consensus_pct: float,
    consensus_min_votes: int,
    consensus_error_pct: float,
    fallback_mode: str,
    filtered_difficulties: Optional[List[int]] = None,
    omit_run_name_in_titles: bool = False,
    show_grading_type_in_titles: bool = False,
    randomize_provider_line_colors: bool = False,
    dataset_base_name: Optional[str] = None,
) -> None:
    recs_by_spec, meta_by_spec, _questions = _load_all_results(run_dir, exclude=exclude)
    consensus_map = _build_consensus_key(recs_by_spec, consensus_pct=consensus_pct, min_votes=consensus_min_votes)
    # Write consensus key files
    _write_consensus_files(run_dir, consensus_map)
    # Build consensus-based summary
    summary_consensus = _regrade_with_consensus(
        recs_by_spec,
        meta_by_spec,
        consensus_map,
        consensus_error_pct,
        fallback_mode,
    )
    (run_dir / "summary_all_consensus.json").write_text(json.dumps(summary_consensus, ensure_ascii=False, indent=2), encoding="utf-8")
    # Build missed-by-all consensus file
    try:
        mba_c = _build_missed_by_all_consensus(recs_by_spec, consensus_map, consensus_error_pct, fallback_mode)
        (run_dir / "missed_by_all_consensus.json").write_text(json.dumps(_sanitize_for_json(mba_c), ensure_ascii=False, indent=2), encoding="utf-8")
        # Enriched consensus variant with dataset fields
        try:
            mba_c_det = _enrich_entries_with_dataset_fields(mba_c, dataset_base_name=dataset_base_name)
            (run_dir / "missed_by_all_consensus_detailed.json").write_text(json.dumps(_sanitize_for_json(mba_c_det), ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[WARN] Could not generate missed_by_all_consensus_detailed.json: {e}")
    except Exception:
        pass
    # Generate consensus graphs with _consensusGrade suffix
    regenerate_graphs_for_run(
        run_dir,
        exclude=exclude,
        sort_bars_desc=sort_bars_desc,
        provider_renames=provider_renames,
        summary_filename="summary_all_consensus.json",
        file_suffix="_consensusGrade",
        missed_by_all_filename="missed_by_all_consensus.json",
        filtered_difficulties=filtered_difficulties,
        omit_run_name_in_titles=omit_run_name_in_titles,
        show_grading_type_in_titles=show_grading_type_in_titles,
        randomize_provider_line_colors=randomize_provider_line_colors,
    )
