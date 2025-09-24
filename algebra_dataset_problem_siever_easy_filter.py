#!/usr/bin/env python3
"""
Algebra Dataset Easy‑Filter Siever (AGSM Stage 2)
=================================================

Purpose (Stage 2 of the two‑stage sieve)
----------------------------------------
Create a culled dataset per difficulty by removing problems that are "too easy"
for a chosen panel. By default, it removes problems that EVERY selected provider/model
can solve; alternatively, cull when at least a threshold fraction of providers solve
the problem (e.g., 50% for a “weak panel” in AGSM).

Behavior
--------
- Iterates difficulties 2..10 for a given base name (e.g., AGSM8K-V2).
- Reads the regular file: algebraTest/<BASE>_diff{d}.json
- Checks the first N problems per difficulty (default 500).
- Uses the same tolerant parsing and grading approach as the benchmark:
  - Strip <think> blocks, parse JSON or JSON fragments to extract numeric
    final_answer, then compare to deterministic_answer with percent error
    tolerance (default 0.5%).
- Temperature is locked at 0.7 for all providers in this siever.
- Retries like the benchmark (up to 3 attempts: initial, +30s, +90s).
  - If a provider errors or times out (or returns empty) after retries,
    this counts as "solved" by that provider for this run (conservative cull).
- A problem is culled if and only if every selected provider counts as
  "solved" for that problem; otherwise it's kept.
- Or, when using `--successThreshold PCT`, cull when the percent of
  selected providers that solve the problem is at least PCT (default 100).
- Outputs for each difficulty d:
  - Regular: algebraTest/<BASE>_EasyCulled_diff{d}.json
  - Lite:    algebraTest/<BASE>_EasyCulled_diff{d}_lite.json
  - Summary: algebraTest/<BASE>_EasyCulled_diff{d}_summary.json

Special handling for difficulty 1
---------------------------------
Difficulty 1 problems have already been pre-screened and should not be
sieved again. The siever therefore pass-through copies diff1 to the
EasyCulled naming convention without calling any providers:

  - Copies algebraTest/<BASE>_diff1.json to
    algebraTest/<BASE>_EasyCulled_diff1.json (optionally sliced by
    --max-items), and writes a corresponding *_lite.json via projection.
  - Emits a *_summary.json with {"pass_through": true} and culled_count=0.
  - Existing outputs are preserved unless --overwrite is provided.

CLI
---
python3 algebra_dataset_problem_siever_easy_filter.py \
  --base-name AGSM8K-V2 \
  --providers-file externalProvidersAndModelsV3.json \
  --max-items 500 \
  --error-pct 0.5 \
  --max-tokens 31000 \
  --request-timeout 300 \
  --select "^(together|openai):" \
  --successThreshold 80 \
  --stopAtNumItems 100

Notes
-----
- Providers config uses the same schema as `algebra_dataset_benchmarking_tool`.
- The dataset files must be present under the dataset directory (default
  algebraTest/) with names <BASE>_diff{d}.json.
- AGSM alignment: to reproduce the paper’s weak‑panel sieve, select your weaker
  models via `--select` and use `--successThreshold 50` to cull items that at least
  half of the weak panel solve easily.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed


# Import adapters and helpers from the benchmarking tool to ensure consistent
# provider behavior, parsing, and grading.
from algebra_dataset_benchmarking_tool import (  # type: ignore
    ModelSpec,
    load_providers_config,
    make_adapter,
    _extract_think_blocks,
    _extract_json_and_number,
    _percent_error,
)


def _log(msg: str) -> None:
    try:
        from tqdm import tqdm  # type: ignore

        tqdm.write(str(msg))
    except Exception:
        print(str(msg))


@dataclass
class EasyFilterConfig:
    base_name: str
    providers_file: Path
    dataset_dir: Path
    select_regex: Optional[str]
    max_items: int
    error_pct: float
    max_tokens: int
    request_timeout: int
    out_suffix: str = "EasyCulled"
    start_diff: int = 2
    end_diff: int = 10
    stop_at_num_items: int = 0  # 0 = disabled
    success_threshold_pct: float = 100.0  # Cull when solved ratio >= this percent
    max_workers: int = 8  # Parallel provider requests per item
    # Pass-through handling for diff1
    copy_diff1: bool = True
    overwrite: bool = False


def _read_regular_dataset(dataset_dir: Path, base_name: str, diff: int) -> List[Dict[str, Any]]:
    p = dataset_dir / f"{base_name}_diff{diff}.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing dataset file: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON: {p}: {e}")
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {p}")
    return data


def _project_lite(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in records:
        out.append(
            {
                "id": r.get("id"),
                "question": r.get("question"),
                "deterministic_answer": r.get("deterministic_answer"),
                "sub_components": r.get("sub_components"),
            }
        )
    return out


def _maybe_filter_specs(specs: List[ModelSpec], select_regex: Optional[str]) -> List[ModelSpec]:
    if not select_regex:
        return specs
    rx = re.compile(select_regex)
    out: List[ModelSpec] = []
    for s in specs:
        key = f"{s.provider}:{s.model}:{s.reasoning_level}"
        if rx.search(key):
            out.append(s)
    return out


def _evaluate_one_provider(
    adapter,
    question: str,
    actual_value: float,
    max_tokens: int,
    request_timeout: int,
    error_pct: float,
) -> bool:
    """Return True if this provider counts as "solved" for the problem.

    Retry semantics: up to 3 attempts (initial, +30s, +90s). On any error or
    empty response after exhausting retries, counts as solved (per spec).
    """
    temperature = 0.7  # locked per spec
    attempts = 0
    last_text: str = ""
    last_exc: Optional[Exception] = None
    while attempts < 3:
        try:
            last_text = adapter.complete(question, max_tokens=int(max_tokens), temperature=float(temperature))
            if not (isinstance(last_text, str) and last_text.strip()):
                raise RuntimeError("empty_response")
            break
        except Exception as e:
            last_exc = e
            attempts += 1
            if attempts == 1:
                time.sleep(30)
            elif attempts == 2:
                time.sleep(90)
            else:
                last_text = ""
                break

    if not last_text:
        # Treat failures/timeouts/empty as "solved" in this instance
        return True

    # Parse and grade like the benchmark does
    try:
        think, stripped = _extract_think_blocks(last_text)
        obj, proposed, mode = _extract_json_and_number(stripped or last_text)
        if proposed is None:
            return False
        err_pct = _percent_error(float(actual_value), float(proposed))
        return bool(err_pct <= float(error_pct))
    except Exception:
        # Conservative: if parsing fails here, count as not solved (so we keep it)
        return False


def _providers_solve_counts(
    specs: List[ModelSpec],
    question: str,
    actual_value: float,
    max_tokens: int,
    request_timeout: int,
    error_pct: float,
    _adapter_cache: Dict[str, Any],
) -> Tuple[int, int]:
    """Evaluate providers and return (solved_count, total_count)."""
    solved = 0
    total = len(specs)
    for s in specs:
        key = f"{s.provider}:{s.model}:{s.reasoning_level}:{request_timeout}"
        ad = _adapter_cache.get(key)
        if ad is None:
            ad = make_adapter(s, timeout=int(request_timeout))
            _adapter_cache[key] = ad
        ok = _evaluate_one_provider(ad, question, actual_value, max_tokens, request_timeout, error_pct)
        if ok:
            solved += 1
    return solved, total


def run_easy_filter(cfg: EasyFilterConfig) -> int:
    # Load and optionally filter provider specs
    specs = load_providers_config(cfg.providers_file)
    specs = _maybe_filter_specs(specs, cfg.select_regex)
    if not specs:
        _log("No providers selected after applying filters; nothing to do.")
        return 2

    # Always handle diff1 as pass-through copy if requested
    if cfg.copy_diff1:
        _base = cfg.dataset_dir / f"{cfg.base_name}_diff1.json"
        if _base.exists():
            try:
                _diff1_passthrough(cfg)
            except Exception as e:
                _log(f"[WARN] diff1 pass-through failed: {e}")
        else:
            _log(f"[WARN] Missing dataset file: {_base} (skipping diff1 pass-through)")

    eff_start = max(2, int(cfg.start_diff))
    _log(f"Using {len(specs)} provider spec(s). Base={cfg.base_name} Diffs={eff_start}..{cfg.end_diff} MaxItems={cfg.max_items}")
    adapter_cache: Dict[str, Any] = {}

    # Never sieve diff1; start from max(2, start_diff)
    for d in range(eff_start, int(cfg.end_diff) + 1):
        # Load base dataset for this difficulty
        try:
            items = _read_regular_dataset(cfg.dataset_dir, cfg.base_name, d)
        except FileNotFoundError as e:
            _log(f"[WARN] {e}")
            continue
        except Exception as e:
            _log(f"[ERROR] Failed reading diff {d}: {e}")
            continue

        # Consider only the first N items
        sub = items[: int(cfg.max_items)]
        kept: List[Dict[str, Any]] = []
        culled_count = 0
        processed_count = 0
        # Iterate and test with progress bars (overall + per provider)
        try:
            from tqdm import tqdm  # type: ignore

            bar = tqdm(total=len(sub), desc=f"EasyFilter d={d}", unit="q", position=0)
            provider_bars: Dict[str, Any] = {}
            pos = 1
            for s in specs:
                label = f"{s.provider}:{s.model}:{s.reasoning_level}"
                provider_bars[label] = tqdm(total=len(sub), desc=label, unit="q", position=pos, leave=False)
                pos += 1
            kept_bar = None
            if int(cfg.stop_at_num_items) > 0:
                kept_bar = tqdm(total=int(cfg.stop_at_num_items), desc=f"Kept target d={d}", unit="keep", position=pos, leave=False)
                pos += 1
        except Exception:
            bar = None  # type: ignore
            provider_bars = {}
            kept_bar = None

        stopped_early = False
        for rec in sub:
            q = str(rec.get("question") or "").strip()
            if not q:
                # If no question text, keep (so it gets reviewed)
                kept.append(rec)
                processed_count += 1
                if bar:
                    bar.update(1)
                # advance kept progress if enabled
                try:
                    if kept_bar is not None:
                        kept_bar.update(1)
                except Exception:
                    pass
                # advance provider bars so totals align
                for _pb in provider_bars.values():
                    try:
                        _pb.update(1)
                    except Exception:
                        pass
                continue
            try:
                actual = float(rec.get("deterministic_answer"))
            except Exception:
                # If malformed actual, keep it for later manual inspection
                kept.append(rec)
                processed_count += 1
                if bar:
                    bar.update(1)
                try:
                    if kept_bar is not None:
                        kept_bar.update(1)
                except Exception:
                    pass
                for _pb in provider_bars.values():
                    try:
                        _pb.update(1)
                    except Exception:
                        pass
                continue

            # Compute provider success ratio and apply threshold with early short-circuit
            solved_count = 0
            total_count = len(specs)
            thr = max(0.0, min(100.0, float(cfg.success_threshold_pct)))
            required = math.ceil((thr / 100.0) * float(total_count))

            # Trivial case: threshold == 0 => always cull without requests
            if required == 0:
                # Update provider bars to reflect skipped evaluation for this item
                for pb in provider_bars.values():
                    try:
                        pb.update(1)
                    except Exception:
                        pass
                culled_count += 1
                processed_count += 1
                if bar:
                    bar.update(1)
                # Check early stop target on kept list (unchanged)
                if cfg.stop_at_num_items and len(kept) >= int(cfg.stop_at_num_items):
                    stopped_early = True
                    _log(
                        f"Reached stopAtNumItems={cfg.stop_at_num_items} for diff {d}; processed={processed_count}, kept={len(kept)}"
                    )
                    break
                continue

            # Helper to run a single provider
            def _eval_for_spec(s: ModelSpec) -> Tuple[str, bool]:
                key = f"{s.provider}:{s.model}:{s.reasoning_level}:{cfg.request_timeout}"
                ad = adapter_cache.get(key)
                if ad is None:
                    ad = make_adapter(s, timeout=int(cfg.request_timeout))
                    adapter_cache[key] = ad
                ok = _evaluate_one_provider(ad, q, actual, cfg.max_tokens, cfg.request_timeout, cfg.error_pct)
                return f"{s.provider}:{s.model}:{s.reasoning_level}", ok

            # Submit up to max_workers and short-circuit when decision is clear
            labels_all = [f"{s.provider}:{s.model}:{s.reasoning_level}" for s in specs]
            done_labels: set[str] = set()
            decision_cull = None  # None/True/False
            maxw = max(1, min(int(cfg.max_workers), total_count))
            idx = 0
            submitted: Dict[str, Any] = {}
            with ThreadPoolExecutor(max_workers=maxw) as ex:
                # Initial fill
                while idx < total_count and len(submitted) < maxw:
                    s = specs[idx]
                    lab = labels_all[idx]
                    submitted[lab] = ex.submit(_eval_for_spec, s)
                    idx += 1
                # Process as they complete
                while submitted:
                    for fu in as_completed(list(submitted.values())):
                        # Map back to label
                        lab = None
                        for k, v in list(submitted.items()):
                            if v is fu:
                                lab = k
                                break
                        try:
                            lab_res, ok = fu.result()
                        except Exception:
                            lab_res, ok = (lab or "<task_error>", True)
                        # Update provider bar for this provider
                        if lab_res in provider_bars:
                            try:
                                provider_bars[lab_res].update(1)
                            except Exception:
                                pass
                        if ok:
                            solved_count += 1
                        done_labels.add(lab_res)
                        # Remove from submitted
                        if lab is not None and lab in submitted:
                            submitted.pop(lab, None)

                        # Early decision check
                        processed_providers = len(done_labels)
                        remaining_possible = total_count - processed_providers
                        if solved_count >= required:
                            decision_cull = True
                        elif solved_count + remaining_possible < required:
                            decision_cull = False

                        # Maintain concurrency if continuing
                        if decision_cull is None and idx < total_count:
                            s = specs[idx]
                            lab_next = labels_all[idx]
                            submitted[lab_next] = ex.submit(_eval_for_spec, s)
                            idx += 1

                        if decision_cull is not None:
                            # Try to cancel remaining tasks (best-effort)
                            for v in submitted.values():
                                try:
                                    v.cancel()
                                except Exception:
                                    pass
                            submitted.clear()
                            break
                    # Break outer while if decision made
                    if decision_cull is not None:
                        break

            # For providers not processed due to short-circuit, advance their bars to keep totals aligned
            for lab in labels_all:
                if lab not in done_labels and lab in provider_bars:
                    try:
                        provider_bars[lab].update(1)
                    except Exception:
                        pass

            # Apply decision
            if decision_cull is True:
                culled_count += 1
            elif decision_cull is False:
                kept.append(rec)
                try:
                    if kept_bar is not None:
                        kept_bar.update(1)
                except Exception:
                    pass
            else:
                # No early decision (all processed naturally): compare against required
                if solved_count >= required:
                    culled_count += 1
                else:
                    kept.append(rec)
                    try:
                        if kept_bar is not None:
                            kept_bar.update(1)
                    except Exception:
                        pass
            processed_count += 1
            if bar:
                bar.update(1)
            # Stop when we have reached the requested number of kept items
            if cfg.stop_at_num_items and len(kept) >= int(cfg.stop_at_num_items):
                stopped_early = True
                _log(
                    f"Reached stopAtNumItems={cfg.stop_at_num_items} for diff {d}; processed={processed_count}, kept={len(kept)}"
                )
                break
        if bar:
            bar.close()
        for pb in provider_bars.values():
            try:
                pb.close()
            except Exception:
                pass
        try:
            if kept_bar is not None:
                kept_bar.close()
        except Exception:
            pass

        kept_count = len(kept)
        checked_count = processed_count

        # Write outputs for this difficulty
        out_base = f"{cfg.base_name}_{cfg.out_suffix}_diff{d}"
        out_regular = cfg.dataset_dir / f"{out_base}.json"
        out_lite = cfg.dataset_dir / f"{out_base}_lite.json"
        out_summary = cfg.dataset_dir / f"{out_base}_summary.json"
        try:
            out_regular.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
            out_lite.write_text(json.dumps(_project_lite(kept), ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            _log(f"[ERROR] Failed writing outputs for diff {d}: {e}")
            continue
        summary = {
            "base_name": cfg.base_name,
            "difficulty": d,
            "checked_count": checked_count,
            "culled_count": culled_count,
            "kept_count": kept_count,
            "stopped_early": stopped_early,
            "providers_used": [f"{s.provider}:{s.model}:{s.reasoning_level}" for s in specs],
            "params": {
                "max_items": cfg.max_items,
                "error_pct": cfg.error_pct,
                "max_tokens": cfg.max_tokens,
                "request_timeout": cfg.request_timeout,
                "temperature": 0.7,
                "stopAtNumItems": cfg.stop_at_num_items,
                "successThreshold": cfg.success_threshold_pct,
            },
        }
        try:
            out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            _log(f"[WARN] Failed writing summary for diff {d}: {e}")
        _log(
            f"diff {d}: checked={checked_count} kept={kept_count} culled={culled_count} -> {out_regular.name}, {out_lite.name}"
        )

    return 0


def _diff1_passthrough(cfg: EasyFilterConfig) -> None:
    """Copy diff1 dataset to EasyCulled outputs without sieving.

    - Reads algebraTest/<BASE>_diff1.json.
    - Writes EasyCulled_diff1.json and _lite.json (projected), optionally
      respecting --max-items by slicing the input list.
    - Writes a summary JSON with pass_through=true and culled_count=0.
    - Skips writing if outputs exist and cfg.overwrite is False.
    """
    src = cfg.dataset_dir / f"{cfg.base_name}_diff1.json"
    if not src.exists():
        raise FileNotFoundError(f"Missing dataset file: {src}")

    out_base = f"{cfg.base_name}_{cfg.out_suffix}_diff1"
    out_regular = cfg.dataset_dir / f"{out_base}.json"
    out_lite = cfg.dataset_dir / f"{out_base}_lite.json"
    out_summary = cfg.dataset_dir / f"{out_base}_summary.json"

    if (out_regular.exists() or out_lite.exists() or out_summary.exists()) and not cfg.overwrite:
        _log(f"[INFO] diff1 pass-through outputs already exist (use --overwrite to replace): {out_regular.name}")
        return

    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON: {src}: {e}")
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {src}")

    subset = data[: int(cfg.max_items)] if int(cfg.max_items) > 0 else data
    kept = subset
    checked_count = len(subset)
    kept_count = len(kept)
    culled_count = 0

    # Write outputs
    out_regular.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
    out_lite.write_text(json.dumps(_project_lite(kept), ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "base_name": cfg.base_name,
        "difficulty": 1,
        "checked_count": checked_count,
        "culled_count": culled_count,
        "kept_count": kept_count,
        "stopped_early": False,
        "providers_used": [],
        "pass_through": True,
        "copied_from": str(src.name),
        "params": {
            "max_items": cfg.max_items,
            "error_pct": cfg.error_pct,
            "max_tokens": cfg.max_tokens,
            "request_timeout": cfg.request_timeout,
            "temperature": 0.7,
            "stopAtNumItems": cfg.stop_at_num_items,
            "successThreshold": cfg.success_threshold_pct,
        },
    }
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"diff 1: pass-through copied -> {out_regular.name}, {out_lite.name}")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Cull problems solved by every provider/model; keep only challenging ones.")
    ap.add_argument("--base-name", required=True, help="Base dataset name, e.g., AGSM8K or AGSM8K-V2")
    ap.add_argument("--providers-file", required=True, help="Providers/models JSON file (same schema as benchmark)")
    ap.add_argument("--dataset-dir", default="algebraTest", help="Directory containing <BASE>_diff{d}.json files")
    ap.add_argument("--select", default=None, help="Regex to filter provider:model:reasoning (optional)")
    ap.add_argument("--max-items", type=int, default=500, help="Max problems per difficulty to check (default 500)")
    ap.add_argument("--error-pct", type=float, default=0.5, help="Percent error tolerance on final answer (default 0.5)")
    ap.add_argument("--max-tokens", type=int, default=31000, help="Max completion tokens per call (default 31000)")
    ap.add_argument("--request-timeout", type=int, default=300, help="Per-request timeout seconds (default 300)")
    ap.add_argument("--start-diff", type=int, default=2, help="Start difficulty to sieve (default 2; diff1 is always pass-through)")
    ap.add_argument("--end-diff", type=int, default=10, help="End difficulty inclusive (default 10)")
    ap.add_argument("--suffix", default="EasyCulled", help="Suffix for new base outputs (default EasyCulled)")
    ap.add_argument("--stopAtNumItems", type=int, default=0, help="Stop current difficulty once at least N kept items are found; 0=disabled")
    ap.add_argument(
        "--successThreshold",
        type=float,
        default=100.0,
        help=(
            "Cull an item if the fraction of models that solve it is at least this percent (default 100). "
            "Example: 80 means cull when >=80%% of selected models solve it."
        ),
    )
    ap.add_argument("--max-workers", type=int, default=8, help="Max parallel provider requests per item (default 8)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing EasyCulled outputs if present")
    ap.add_argument("--no-copy-diff1", action="store_true", help="Do not pass-through copy diff1 outputs")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = EasyFilterConfig(
        base_name=args.base_name,
        providers_file=Path(args.providers_file),
        dataset_dir=Path(args.dataset_dir),
        select_regex=args.select,
        max_items=int(args.max_items),
        error_pct=float(args.error_pct),
        max_tokens=int(args.max_tokens),
        request_timeout=int(args.request_timeout),
        out_suffix=str(args.suffix),
        start_diff=int(args.start_diff),
        end_diff=int(args.end_diff),
        stop_at_num_items=int(args.stopAtNumItems),
        success_threshold_pct=float(args.successThreshold),
        max_workers=int(args.max_workers),
        overwrite=bool(args.overwrite),
        copy_diff1=not bool(args.no_copy_diff1),
    )
    try:
        return run_easy_filter(cfg)
    except KeyboardInterrupt:
        _log("Interrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
