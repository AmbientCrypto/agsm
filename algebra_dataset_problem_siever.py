#!/usr/bin/env python3
"""
Algebra Dataset Problem Siever (AGSM Stage 1)
============================================

Goal (Stage 1 of the two‑stage sieve)
-------------------------------------
Filter a source dataset (JSON array or JSONL) into two files based on
tractability across a set of providers/models, and emit a CSV summary:

- <input>_tractable_<N>.json     — problems solvable by at least one model (N = count)
- <input>_intractable_<M>.json   — problems that no model solved (M = count)
- <input>_sieve_summary.csv      — row per problem with solver provider/model and attempt (if tractable)

For each problem, providers/models are tried sequentially in the order given
by a providers JSON (same schema as the benchmarking tool). As soon as a
provider/model returns a correct answer (within percent‑error tolerance), the
problem is marked tractable and evaluation stops for that item.

In AGSM, this stage ensures that candidate leaf problems are explicitly solvable by
at least one model (beyond round‑trip parse checks). These tractable items can then
be composed into multi‑step problems by the deterministic composite builder.

Input expectations
------------------
- Each record includes:
  - problem_text (string)
  - solution_eval (numeric solutions). This may be one of:
    - {"x": 3.0} (single var)
    - {"x": [3.0]} (single-var list)
    - {"x": 4.0, "y": 6.0} (multi-var)
    - {"x": [4.0], "y": [6.0]} (multi-var lists)
    - [4.0, 6.0] (multi-var list; variable order inferred from `variables` field)

Grading rules
-------------
- Multi-variable: all variables must be correct.
- Multi-root: if solution_eval[var] is a list of numeric values, any one match is acceptable for that variable.
- Tolerance: percent error (absolute percentage) per component; default 0.5%% (use --error-pct to override).

Prompt contract
---------------
- Single-variable: return {"final_answer": <number>}
- Multi-variable: return {"answers": {"x": <number>, "y": <number>, ...}}

Parallelism
-----------
- Problems are processed in parallel (configurable with --jobs). Within a problem, providers/models are tried sequentially.

Examples
--------
1) Basic sieving over all providers/models:
   python3 algebra_dataset_problem_siever.py \
     --input algebraTest/quickTestAmbientV6a.json \
     --providers-file externalProvidersAndModelsV3.json \
     --error-pct 0.5 --max-tokens 30000 --request-timeout 300 --jobs 8

2) Narrow to Together + OpenAI only:
   python3 algebra_dataset_problem_siever.py \
     --input algebraTest/quickTestAmbientV6a.json \
     --providers-file externalProvidersAndModelsV3.json \
     --select '^(together|openai):' --jobs 4

Notes
-----
- Requires the adapters used by `algebra_dataset_benchmarking_tool.py`.
- Implements simple backoff on provider errors/empty responses: up to 3 attempts
  per provider with waits of 30s, then 90s between attempts, then fail.
- See also Stage 2 (easy‑filter): `algebra_dataset_problem_siever_easy_filter.py`, which can
  discard composites that are too easy for a chosen panel (e.g., “weak” models) when a
  configurable fraction of providers solve them.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import csv
from typing import Any, Dict, List, Optional, Tuple

# --- Logging helper (define early so main can use it) ---
def _log(msg: str) -> None:
    try:
        from tqdm import tqdm  # type: ignore
        tqdm.write(str(msg))
    except Exception:
        try:
            print(str(msg))
        except Exception:
            pass


def load_records(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    text = p.read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
    except Exception:
        pass
    # JSONL fallback
    out: List[Dict[str, Any]] = []
    for i, ln in enumerate(text.splitlines()):
        ln = ln.strip()
        if not ln:
            continue
        try:
            o = json.loads(ln)
        except Exception as e:
            raise ValueError(f"Invalid JSONL on line {i+1}: {e}")
        if not isinstance(o, dict):
            raise ValueError(f"Expected an object per line (line {i+1})")
        out.append(o)
    return out


def _to_float(x: Any) -> Optional[float]:
    try:
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            s = x.strip().replace(",", "")
            # Fractional forms e.g., "-23999/6000"
            if "/" in s and all(part.strip("+- ").replace(".", "", 1).isdigit() for part in s.split("/", 1)):
                try:
                    num, den = s.split("/", 1)
                    return float(num) / float(den)
                except Exception:
                    pass
            return float(s)
    except Exception:
        return None
    return None


def normalize_solution_eval(sol_eval: Any, var_names: List[str]) -> Dict[str, List[float]]:
    """Return a mapping var -> list[float] of acceptable values.

    Accepts:
    - dict var-> number | list[number]
    - list[number] (mapped positionally to var_names)
    """
    out: Dict[str, List[float]] = {}
    if isinstance(sol_eval, dict):
        for k, v in sol_eval.items():
            if isinstance(v, (list, tuple)):
                vs = []
                for t in v:
                    fv = _to_float(t)
                    if fv is not None and math.isfinite(fv):
                        vs.append(fv)
                if vs:
                    out[str(k)] = vs
            else:
                fv = _to_float(v)
                if fv is not None and math.isfinite(fv):
                    out[str(k)] = [fv]
        return out
    # list/tuple positional
    if isinstance(sol_eval, (list, tuple)):
        for i, t in enumerate(sol_eval):
            if i >= len(var_names):
                break
            fv = _to_float(t)
            if fv is not None and math.isfinite(fv):
                out[str(var_names[i])] = [fv]
        return out
    return out


def _var_order(rec: Dict[str, Any]) -> List[str]:
    vars_field = rec.get("variables")
    if isinstance(vars_field, dict) and vars_field:
        return list(vars_field.keys())
    # fallback: try eq_system_str to infer symbols
    return []


def _percent_error(actual: float, proposed: float) -> float:
    denom = abs(actual)
    if denom < 1e-12:
        return 0.0 if abs(actual - proposed) <= 1e-12 else float("inf")
    return 100.0 * (abs(actual - proposed) / denom)


def build_prompt(problem_text: str, is_multivar: bool) -> str:
    header = (
        "Solve the following algebra problem and return JSON only. "
        "Do not include any extra text or code fences.\n\n"
    )
    if is_multivar:
        schema = (
            "Output schema:\n"
            "{\n  \"answers\": {\n    \"x\": <number>,\n    \"y\": <number>,\n    ...\n  }\n}\n"
        )
    else:
        schema = (
            "Output schema:\n"
            "{\n  \"final_answer\": <number>\n}\n"
        )
    return f"{header}{schema}\nProblem:\n{problem_text.strip()}\n"


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extract a top-level JSON object from text (no heavy heuristics)."""
    s = str(text or "").strip()
    if not s:
        return None
    # Try direct parse
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Try to locate first {...} block
    lb = s.find("{")
    rb = s.rfind("}")
    if lb != -1 and rb != -1 and rb > lb:
        try:
            frag = s[lb : rb + 1]
            obj = json.loads(frag)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


def parse_model_answers(text: str, var_names: List[str]) -> Tuple[Optional[float], Dict[str, float]]:
    """Parse model output.

    Returns (final_answer_if_single, answers_by_var).
    For multi-variable, answers_by_var contains numeric values for any variables found.
    """
    obj = _extract_json(text)
    if not obj:
        return None, {}
    # Single-variable convenience
    if "final_answer" in obj:
        fv = _to_float(obj.get("final_answer"))
        return (fv if fv is not None else None), {}
    # Multi-variable common shapes
    if isinstance(obj.get("answers"), dict):
        out: Dict[str, float] = {}
        for k, v in obj["answers"].items():
            fv = _to_float(v)
            if fv is not None and math.isfinite(fv):
                out[str(k)] = fv
        return None, out
    # Fallback: look for direct vars at top-level
    out: Dict[str, float] = {}
    for k in var_names:
        if k in obj:
            fv = _to_float(obj.get(k))
            if fv is not None and math.isfinite(fv):
                out[k] = fv
    return None, out


def grade_prediction(
    proposed_scalar: Optional[float],
    proposed_map: Dict[str, float],
    sol_eval: Dict[str, List[float]],
    error_pct: float,
) -> bool:
    # Single variable case: if a single proposed scalar exists, match against any ground-truth value
    if proposed_scalar is not None and len(sol_eval) == 1:
        gt_vals = next(iter(sol_eval.values()))
        for v in gt_vals:
            if _percent_error(v, float(proposed_scalar)) <= error_pct:
                return True
        return False
    # Multi-variable: every variable must match (any one ground-truth per var)
    for var, gt_vals in sol_eval.items():
        pv = proposed_map.get(var)
        if pv is None:
            return False
        ok = any(_percent_error(gv, float(pv)) <= error_pct for gv in gt_vals)
        if not ok:
            return False
    return True if sol_eval else False


def run_one_problem(
    rec: Dict[str, Any],
    specs: List[Dict[str, Any]],
    error_pct: float,
    max_tokens: int,
    request_timeout: int,
) -> Dict[str, Any]:
    """Return a summary dict with keys: ok(bool), id, solver_provider/model/reasoning, solver_attempt(int), providers_tried(int)."""
    problem_text = str(rec.get("problem_text") or rec.get("item_c_problem") or "").strip()
    if not problem_text:
        return False
    var_order = _var_order(rec)
    sol_eval_in = rec.get("solution_eval")
    sol_eval = normalize_solution_eval(sol_eval_in, var_order)
    if not sol_eval:
        return False
    is_multivar = len(sol_eval) > 1

    # Build prompt once
    prompt = build_prompt(problem_text, is_multivar=is_multivar)

    # Provider adapters
    from algebra_dataset_benchmarking_tool import ModelSpec, make_adapter

    # Try providers sequentially
    for so in specs:
        spec = ModelSpec.from_obj(so)
        adapter = make_adapter(spec, timeout=int(request_timeout))
        # temperature by reasoning level
        temp_map = {"low": 0.0, "medium": 0.3, "high": 0.7}
        temperature = temp_map.get(spec.reasoning_level, 0.2)

        # Backoff: up to 3 attempts
        attempts = 0
        last_text = ""
        last_err: Optional[str] = None
        while attempts < 3:
            try:
                last_text = adapter.complete(prompt, max_tokens=int(max_tokens), temperature=float(temperature))
                if not (isinstance(last_text, str) and last_text.strip()):
                    raise RuntimeError("empty_response")
                break
            except Exception as e:
                attempts += 1
                last_err = f"{type(e).__name__}: {e}"
                import time as _t
                if attempts == 1:
                    _t.sleep(30)
                elif attempts == 2:
                    _t.sleep(90)
                else:
                    last_text = ""
                    break

        if not last_text:
            # Informative error logging after exhausting attempts
            try:
                pid = rec.get("id")
                perr = getattr(adapter, "last_error", None)
                msg = last_err or (str(perr) if perr else "unknown_error")
                _log(f"[ERROR] {spec.provider}:{spec.model}:{spec.reasoning_level} id={pid} reason={msg}")
            except Exception:
                pass
            continue

        # Parse
        scalar, mapping = parse_model_answers(last_text, list(sol_eval.keys()))
        if grade_prediction(scalar, mapping, sol_eval, error_pct=error_pct):
            return {
                "ok": True,
                "id": rec.get("id"),
                "solver_provider": spec.provider,
                "solver_model": spec.model,
                "solver_reasoning": spec.reasoning_level,
                "solver_attempt": attempts + 1 if attempts < 3 and last_text else (attempts or 1),
                "providers_tried": specs.index(so) + 1 if so in specs else None,
            }
    return {
        "ok": False,
        "id": rec.get("id"),
        "solver_provider": None,
        "solver_model": None,
        "solver_reasoning": None,
        "solver_attempt": 0,
        "providers_tried": len(specs),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Sieve an algebra dataset into tractable vs intractable via LLM providers.")
    ap.add_argument("--input", "-i", required=True, help="Path to input dataset (JSON or JSONL)")
    ap.add_argument("--providers-file", required=True, help="Providers/models JSON (same schema as benchmarking tool)")
    ap.add_argument("--select", default=None, help="Regex to filter provider:model:reasoning (optional)")
    ap.add_argument("--error-pct", type=float, default=0.5, help="Acceptable percent error per component (default 0.5)")
    ap.add_argument("--max-tokens", type=int, default=30000, help="Max completion tokens per call")
    ap.add_argument("--request-timeout", type=int, default=300, help="Per-request timeout seconds")
    ap.add_argument("--jobs", type=int, default=4, help="Parallel workers (problems in parallel)")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    records = load_records(args.input)
    # Load providers
    from algebra_dataset_benchmarking_tool import load_providers_config
    specs = [s.__dict__ for s in load_providers_config(Path(args.providers_file))]
    if args.select:
        import re
        rx = re.compile(args.select)
        specs = [s for s in specs if rx.search(f"{s.get('provider')}:{s.get('model')}:{s.get('reasoning_level')}")]
    if not specs:
        print("No providers selected.")
        return 2

    # Process in parallel (threaded IO) with progress bar when available
    from concurrent.futures import ThreadPoolExecutor, as_completed
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        tqdm = None  # type: ignore
    tractable: List[Dict[str, Any]] = []
    intractable: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    futs = {}
    with ThreadPoolExecutor(max_workers=max(1, int(args.jobs))) as ex:
        for rec in records:
            fut = ex.submit(
                run_one_problem,
                rec,
                specs,
                float(args.error_pct),
                int(args.max_tokens),
                int(args.request_timeout),
            )
            futs[fut] = rec
        bar = None
        if tqdm is not None:
            bar = tqdm(total=len(records), desc="Sieving", unit="prob")
        for fut in as_completed(futs):
            ok = False
            res: Dict[str, Any] = {}
            try:
                res = fut.result() or {}
                ok = bool(res.get("ok"))
            except Exception:
                ok = False
            if ok:
                tractable.append(futs[fut])
            else:
                intractable.append(futs[fut])
            # Record CSV summary row
            rid = res.get("id") if res else (futs[fut].get("id"))
            summary_rows.append({
                "id": rid,
                "tractable": bool(ok),
                "solver_provider": res.get("solver_provider") if res else None,
                "solver_model": res.get("solver_model") if res else None,
                "solver_reasoning": res.get("solver_reasoning") if res else None,
                "solver_attempt": res.get("solver_attempt") if res else None,
                "providers_tried": res.get("providers_tried") if res else None,
            })
            if bar is not None:
                bar.update(1)
                bar.set_postfix(keep=len(tractable), drop=len(intractable))
        if bar is not None:
            bar.close()

    # Emit files
    p = Path(args.input)
    stem = p.stem
    suffix = p.suffix or ".json"
    out_dir = p.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    t_count = len(tractable)
    it_count = len(intractable)
    t_path = out_dir / f"{stem}_tractable_{t_count}{suffix}"
    it_path = out_dir / f"{stem}_intractable_{it_count}{suffix}"
    t_path.write_text(json.dumps(tractable, ensure_ascii=False, indent=2), encoding="utf-8")
    it_path.write_text(json.dumps(intractable, ensure_ascii=False, indent=2), encoding="utf-8")
    # Write CSV summary
    csv_path = out_dir / f"{stem}_sieve_summary.csv"
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "id", "tractable", "solver_provider", "solver_model", "solver_reasoning", "solver_attempt", "providers_tried"
            ])
            w.writeheader()
            for row in summary_rows:
                w.writerow(row)
        _log(f"Wrote CSV summary: {csv_path.name}")
    except Exception as e:
        _log(f"[WARN] Failed to write CSV summary: {e}")
    _log(f"Tractable: {t_count}  Intractable: {it_count}")
    _log(f"Wrote {t_path.name} and {it_path.name} in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
