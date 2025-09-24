#!/usr/bin/env python3
"""
Deterministic Composite Dataset Generator
========================================

Build composite linear problems from an existing algebra dataset (JSON/JSONL) — no LLM calls.

AGSM alignment
--------------
Implements AGSM’s difficulty scaling by composing K verified leaf problems into
one multi‑step word problem whose final answer is a linear expression of the K
sub‑answers. Difficulty level n corresponds to n sub‑problems.

What it does
------------
For each composite item:
- Uniformly sample K sub‑problems from the input dataset (by default, without replacement within a composite).
- From each sub‑problem, pick the numerically largest value (most positive) across its `solution_eval` entries.
  - If a value is a list/tuple, every numeric member is considered; the maximum is selected.
  - Non‑numeric entries (`None`, strings) are ignored. If a sub‑problem has no numeric value, `0.0` is used.
- Assign an integer coefficient sampled uniformly from `[coeff_min, coeff_max]` (inclusive).
- Build a linear template string:
    `c1*(sub_1) + c2*(sub_2) + ... + cK*(sub_K)`
- Emit a natural‑language prompt that includes the equation template and the K sub‑problem texts.
- Compute the deterministic numeric answer: `sum_i c_i * chosen_answer_i`.

Expected input
--------------
An algebra dataset file (JSON array or JSONL) where each record contains at least:
- `solution_eval`: a mapping from variable names to numeric values (number or list of numbers).
- `problem_text` (preferred) or `item_c_problem`: the sub‑problem text to display.

Output format
-------------
Each composite item has:
- `id`: uuid for the composite item
- `question`: composed prompt string
- `equation_template`: linear expression with `sub_i` placeholders
- `deterministic_answer`: float (the computed sum)
- `sub_components` (object `sub_1..sub_K` → details):
  - `coefficient` (int)
  - `chosen_answer` (float)
  - `chosen_answer_var` (optional str; source variable name)
  - `problem_text` (str)
  - `source` (the full original sub‑problem JSON)

Lite files
----------
For convenience, the program also writes a companion “lite” file alongside the full output. The lite file contains the
minimal composite fields for quick inspection and downstream consumption, plus a compact per‑sub‑problem summary with
the exact equations and solutions from the original sub‑problem source:
- `id`
- `question`
- `deterministic_answer`
- `sub_components` (object `sub_1..sub_K` → selected fields from the original source):
  - `eq_system_str`: list of SymPy equation strings
  - `eq_system_ast`: list of SymPy AST `repr` strings
  - `sympy_src`: minimal Python snippet to reproduce and solve
  - `solution`: mapping variable → string or list of strings
  - `solution_eval`: mapping variable → float or list of floats

Naming: It uses the same base name as the full output, with `_lite` inserted before the extension. In scaled difficulty
batch mode, `_lite` is added after the difficulty suffix (e.g., `..._diff3_lite.json`). Lite files are always generated.

CLI Usage
---------
    python algebra_dataset_generator_deterministic_composite.py \
      --input algebraTest/algebra_synth_selected.jsonl \
      --output algebraTest/deterministic_composites.json \
      --runs 50 --subproblems 3 --seed 42 --coeff-min 1 --coeff-max 100 \
      --insert-random-facts --format json

Quick Demo
----------
Using your existing dataset `algebraTest/quickTestAmbientV6a.json`, generate a single composite with 4 sub‑problems,
seeded for reproducibility, and include random facts where available:

    python algebra_dataset_generator_deterministic_composite.py \
      --input algebraTest/quickTestAmbientV6a.json \
      --output algebraTest/deterministic_composites_demo_v6a.json \
      --runs 1 --subproblems 4 --seed 420 --coeff-min 1 --coeff-max 10 \
      --insert-random-facts --format json

This writes a pretty JSON array with one item to `algebraTest/deterministic_composites_demo_v6a.json`. In the composed
question, each sub‑problem writeup may have its `random_fact` prepended or appended (when present in the input record),
and the `equation_template` and `deterministic_answer` reflect the sampled coefficients and chosen sub‑answers.

Parameters
----------
- `--input, -i` (str, required):
  Path to the source dataset. Accepts a JSON array or JSONL. Each record should include `solution_eval` and
  either `problem_text` (preferred) or `item_c_problem`.

- `--output, -o` (str, default: ./algebraTest/deterministic_composites.json):
  Output file path. The extension determines nothing; use `--format` to choose JSON vs JSONL.

- `--runs, -n` (int, default: 10):
  Number of composite items to generate.

- `--subproblems, -k` (int, default: 3):
  Number of sub‑problems to include per composite.
  - Without `--with-replacement`, K must be ≤ number of input records.
  - With `--with-replacement`, repeated sub‑problems in a single composite are allowed.

- `--seed` (int, default: 42):
  Random seed for reproducibility (affects sub‑problem sampling and coefficients).

- `--coeff-min` (int, default: 1) and `--coeff-max` (int, default: 100):
  Inclusive coefficient bounds for the per‑sub‑problem multiplier. Must satisfy `coeff_min ≤ coeff_max`.

- `--with-replacement` (flag, default: off):
  Sample sub‑problems with replacement within a composite. If not set, sampling is without replacement per composite.

- `--insert-random-facts` (flag, default: off):
  If enabled, and when a sub‑problem record includes a `random_fact` field (string), the generator will randomly decide
  (independently per sub‑problem) to prepend or append that fact to the sub‑problem writeup inside the composed `question`.
  The prepend/append coin flip is deterministic w.r.t. `--seed`.

- `--format` (`json`|`jsonl`, default: `json`):
  Output format. `json` writes a pretty JSON array; `jsonl` writes one compact JSON object per line.

- `--indent` (int, default: 2):
  Indent for pretty JSON; ignored when `--format jsonl`.

Notes
-----
- The `source` field is included verbatim under each `sub_components[sub_i]`. Depending on input size, this can be large.
- This tool does not call any LLMs; it is fully deterministic given `--seed` and the input file.
 - `--maximize-family-variety`: When enabled, each composite will prefer sub-problems from distinct `family` values
   (as found in the input records) until all distinct families have been used once; then repeats are allowed.
 - Narrative perturbations: if upstream records include `random_fact`, enabling `--insert-random-facts` will prepend
   or append a short topical fact to each sub‑problem’s text. This is semantics‑independent and does not change
   variables, constants, coefficients, or the deterministic answer.

Scaled Difficulty Batch Mode
----------------------------
To auto‑generate a ladder of difficulties in one command, pass:

    --generate-batch-of-scaled-difficulty

This runs 10 batches (diff1..diff10). For each level N (1 ≤ N ≤ 10):
- Uses N sub‑problems per composite.
- Writes to a file that appends `_diffN` before the output extension (e.g., `output_diff3.json`).
- Runs the same number of composites per file (`--runs`).
- Samples sub‑problems independently per level, using a deterministic seed offset (`--seed + N`).

Example:

    python algebra_dataset_generator_deterministic_composite.py \
      --input algebraTest/quickTestAmbientV6a.json \
      --output algebraTest/composites_scaled.json \
      --runs 5 --seed 420 --coeff-min 1 --coeff-max 10 \
      --insert-random-facts --generate-batch-of-scaled-difficulty --format json

This writes 10 files (`..._diff1.json` through `..._diff10.json`) to `algebraTest/`.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _is_number(x: Any) -> bool:
    try:
        return isinstance(x, (int, float)) and math.isfinite(float(x))
    except Exception:
        return False


def load_input_records(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    text = p.read_text(encoding="utf-8")
    s = text.lstrip()
    # Heuristic: JSONL if first non-space char is '{' and there are many newlines with JSON objects on each
    # Prefer robust detection: try JSON parse first; if fails, try JSONL
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        # Single JSON object; wrap as list
        return [obj]
    except Exception:
        pass
    # Try JSONL
    recs: List[Dict[str, Any]] = []
    for i, ln in enumerate(text.splitlines()):
        ln = ln.strip()
        if not ln:
            continue
        try:
            rec = json.loads(ln)
        except Exception as e:
            raise ValueError(f"Invalid JSONL on line {i+1}: {e}")
        if not isinstance(rec, dict):
            raise ValueError(f"Expected JSON object per line; got {type(rec)} on line {i+1}")
        recs.append(rec)
    return recs


def pick_most_positive(solution_eval: Dict[str, Any]) -> Tuple[float, Optional[str]]:
    """Return (max_value, source_var) across all entries of solution_eval.
    Accept floats/ints directly; for lists/tuples, consider all numeric members.
    Raises ValueError if no numeric values found.
    """
    best_val: Optional[float] = None
    best_var: Optional[str] = None
    for var, val in (solution_eval or {}).items():
        if isinstance(val, (list, tuple)):
            for t in val:
                if _is_number(t):
                    fv = float(t)
                    if (best_val is None) or (fv > best_val):
                        best_val, best_var = fv, var
        else:
            if _is_number(val):
                fv = float(val)
                if (best_val is None) or (fv > best_val):
                    best_val, best_var = fv, var
    if best_val is None:
        raise ValueError("No numeric solutions in solution_eval")
    return best_val, best_var


@dataclass
class SubComponent:
    coefficient: int
    chosen_answer: float
    chosen_answer_var: Optional[str]
    problem_text: str
    source: Dict[str, Any]


@dataclass
class CompositeRecord:
    id: str
    question: str
    equation_template: str
    deterministic_answer: float
    sub_components: Dict[str, SubComponent]
    # Optional run metadata
    seed: Optional[int] = None
    coeff_min: Optional[int] = None
    coeff_max: Optional[int] = None


def build_composites(
    records: List[Dict[str, Any]],
    runs: int,
    k: int,
    coeff_min: int = 1,
    coeff_max: int = 100,
    seed: Optional[int] = None,
    sample_within_composite_with_replacement: bool = False,
    insert_random_facts: bool = False,
    maximize_family_variety: bool = False,
    avoid_repeat_subproblems: bool = False,
    family_quota_mode: str = "none",
    per_family_quota: Optional[int] = None,
    global_state: Optional[Dict[str, Any]] = None,
    quota_limits: Optional[Dict[str, int]] = None,
) -> List[CompositeRecord]:
    if runs <= 0:
        return []
    if k <= 0:
        raise ValueError("--subproblems must be >= 1")
    if coeff_min > coeff_max:
        raise ValueError("--coeff-min must be <= --coeff-max")
    rng = random.Random(seed)
    n = len(records)
    if n == 0:
        raise ValueError("Input dataset is empty")

    out: List[CompositeRecord] = []
    # Precompute families for variety control
    families: List[str] = []
    for r in records:
        fam = r.get("family")
        families.append(str(fam) if fam is not None else "unknown")
    all_families = set(families)
    fam_to_indices: Dict[str, List[int]] = {}
    for i, fam in enumerate(families):
        fam_to_indices.setdefault(fam, []).append(i)

    # Global state for no-repeat and per-family usage across calls (batch mode)
    if global_state is None:
        global_state = {"used": set(), "family_usage": {}}
    used_global: set[int] = set(global_state.get("used") or set())
    fam_usage: Dict[str, int] = dict(global_state.get("family_usage") or {})

    # Build family quota limits if not provided
    fam_limits: Optional[Dict[str, int]] = None
    if quota_limits is not None:
        fam_limits = dict(quota_limits)
    else:
        # Determine total picks for this run
        total_needed = max(0, int(runs) * int(k))
        fam_list = sorted(all_families)
        fam_counts = {f: len(fam_to_indices.get(f, [])) for f in fam_list}
        if family_quota_mode in ("equal", "proportional") or (per_family_quota and per_family_quota > 0):
            fam_limits = {}
            if family_quota_mode == "equal":
                base = total_needed // max(1, len(fam_list))
                rem = total_needed - base * max(1, len(fam_list))
                for i, f in enumerate(fam_list):
                    fam_limits[f] = base + (1 if i < rem else 0)
            elif family_quota_mode == "proportional":
                total_records = max(1, len(records))
                provisional = {}
                assigned = 0
                for f in fam_list:
                    share = total_needed * (fam_counts.get(f, 0) / total_records)
                    lim = int(round(share))
                    lim = max(1, lim) if total_needed > 0 else 0
                    provisional[f] = lim
                    assigned += lim
                fam_limits = dict(provisional)
                # Adjust to match at least total_needed (soft cap)
                i = 0
                while assigned < total_needed and fam_list:
                    fam_limits[fam_list[i % len(fam_list)]] += 1
                    assigned += 1
                    i += 1
            else:
                # No automatic mode, but explicit per_family_quota given
                for f in fam_list:
                    fam_limits[f] = per_family_quota if per_family_quota else total_needed
            # Apply explicit per_family_quota as a hard cap if provided
            if per_family_quota and per_family_quota > 0:
                for f in fam_list:
                    fam_limits[f] = min(fam_limits.get(f, total_needed), int(per_family_quota))

    # Helper: choose index under constraints
    def choose_index(rng, used_local: set[int], used_fams_local: set[str]) -> int:
        # Build initial available set
        if sample_within_composite_with_replacement:
            available = list(range(n))
        else:
            available = [i for i in range(n) if i not in used_local]
            if not available:
                available = list(range(n))

        # Enforce avoid-repeat across composites (soft: relax if empty)
        pool = available
        if avoid_repeat_subproblems:
            cand = [i for i in pool if i not in used_global]
            pool = cand if cand else pool

        # Enforce per-family quotas (soft cap)
        if fam_limits is not None:
            cand = [i for i in pool if fam_usage.get(families[i], 0) < fam_limits.get(families[i], total_needed if runs and k else 0)]
            pool = cand if cand else pool

        # Prefer variety within composite if enabled
        if maximize_family_variety:
            unseen_fams = [i for i in pool if families[i] not in used_fams_local]
            if unseen_fams:
                pool = unseen_fams

        if not pool:
            # As a final fallback, pick from all
            pool = list(range(n))
        return rng.choice(pool)

    # Special semantics for choose-1 now handled by general picker to honor global constraints
    if k == 1 and runs > n and not sample_within_composite_with_replacement:
        print(
            f"[WARN] Requested runs ({runs}) exceeds available input records ({n}) for k=1; consider --with-replacement.",
            file=sys.stderr,
        )

    # General case (k >= 2): optionally maximize family variety within each composite
    for _ in range(runs):
        used_idxs: set[int] = set()
        used_fams_local: set[str] = set()
        idxs: List[int] = []
        for _i in range(k):
            pick = choose_index(rng, used_local=used_idxs, used_fams_local=used_fams_local)
            idxs.append(pick)
            used_fams_local.add(families[pick])
            if not sample_within_composite_with_replacement:
                used_idxs.add(pick)
            # Update global state
            if avoid_repeat_subproblems:
                used_global.add(pick)
            fam_usage[families[pick]] = fam_usage.get(families[pick], 0) + 1

        sub_map: Dict[str, SubComponent] = {}
        eq_terms: List[str] = []
        det_answer = 0.0

        for i, idx in enumerate(idxs, start=1):
            rec = records[idx]
            prob_text = str(rec.get("problem_text") or rec.get("item_c_problem") or "")
            sol_eval = rec.get("solution_eval")
            if not isinstance(sol_eval, dict):
                # Skip invalid sub-problem; try to re-pick another
                # Fallback: treat as zero answer so composite remains valid
                best_val, best_var = 0.0, None
            else:
                try:
                    best_val, best_var = pick_most_positive(sol_eval)
                except Exception:
                    best_val, best_var = 0.0, None

            coeff = rng.randint(coeff_min, coeff_max)
            det_answer += coeff * best_val
            key = f"sub_{i}"
            sub_map[key] = SubComponent(
                coefficient=coeff,
                chosen_answer=float(best_val),
                chosen_answer_var=best_var,
                problem_text=prob_text,
                source=rec,
            )
            eq_terms.append(f"{coeff}*(sub_{i})")

        equation_template = " + ".join(eq_terms)

        # Build question text
        question = (
            "Solve the following. For each sub-problem:\n"
            "- Compute all real numeric solutions.\n"
            "- If a variable has multiple real solutions, select the numerically largest for that variable.\n"
            "- If the sub-problem has multiple variables, select the numerically largest value among those variables’ selected values.\n"
            "- If no real numeric solution exists, use 0.0 for that sub-problem.\n\n"
            "Let sub_i denote the selected numeric value for sub-problem i. "
            "Compute the value of the linear expression shown below. "
            "Return only a single JSON object: {\"final_answer\": <number>}. "
            "Do not include any other text, markdown, or code fences. "
            "Use a plain number (scientific notation allowed).\n"
            "Equation and sub-problems follow (where sub-problem answers are specified by sub_1, sub_2, etc.):\n"
            + equation_template
            + "\n\n"
        )
        # Append each sub-problem text
        for i in range(1, k + 1):
            it = sub_map[f"sub_{i}"]
            base_text = (it.problem_text or '').strip()
            enriched = base_text
            if insert_random_facts:
                try:
                    fact = (it.source or {}).get('random_fact')
                    if isinstance(fact, str) and fact.strip():
                        # Randomly choose to prepend or append (deterministic via rng/seed)
                        if rng.choice([True, False]):
                            enriched = f"{fact.strip()}\n{base_text}"
                        else:
                            enriched = f"{base_text}\n{fact.strip()}"
                except Exception:
                    enriched = base_text
            question += f"Sub-problem {i}:\n{enriched}\n\n"

        out.append(
            CompositeRecord(
                id=str(uuid.uuid4()),
                question=question.strip(),
                equation_template=equation_template,
                deterministic_answer=float(det_answer),
                sub_components=sub_map,
                seed=seed,
                coeff_min=coeff_min,
                coeff_max=coeff_max,
            )
        )
    # persist global state
    global_state["used"] = used_global
    global_state["family_usage"] = fam_usage
    return out


def write_output(path: str, composites: List[CompositeRecord], fmt: str = "json", indent: int = 2) -> None:
    fmt = (fmt or "json").lower()
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    data = []
    for c in composites:
        obj = asdict(c)
        # dataclasses in sub_components values already dict-ified by asdict
        data.append(obj)
    if fmt == "jsonl":
        with open(path, "w", encoding="utf-8") as f:
            for obj in data:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        return
    # default pretty JSON array
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def write_lite_output(path: str, composites: List[CompositeRecord], fmt: str = "json", indent: int = 2) -> None:
    """Write compact lite records for each composite including per‑sub equations/solutions.

    Fields per item:
      - id, question, deterministic_answer
      - sub_components[sub_i]: eq_system_str, eq_system_ast, sympy_src, solution, solution_eval

    - json: pretty JSON array
    - jsonl: one JSON object per line
    """
    fmt = (fmt or "json").lower()
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    def _lite_sub(sc: SubComponent) -> Dict[str, Any]:
        src = sc.source if isinstance(sc.source, dict) else {}
        # Pull the exact equation/solution fields from the original sub‑problem record.
        return {
            "eq_system_str": list(src.get("eq_system_str") or []),
            "eq_system_ast": list(src.get("eq_system_ast") or []),
            "sympy_src": src.get("sympy_src"),
            "solution": src.get("solution"),
            "solution_eval": src.get("solution_eval"),
        }

    items: List[Dict[str, Any]] = []
    for c in composites:
        subs: Dict[str, Any] = {}
        # Preserve sub_i ordering by numeric index where possible
        for key in sorted(c.sub_components.keys(), key=lambda k: (int(k.split("_")[-1]) if k and k.split("_")[-1].isdigit() else 1_000_000, k)):
            subs[key] = _lite_sub(c.sub_components[key])
        items.append({
            "id": c.id,
            "question": c.question,
            "deterministic_answer": c.deterministic_answer,
            "sub_components": subs,
        })

    if fmt == "jsonl":
        with open(path, "w", encoding="utf-8") as f:
            for obj in items:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=indent)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Deterministic composite generator (no LLMs). See module docstring for details and examples.")
    ap.add_argument(
        "--input", "-i", type=str, required=True,
        help=(
            "Path to source dataset (JSON array or JSONL). Records should contain 'solution_eval' and either "
            "'problem_text' or 'item_c_problem'."
        ),
    )
    ap.add_argument(
        "--output", "-o", type=str,
        default=str(Path.cwd() / "algebraTest" / "deterministic_composites.json"),
        help=(
            "Output file path. Use --format to control JSON vs JSONL (extension alone is not used to infer format)."
        ),
    )
    ap.add_argument(
        "--runs", "-n", type=int, default=10,
        help="Number of composite problems to generate (default: 10)",
    )
    ap.add_argument(
        "--subproblems", "-k", type=int, default=3,
        help=(
            "Number of sub-problems per composite (default: 3). Without --with-replacement, must be ≤ number of input records."
        ),
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility across sampling and coefficient choices (default: 42)",
    )
    ap.add_argument(
        "--coeff-min", type=int, default=1,
        help="Minimum coefficient (inclusive, default: 1)",
    )
    ap.add_argument(
        "--coeff-max", type=int, default=100,
        help="Maximum coefficient (inclusive, default: 100)",
    )
    ap.add_argument(
        "--with-replacement", action="store_true",
        help="Allow repeated sub-problems within a composite (sampling with replacement).",
    )
    ap.add_argument(
        "--insert-random-facts", action="store_true",
        help=(
            "If present, and when a sub-problem record contains 'random_fact', randomly (per sub) prepend or append "
            "that fact to the sub-problem writeup in the composed question (coin flips are seeded by --seed)."
        ),
    )
    ap.add_argument(
        "--maximize-family-variety", action="store_true",
        help=(
            "Prefer unique 'family' values among the k sub-problems within each composite until all distinct families in the input "
            "have been used at least once; then allow repeats."
        ),
    )
    ap.add_argument(
        "--avoid-repeat-subproblems", action="store_true",
        help=(
            "Avoid reusing the same source sub-problem across composites for the entire run. If the pool is exhausted, the "
            "constraint is relaxed to allow minimal repeats."
        ),
    )
    ap.add_argument(
        "--per-family-quota", type=int, default=None,
        help=(
            "Optional soft cap on the total number of selections drawn from any single family across the run."
        ),
    )
    ap.add_argument(
        "--family-quota-mode", type=str, default="none", choices=["none", "equal", "proportional"],
        help=(
            "Automatic family quota allocation: 'equal' distributes selections evenly across families; 'proportional' allocates "
            "by input family frequency. Caps are soft; if no eligible candidates remain, constraints are relaxed."
        ),
    )
    ap.add_argument(
        "--format", type=str, default="json", choices=["json", "jsonl"],
        help="Output format: 'json' (pretty JSON array) or 'jsonl' (one JSON object per line).",
    )
    ap.add_argument(
        "--indent", type=int, default=2,
        help="Indentation for pretty JSON when --format json (default: 2).",
    )
    ap.add_argument(
        "--generate-batch-of-scaled-difficulty", action="store_true",
        help=(
            "Generate a series of 10 files with increasing difficulty (diff1..diff10), where diffN uses N sub-problems per "
            "composite. Uses the provided --output as a base name and inserts _diffN before the extension. Each file runs "
            "with the same --runs count; sub-problems are sampled independently with seeds offset by +N for reproducibility."
        ),
    )
    args = ap.parse_args(argv)

    try:
        recs = load_input_records(args.input)
    except Exception as e:
        print(f"Error loading input: {e}", file=sys.stderr)
        return 2

    # Scaled difficulty batch mode: generate 10 files diff1..diff10
    if bool(getattr(args, 'generate_batch_of_scaled_difficulty', False)):
        out_base = Path(args.output)
        parent = out_base.parent
        stem = out_base.stem
        suffix = out_base.suffix or ".json"
        # Global state across levels for --avoid-repeat-subproblems and quotas
        global_state = {"used": set(), "family_usage": {}}
        # Optional global quota limits across all levels
        fam_list_all = sorted({str(r.get('family') or 'unknown') for r in recs})
        fam_counts_all = {f: 0 for f in fam_list_all}
        for r in recs:
            fam = str(r.get('family') or 'unknown')
            fam_counts_all[fam] = fam_counts_all.get(fam, 0) + 1
        quota_limits_global: Optional[Dict[str, int]] = None
        if args.family_quota_mode in ("equal", "proportional") or (args.per_family_quota and args.per_family_quota > 0):
            total_needed_all = int(args.runs) * sum(range(1, 11))  # runs * (1+..+10)
            quota_limits_global = {}
            if args.family_quota_mode == "equal":
                base = total_needed_all // max(1, len(fam_list_all))
                rem = total_needed_all - base * max(1, len(fam_list_all))
                for i, f in enumerate(fam_list_all):
                    quota_limits_global[f] = base + (1 if i < rem else 0)
            elif args.family_quota_mode == "proportional":
                total_records = max(1, len(recs))
                assigned = 0
                for f in fam_list_all:
                    share = total_needed_all * (fam_counts_all.get(f, 0) / total_records)
                    lim = int(round(share))
                    lim = max(1, lim) if total_needed_all > 0 else 0
                    quota_limits_global[f] = lim
                    assigned += lim
                i = 0
                while assigned < total_needed_all and fam_list_all:
                    quota_limits_global[fam_list_all[i % len(fam_list_all)]] += 1
                    assigned += 1
                    i += 1
            # Apply explicit hard cap if provided
            if args.per_family_quota and args.per_family_quota > 0:
                for f in fam_list_all:
                    quota_limits_global[f] = min(quota_limits_global.get(f, total_needed_all), int(args.per_family_quota))
        for level in range(1, 11):
            k_level = level
            out_name = f"{stem}_diff{level}{suffix}"
            out_path = str(parent / out_name)
            try:
                comps = build_composites(
                    records=recs,
                    runs=int(args.runs),
                    k=int(k_level),
                    coeff_min=int(args.coeff_min),
                    coeff_max=int(args.coeff_max),
                    seed=int(args.seed) + level,
                    sample_within_composite_with_replacement=bool(args.with_replacement),
                    insert_random_facts=bool(args.insert_random_facts),
                    maximize_family_variety=bool(args.maximize_family_variety),
                    avoid_repeat_subproblems=bool(args.avoid_repeat_subproblems),
                    family_quota_mode=str(args.family_quota_mode),
                    per_family_quota=(None if args.per_family_quota is None else int(args.per_family_quota)),
                    global_state=global_state,
                    quota_limits=quota_limits_global,
                )
            except Exception as e:
                print(f"Error building composites for diff{level}: {e}", file=sys.stderr)
                return 3
            try:
                write_output(out_path, comps, fmt=args.format, indent=int(args.indent))
            except Exception as e:
                print(f"Error writing output for diff{level}: {e}", file=sys.stderr)
                return 4
            print(f"Wrote {len(comps)} composite items to {out_path}")
            # Also write the companion lite file
            try:
                out_lite_path = str(parent / f"{stem}_diff{level}_lite{suffix}")
                write_lite_output(out_lite_path, comps, fmt=args.format, indent=int(args.indent))
                print(f"Wrote lite file to {out_lite_path}")
            except Exception as e:
                print(f"Error writing lite output for diff{level}: {e}", file=sys.stderr)
                return 4
        return 0
    else:
        try:
            comps = build_composites(
                records=recs,
                runs=int(args.runs),
                k=int(args.subproblems),
                coeff_min=int(args.coeff_min),
                coeff_max=int(args.coeff_max),
                seed=int(args.seed),
                sample_within_composite_with_replacement=bool(args.with_replacement),
                insert_random_facts=bool(args.insert_random_facts),
                maximize_family_variety=bool(args.maximize_family_variety),
                avoid_repeat_subproblems=bool(args.avoid_repeat_subproblems),
                family_quota_mode=str(args.family_quota_mode),
                per_family_quota=(None if args.per_family_quota is None else int(args.per_family_quota)),
            )
        except Exception as e:
            print(f"Error building composites: {e}", file=sys.stderr)
            return 3

        try:
            write_output(args.output, comps, fmt=args.format, indent=int(args.indent))
        except Exception as e:
            print(f"Error writing output: {e}", file=sys.stderr)
            return 4

        print(f"Wrote {len(comps)} composite items to {args.output}")
        # Also write the companion lite file
        try:
            outp = Path(args.output)
            out_lite_path = str(outp.parent / f"{outp.stem}_lite{(outp.suffix or '.json')}")
            write_lite_output(out_lite_path, comps, fmt=args.format, indent=int(args.indent))
            print(f"Wrote lite file to {out_lite_path}")
        except Exception as e:
            print(f"Error writing lite output: {e}", file=sys.stderr)
            return 4
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
