
"""
algebra_dataset_generator.py
----------------------------

AGSM alignment (Ambient Grade School Math)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This module implements the first stages of the AGSM pipeline:

- Programmatic algebra generation with SymPy (ground truth equations + solutions)
- Language realization with an LLM (equation → word problem)
- Round‑trip verification (word problem → equations → solution) to ensure fidelity

It outputs verified, single‑answer word problems across multiple algebraic families
(linear 1‑var, linear 2‑var systems, quadratics including non‑square discriminants,
rational equations with domain guards, and logarithmic/exponential forms).

Difficulty scaling in AGSM is provided by deterministic composition of verified
sub‑problems into multi‑step tasks (see `algebra_dataset_generator_deterministic_composite.py`).
This file focuses on generating and verifying the leaf problems that feed those
deterministic composites.

Basic usage
~~~~~~~~~~~
Synthetic dataset generator for algebra word problems using SymPy as the source of
truth and optional LLM rationalization (Ambient or Together) to turn equations into
conventional word problems with round‑trip verification.

Quick start (create venv and run):
  python3 -m venv .venv && source .venv/bin/activate
  pip install -U pip && pip install -r requirements.txt
  # Ambient (DeepSeek R1), strict equivalence verification
  python algebra_dataset_generator.py --per_family 3 --seed 42 --use_llm \
      --provider ambient --model deepseek-ai/DeepSeek-R1 \
      --verify-mode equivalence --parse-retries 10 --parse-retry-delay 5 \
      --format json --out algebraTest/algebra_synth_ambient.json
  # Together (DeepSeek R1), numeric verification
  python algebra_dataset_generator.py --per_family 3 --seed 42 --use_llm \
      --provider together --model deepseek-ai/DeepSeek-R1 \
      --verify-mode numeric --numeric-tol 1e-6 \
      --format json --out algebraTest/algebra_synth_together.json

Topic diversification (optional):
  Add a coherent real‑world domain using a hierarchical topic path sampled from
  Encyclopedia70K. The topic path is passed to the LLM as context and saved in
  each record (topic_path/topic_name).
  Example:
  python algebra_dataset_generator.py --per_family 3 --seed 42 --use_llm \
      --provider together --model deepseek-ai/DeepSeek-R1 \
      --use-encyclopedia-topics --encyclopedia-file Encyclopedia70K_20250810_174254.json \
      --verify-mode robustOptimal --format json --out algebraTest/

  # Post hoc rationalization (pretty JSON by default, verification disabled unless overridden)
  python algebra_dataset_generator.py --per_family 3 --seed 42 --use_llm \
      --provider ambient --model deepseek-ai/DeepSeek-R1 \
      --mode posthoc --parse-retries 10 --parse-retry-delay 5 \
      --out algebraTest/posthoc_dataset.json

Keys:
  - Ambient: AMBIENT_API_KEY or ambient_api_key.txt (or ~/.ambient_api_key)
  - Together: TOGETHER_API_KEY or togetherai_api_key.txt (or ~/.together_api_key)
  - OpenAI: OPENAI_API_KEY or openai_api_key.txt (or ~/.openai_api_key)

CLI flags of interest:
  --use_llm                 Enable LLM rationalization (default off)
  --provider [ambient|together|openai|google]  LLM provider (default ambient)
  --model MODEL             Provider model (default deepseek-ai/DeepSeek-R1)
  --verify-mode [equivalence|numeric|robustOptimal|none]
                            Round‑trip verification strategy (default equivalence)
  --numeric-tol FLOAT       Tolerance for numeric verification (default 1e-6)
  --allow-nonunique         Allow non‑unique original solutions in equivalence mode
  --parse-retries INT       Re‑ask parser on invalid JSON up to N times (default 0)
  --parse-retry-delay SEC   Delay between parse retries (default 5.0)
  --no-progress             Hide progress bar/ETA
  --format [jsonl|json]     Output format: JSONL (one object per line) or pretty JSON array (default jsonl)
  --indent INT              Pretty JSON indent (used when --format json, default 2)
  --mode [standard|posthoc] Generation mode; posthoc adds item_a..d fields and a reasoning trace
  --jobs N                  Run up to N worker processes in parallel (max 10)
  --trace-max-tokens N      Max tokens for reasoning trace (default 18000)
  --trace-temperature F     Temperature for reasoning trace (default 0.2)
  --save-roundtrip          Save story→equations parser JSON/raw output in each record (default off)
  --student-solve N         Ask LLM to solve each problem N times; record reasoning and match (default 0)
  --use-encyclopedia-topics Use hierarchical topics to diversify scenarios (default off)
  --encyclopedia-file PATH  JSON topic tree (default ./Encyclopedia70K_20250810_174254.json)

Output path behavior:
  --out may be a file or a directory. If a directory is provided, a timestamped
  filename is auto‑generated (algebra_synth_{provider}_{verifyMode}_YYYYMMDD_HHMMSS.{json|jsonl}).
  --save-roundtrip          Save parser JSON/raw output from story→equations in each record (default off)
  --student-solve N         Ask LLM to solve each problem N times; record reasoning, final answers, and match (default 0)
  --use-encyclopedia-topics Use hierarchical topics to diversify scenarios (default off)
  --encyclopedia-file PATH  JSON file containing topic tree (default ./Encyclopedia70K_20250810_174254.json)

Verification modes:
  equivalence
    - What it does: Parses the generated story back into SymPy equations via the LLM, then
      substitutes the original SymPy ground‑truth solution into those parsed equations and
      requires every equation to reduce to zero (exact symbolic residual check).
    - When to use: Highest precision; best with models that follow instructions tightly
      (e.g., Ambient + DeepSeek‑R1). Use when you want strict equation‑level equivalence.
    - Notes: By default requires a unique original solution (override with --allow-nonunique).
      Common failures: parser returns no equations, minor algebraic drift that changes structure,
      or multi‑solution sources.

  numeric
    - What it does: Solves the parsed equations numerically (over ℝ) and compares the solution(s)
      against the original SymPy ground truth within a tolerance (--numeric-tol, default 1e‑6).
      For one variable, compares candidate roots; for multi‑variable systems, compares component‑wise.
    - When to use: More tolerant to benign algebraic re‑arrangements; recommended with Together
      (DeepSeek‑R1) where JSON/structure can vary but numeric answers are stable.
    - Notes: Still fails if the parser changes coefficients/constants or produces a different
      solution set. Tolerance only covers numeric noise, not incorrect numbers.

  robustOptimal
    - What it does: Hybrid gate to resist coefficient/constant drift. Attempts a structural
      proportional‑identity match between each parsed equation and an original one using random
      real assignments (accepts global scalar multiples and reordering). Records a number‑literal
      fidelity signal but does not fail solely on it. Falls back to the numeric solution check
      if structural matching fails.
    - When to use: Default for mixed families and noisier parsers; stricter than numeric while
      remaining tolerant to benign algebraic transformations (e.g., multiplying both sides by −1).
    - Notes: For non‑polynomial equations with domain guards (rational/log/exp), the structural
      check skips singular points. If both structural and numeric checks fail, the item is rejected.

  none
    - What it does: Skips round‑trip verification entirely and accepts items as‑is.
    - When to use: Fast preview or post‑hoc rationalization mode where you only want problem text
      and traces without gating. Not recommended for building high‑precision training corpora.
    - Notes: Dataset records will mark verification as disabled. Consider enabling at least numeric
      checks before using outputs for training.

Problem families implemented:
  - linear_1var
  - linear_2var_system
  - quadratic (standard form)
  - quadratic_complete_square (requires completing the square)
  - rational_equation (with domain filters to avoid zero denominators)
  - logexp (logarithmic/exponential with real-domain constraints)

The generator creates JSONL with the following fields per record:
  id, family, problem_text, variables, domain, eq_system_str, eq_system_ast,
  sympy_src, solution, solution_eval, difficulty_meta,
  roundtrip_verified, verifier_feedback, topic_path, topic_name

Additional optional fields (when enabled):
- problem_text_reasoning: stripped <think> content from the story LLM.
- roundtrip_parser_json/raw/reasoning: artifacts from the story→equations parse (with --save-roundtrip).
- student_attempts / student_success_count: N solver attempts and match stats (with --student-solve N)., topic_path, topic_name

If `use_llm=True`, a provider client (Ambient or Together) is used for
equation->story and (optionally) story->equation round‑trip verification.
Otherwise a simple templater is used.

NOTE: This module is offline-friendly for preview; LLM calls require network.

Benchmarking generated datasets
-------------------------------
- Use `algebra_dataset_benchmarking_tool.py` (AGSM harness) to evaluate provider/model accuracy over difficulty tiers.
- The harness expects strict JSON `{"final_answer": ...}` but includes tolerant parsing fallbacks and grades by percent error (default exact match). It can produce both ground‑truth and consensus‑graded charts.
- Artifacts live under `algebraTest/benchmark_runs/<RUN_NAME>/` with a `run_meta.json` capturing the dataset `base_name` for reproducibility.
- Graphs include reference baselines per model: a naïve d=1 curve and a WLS‑fit curve `p^d` with confidence bands, enabling “expected vs actual” comparisons as described in the AGSM writeup.




"""
import json
import os
import random
import uuid
import time
import sys
import re
from dataclasses import dataclass, asdict, field
import time
import socket
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import sympy as sp
from sympy import Eq, symbols, S
from sympy.parsing.sympy_parser import parse_expr
from collections import Counter

# ---- LLM clients (optional) ----
try:
    from ambient_api_client_v3 import AmbientAPIClientV3, create_ambient_client
except Exception:
    AmbientAPIClientV3 = None
    def create_ambient_client(api_key: str = None):
        raise RuntimeError("Ambient client not available in this environment.")

# Together client (optional)
try:
    from together_client import TogetherClient as _TogetherClient
except Exception:
    _TogetherClient = None

# OpenAI client (optional)
try:
    from openai_client import OpenAIClient as _OpenAIClient
except Exception:
    _OpenAIClient = None

# Grok client (optional)
try:
    from grok_client import GrokClient as _GrokClient
except Exception:
    _GrokClient = None


# -------------------------- Utilities --------------------------

def int_nonzero(low: int, high: int, forbid: set = None) -> int:
    """Sample a nonzero integer in [low, high] not in forbid."""
    forbid = forbid or set()
    while True:
        x = random.randint(low, high)
        if x != 0 and x not in forbid:
            return x

def choose_distinct(low: int, high: int, k: int) -> List[int]:
    vals = set()
    while len(vals) < k:
        vals.add(int_nonzero(low, high, forbid=vals))
    return list(vals)

def unique_solution_dict(sol_set: Any, varsyms: List[sp.Symbol]) -> Optional[Dict[str, Any]]:
    """
    Try to coerce a SymPy solution object into a unique mapping {var: value}.
    Returns None if not uniquely solvable.
    """
    # solveset might return FiniteSet({val}), Tuple of FiniteSets, or dicts from solve()
    try:
        if isinstance(sol_set, dict):
            # solve(...) form
            if all(v.is_number or v.is_Rational or v.is_Integer for v in sol_set.values()):
                return {str(k): sp.nsimplify(v) for k, v in sol_set.items()}
            return None
        if hasattr(sol_set, 'subs'):
            # Most SymPy sets have .subs; try to extract for 1 variable
            if len(varsyms) == 1:
                var = varsyms[0]
                if sol_set.is_FiniteSet and len(sol_set) == 1:
                    v = list(sol_set)[0]
                    return {str(var): sp.nsimplify(v)}
                # Otherwise give up
                return None
    except Exception:
        return None
    return None

def eval_solution(solution: Dict[str, Any]) -> Dict[str, float]:
    out = {}
    for k, v in solution.items():
        try:
            if isinstance(v, (list, tuple)):
                out[k] = [float(sp.N(t)) for t in v]
            else:
                out[k] = float(sp.N(v))
        except Exception:
            out[k] = None
    return out

def eqs_to_strings(eqs: List[sp.Eq]) -> List[str]:
    return [str(eq) for eq in eqs]

def serialize_eqs_ast(eqs: List[sp.Eq]) -> List[str]:
    # Stringified s-expr via sympy srepr for reproducibility
    return [sp.srepr(eq) for eq in eqs]


# -------------------------- Generators --------------------------

@dataclass
class ProblemArtifacts:
    family: str
    eqs: List[sp.Eq]
    variables: Dict[str, str]     # semantic labels per symbol
    domain: str                   # 'R' or 'Z' etc.
    solution: Dict[str, Any]
    sympy_src: str
    difficulty_meta: Dict[str, Any] = field(default_factory=dict)

def generate_linear_1var() -> ProblemArtifacts:
    # Choose integer solution x0 and linear coefficients
    x = symbols('x', real=True)
    x0 = int_nonzero(-12, 12)
    m = int_nonzero(-9, 9)
    b = random.randint(-15, 15)
    # Equation m*x + b = m*x0 + b
    rhs = m*x0 + b
    eq = Eq(m*x + b, rhs)
    # Ensure unique solution
    sol = sp.solveset(eq, x, domain=S.Reals)
    assert sol.is_FiniteSet and len(sol) == 1
    solution = { 'x': sp.nsimplify(list(sol)[0]) }
    sym_src = f"from sympy import symbols, Eq, solveset, S\nx = symbols('x', real=True)\neq = Eq({m}*x + {b}, {rhs})\nsol = solveset(eq, x, domain=S.Reals)"
    return ProblemArtifacts(
        family="linear_1var",
        eqs=[eq],
        variables={"x": "unknown quantity"},
        domain="R",
        solution=solution,
        sympy_src=sym_src,
        difficulty_meta={"coeff_m": m, "const_b": b}
    )

def generate_linear_2var_system() -> ProblemArtifacts:
    x, y = symbols('x y', real=True)
    # Choose integer solution (x0, y0)
    x0, y0 = choose_distinct(-8, 8, 2)
    # Choose invertible 2x2 integer matrix A and compute b = A*[x0, y0]
    a11, a12, a21, a22 = choose_distinct(-6, 6, 4)
    det = a11*a22 - a12*a21
    while det == 0:
        a11, a12, a21, a22 = choose_distinct(-6, 6, 4)
        det = a11*a22 - a12*a21
    b1 = a11*x0 + a12*y0
    b2 = a21*x0 + a22*y0
    eq1 = Eq(a11*x + a12*y, b1)
    eq2 = Eq(a21*x + a22*y, b2)
    sol = sp.solve([eq1, eq2], (x, y), dict=True)
    assert len(sol) == 1
    solution = { 'x': sp.nsimplify(sol[0][x]), 'y': sp.nsimplify(sol[0][y]) }
    sym_src = f"from sympy import symbols, Eq, solve\nx,y = symbols('x y', real=True)\neq1 = Eq({a11}*x + {a12}*y, {b1})\neq2 = Eq({a21}*x + {a22}*y, {b2})\nsol = solve([eq1, eq2], (x,y))"
    return ProblemArtifacts(
        family="linear_2var_system",
        eqs=[eq1, eq2],
        variables={"x": "first unknown", "y": "second unknown"},
        domain="R",
        solution=solution,
        sympy_src=sym_src,
        difficulty_meta={"det": det}
    )

def generate_quadratic(complete_square_required: bool=False) -> ProblemArtifacts:
    x = symbols('x', real=True)
    if complete_square_required:
        # Create a quadratic with non-square discriminant (requires completing the square/ quadratic formula)
        while True:
            a = random.choice([1, 1, 2, 3])
            b = int_nonzero(-14, 14)
            c = random.randint(-20, 20)
            D = b*b - 4*a*c
            if D > 0:
                s = int(D**0.5)
                if s*s != D:
                    break
        eq = Eq(a*x**2 + b*x + c, 0)
        sol = sp.solve(eq, x)
        solution = {'x': [sp.nsimplify(v) for v in sol]}
        sym_src = (
            f"from sympy import symbols, Eq, solve\n"
            f"x = symbols('x', real=True)\n"
            f"eq = Eq({a}*x**2 + {b}*x + {c}, 0)\n"
            f"sol = solve(eq, x)"
        )
        return ProblemArtifacts(
            family="quadratic_complete_square",
            eqs=[eq],
            variables={"x": "unknown quantity"},
            domain="R",
            solution=solution,
            sympy_src=sym_src,
            difficulty_meta={"a": a, "b": b, "c": c, "disc": D, "complete_square_req": True}
        )
    # Factorable case from integer roots
    if random.random() < 0.5:
        r1 = int_nonzero(-9, 9)
        r2 = r1  # perfect square
    else:
        r1, r2 = choose_distinct(-9, 9, 2)
    a = random.choice([1, 1, 1, 2, 3])  # bias to monic
    poly = a*(x - r1)*(x - r2)
    eq = Eq(sp.expand(poly), 0)
    sol = sp.solve(eq, x)
    solution = {'x': [sp.nsimplify(s) for s in sol]}
    sym_src = f"from sympy import symbols, Eq, solve\nx = symbols('x', real=True)\neq = Eq({sp.expand(poly)}, 0)\nsol = solve(eq, x)"
    return ProblemArtifacts(
        family="quadratic",
        eqs=[eq],
        variables={"x": "unknown quantity"},
        domain="R",
        solution=solution,
        sympy_src=sym_src,
        difficulty_meta={"a": a, "roots": [r1, r2], "perfect_square": r1==r2, "complete_square_req": False}
    )

def generate_rational_equation() -> ProblemArtifacts:
    x = symbols('x', real=True)
    # Build something like A/(x - p) + B/(x + q) = C
    p, q = choose_distinct(-10, 10, 2)
    A, B = choose_distinct(-8, 8, 2)
    C = int_nonzero(-10, 10)
    eq = Eq(A/(x - p) + B/(x + q), C)
    # Solve and filter extraneous roots (denominator != 0)
    sol = sp.solveset(eq, x, domain=S.Reals)
    # Remove values that violate domain
    sol_vals = []
    for v in sol:
        if v != p and v != -q:
            sol_vals.append(sp.nsimplify(v))
    # Require at least one valid solution
    if not sol_vals:
        return generate_rational_equation()
    solution = {'x': sol_vals if len(sol_vals) > 1 else sol_vals[0]}
    sym_src = f"from sympy import symbols, Eq, solveset, S\nx = symbols('x', real=True)\neq = Eq({A}/(x - {p}) + {B}/(x + {q}), {C})\nsol = solveset(eq, x, domain=S.Reals)"
    return ProblemArtifacts(
        family="rational_equation",
        eqs=[eq],
        variables={"x": "unknown real satisfying domain constraints (denominators non-zero)"},
        domain="R",
        solution=solution,
        sympy_src=sym_src,
        difficulty_meta={"p": p, "q": q, "A": A, "B": B, "C": C}
    )

def generate_logexp() -> ProblemArtifacts:
    x = symbols('x', real=True)
    # Choose between a simple log or exp form
    if random.random() < 0.5:
        # log(a*x + b, base) = k  -> a*x + b = base**k
        base = random.choice([2, 3, 5, 10])
        a = int_nonzero(-6, 6)
        b = int_nonzero(-10, 10)
        k = int_nonzero(-3, 3)
        # Ensure argument of log is positive for the chosen solution domain:
        # We can select a target solution x0 and backsolve b so that a*x0 + b > 0
        x0 = int_nonzero(-6, 6)
        # Set RHS value and ensure positivity constraints hold
        rhs_val = base**k
        # Construct equation: log(a*x + b, base) = k
        # Enforce a*x0 + b = rhs_val  => b = rhs_val - a*x0
        b = int(rhs_val - a*x0)
        eq = Eq(sp.log(a*x + b, base), k)
        # Solve over reals
        sol = sp.solveset(eq, x, domain=S.Reals)
        # Filter domain: a*x + b > 0
        valid = []
        for v in sol:
            if (a*v + b).subs(x, v).evalf() > 0:
                valid.append(sp.nsimplify(v))
        if not valid:
            return generate_logexp()
        solution = {'x': valid if len(valid) > 1 else valid[0]}
        sym_src = f"from sympy import symbols, Eq, solveset, S, log\nx = symbols('x', real=True)\neq = Eq(log({a}*x + {b}, {base}), {k})\nsol = solveset(eq, x, domain=S.Reals)"
        fam = "log_equation"
        variables = {"x": "unknown real (log argument positive)"}
    else:
        # exp(kx + b) = m  -> solve for x
        k = int_nonzero(-5, 5)
        b = random.randint(-5, 8)
        m = random.choice([2,3,4,5,6,7,8,9,10])
        eq = Eq(sp.exp(k*x + b), m)
        sol = sp.solveset(eq, x, domain=S.Reals)
        solution = {'x': [sp.nsimplify(v) for v in sol]}  # typically single value: (log(m) - b)/k
        sym_src = f"from sympy import symbols, Eq, solveset, S, exp\nx = symbols('x', real=True)\neq = Eq(exp({k}*x + {b}), {m})\nsol = solveset(eq, x, domain=S.Reals)"
        fam = "exp_equation"
        variables = {"x": "unknown real"}
    return ProblemArtifacts(
        family=fam,
        eqs=[eq],
        variables=variables,
        domain="R",
        solution=solution,
        sympy_src=sym_src,
        difficulty_meta={}
    )


# -------------------------- LLM Rationalizer --------------------------

@dataclass
class RationalizationConfig:
    style: str = "conventional"  # 'conventional' word-problem
    require_roundtrip: bool = True
    temperature: float = 0.2
    max_tokens: int = 50000
    parse_retries: int = 0
    parse_retry_delay: float = 5.0
    trace_max_tokens: int = 50000
    trace_temperature: float = 0.2

STORY_SYSTEM_PROMPT = """You are a precise math problem writer.
Write a SINGLE real-world algebra word problem (no extra commentary) that exactly corresponds to the given variables and equations.
Requirements:
- Explicitly state the meaning of each variable (e.g., "Let x be ...").
- Use the exact numbers from the equations verbatim; do not introduce new numbers or rephrase them.
- If domain constraints are relevant (e.g., denominators nonzero, log arguments positive), mention them briefly.
- Plain text only: no Markdown headings, lists, or LaTeX.
Return ONLY the problem statement.
"""

STORY_USER_TEMPLATE = """Create a word problem for the following symbolic specification.

Variables (use these exact semantics):
{var_semantics}

Equations (SymPy form):
{eqs}

Domain: {domain}

{topic_block}

Constraints:
- The problem must have a UNIQUE correct solution consistent with the domain.
- The numbers in the story MUST correspond to the equation coefficients/constants (no changes).
- Avoid giving the final answer; state only the givens and the question.
Write the problem now:
"""

NATURAL_STORY_SYSTEM_PROMPT = """You are a precise math problem writer.
Write a SINGLE realistic word problem that implies relationships naturally — never state equations or templates.
Hard constraints:
- Define each variable in plain language (e.g., "Let x be the delivery delay in hours").
- Use every given number exactly as-is; do not invent new numbers.
- Do NOT include equations, math symbols, or operator words: no '=', '+', '-', '×', '/', 'Eq(', 'equation', 'formula', 'expression'. Also avoid: 'equals', 'squared', 'times', 'solve', 'satisfies'.
- Embed numbers into natural roles (rates, fees/waivers, totals, counts, times, distances, measurements). Express negatives as credits/shortfalls/refunds.
- Prefer narrative patterns per relation type:
  • Linear (a*x + b = c): per‑unit rate (a per …), plus/minus fixed fee/waiver (b), total/credit (c).
  • Quadratic (x^2 + b*x = c): rectangle sides x and x+b with area c; avoid saying 'squared'.
  • Rational (A/(x±k) + B/(x±m) = C): combined/parallel rates using 'per' or 'over' (e.g., '6 per (x+8) hours').
  • Logarithmic (log_base_k(...)=r): explicitly say 'logarithmic (base k)' or 'log base k', and make the 'logarithmic reading' of the processed measure equal to r; do NOT state that the processed measure itself equals r.
  • Exponential: a growth/decay 'level reads m at time x' without formulas.
- 2–4 sentences, realistic units, and exactly one explicit question about the target variable(s).
- Plain ASCII only: no LaTeX/Markdown/code fences.

Bad→good examples:
- Bad: "8 times the delivery delay minus 12 equals −92." → Good: "A carrier charges $8 per hour of delay. After a $12 waiver, your statement shows a $92 credit. Let x be the delay (hours). What is x?"
- Bad: "x^2 + 5x − 24 = 0." → Good: "A rectangular plot has length x meters and width (x+5) meters. Its area is 24 m^2. Let x be the length. What is x?"
- Bad: "−6/(x+8) − 1/(x+3) = 7." → Good: "Two valves drain a tank. Set to x, one removes 6 units over (x+8) hours and another 1 unit over (x+3) hours. Together the gauge drops by 7. What is x?"
- Bad: "log_2(2x+10) = −3." → Good: "A sensor doubles x and then adds 10 units. On a logarithmic scale with base 2, the logarithmic reading of that quantity is −3. Let x be the raw input. What is x?"
- Bad: "e^(5−3x) = 4." → Good: "A sample’s level after x hours follows a standard decay. At that time the meter reads 4, starting near 5 with a 3‑per‑hour factor. What is x?"

Self-check silently before output:
- If any sentence could be read as an equation (e.g., 'A times x plus B equals C'), rewrite with rates/fees/totals/scale.
- Logarithmic self-check: confirm your wording implies 'log base k of [quantity] equals r' (i.e., the 'logarithmic reading' equals r) and not '[quantity] equals r'.
Output ONLY the final story.
"""

NATURAL_STORY_USER_TEMPLATE = """Create a word problem for the following specification.

Variables (use these exact names and define them naturally):
{var_semantics}

Relations (numbers to incorporate as natural facts; do NOT restate or paraphrase the equations):
{eqs}

Domain: {domain}

{topic_block}

Constraints:
- Translate each relation into realistic narrative facts; do not hint at algebraic structure or restate operations.
- Use the exact numbers once each in natural roles (rate, fee/waiver, total/credit, time, distance, count). For reciprocals, use 'per' or 'over' (e.g., '6 per (x+8) hours').
- Avoid math jargon and operator words (no '=', '+', '-', 'Eq(', 'expression', 'equation', 'squared', 'times', 'solve', 'satisfies').
- Ask one clear question about the target variable(s).
- Length target: 45–90 words.
Write the problem now:
"""

# Round-trip: story -> equations (as JSON) for automatic checking
PARSE_SYSTEM_PROMPT = """You convert algebra word problems into explicit variables and equations in SymPy form.
Output ONLY compact JSON matching this schema:
{"variables": {"x": "semantic description", "...": "..."}, "equations": ["Eq(...)", "Eq(...)", "..."]}
Constraints and conventions:
- Interpret negation from natural phrasing precisely:
  • credit/refund/shortfall/deficit → negative totals (e.g., "a $52 credit" → -52).
  • charge/due/invoice total → positive totals.
  • "drops by k" or "decreases by k" → -k; "rises/increases by k" → +k.
- Return ONLY JSON; no code fences, commentary, or extra text.
Make sure equations are valid SymPy and solve to a single solution over the reals."""

PARSE_USER_TEMPLATE = """Extract variables and equations from the following word problem.
Problem:
{problem_text}
Return JSON now:
"""

def build_chat_prompt(system: str, user: str) -> str:
    # The Ambient client expects a string prompt; we'll concatenate role tags.
    return f"<SYSTEM>\n{system}\n</SYSTEM>\n<USER>\n{user}\n</USER>"

class WordProblemRationalizer:
    def __init__(self, client: Optional[object] = None, cfg: RationalizationConfig = None):
        self.client = client
        self.cfg = cfg or RationalizationConfig()
        # Optional logging setup provided by DatasetGenerator
        self.llm_log_path: Optional[str] = None
        # Optional multiprocessing log queue; when set, entries are sent to the queue
        # and the parent process is responsible for writing a single coherent JSONL.
        self.llm_log_queue: Optional[Any] = None
        # Optional extra context (e.g., run_id, job_id) merged into each log entry
        self.log_extra: Optional[Dict[str, Any]] = None
        self.provider: Optional[str] = None

    def _log_llm(self, entry: Dict[str, Any]) -> None:
        try:
            # Merge extra context and add pid for debugging
            merged = dict(entry)
            try:
                import os as _os
                merged.setdefault("pid", _os.getpid())
            except Exception:
                pass
            extra = getattr(self, 'log_extra', None)
            if isinstance(extra, dict):
                for k, v in extra.items():
                    merged.setdefault(k, v)
            # Prefer queue sink when available to keep a single writer
            q = getattr(self, 'llm_log_queue', None)
            if q is not None:
                try:
                    q.put(merged)
                    return
                except Exception:
                    pass
            path = getattr(self, 'llm_log_path', None)
            if not path:
                return
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open('a', encoding='utf-8') as f:
                f.write(json.dumps(merged, ensure_ascii=False, indent=2) + "\n")
                try:
                    f.flush()
                except Exception:
                    pass
        except Exception:
            pass

    def _build_curl(self, provider: Optional[str], prompt: str, model: Optional[str], max_tokens: int, temperature: float) -> str:
        prov = (provider or '').lower()
        if prov == 'ambient':
            try:
                base_url = getattr(self.client, 'base_url', 'https://api.ambient.xyz/v1')
            except Exception:
                base_url = 'https://api.ambient.xyz/v1'
            url = f"{base_url.rstrip('/')}/chat/completions"
            auth_env = "$AMBIENT_API_KEY"
        elif prov == 'together':
            url = 'https://api.together.xyz/v1/chat/completions'
            auth_env = "$TOGETHER_API_KEY"
        elif prov == 'openai':
            url = 'https://api.openai.com/v1/chat/completions'
            auth_env = "$OPENAI_API_KEY"
        else:
            url = 'https://api.together.xyz/v1/chat/completions'
            auth_env = "$TOGETHER_API_KEY"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        body = json.dumps(payload, ensure_ascii=False, indent=2)
        curl = (
            f"curl -sS -X POST \"{url}\" \\\n+ -H \"Authorization: Bearer {auth_env}\" \\\n+ -H \"Content-Type: application/json\" \\\n+ -d \"{body_sh}\""
        )
        return curl

    def _complete(self, system: str, user: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None, tag: Optional[str] = None) -> str:
        if self.client is None:
            raise RuntimeError("Ambient client not provided; set use_llm=True with a configured client.")
        prompt = build_chat_prompt(system, user)
        mx = int(max_tokens if max_tokens is not None else self.cfg.max_tokens)
        tp = float(temperature if temperature is not None else self.cfg.temperature)
        meta_req = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tag": tag or "complete",
            "provider": getattr(self, 'provider', None),
            "model": getattr(self.client, 'model', None),
            "max_tokens": mx,
            "temperature": tp,
            "system": system,
            "user": user,
            "prompt": prompt,
        }
        resp_text = ""
        error_text = None
        is_timeout = False
        curl_cmd = None
        started = time.time()
        try:
            resp_text = self.client.complete(prompt, max_tokens=mx, temperature=tp)
            # Capture provider-native reasoning content when available
            try:
                rc = getattr(self.client, 'last_reasoning_content', None)
                if not rc:
                    rc = getattr(self.client, 'last_reasoning', None)
                self.last_reasoning_content = rc
            except Exception:
                self.last_reasoning_content = None
            return resp_text
        except Exception as e:
            error_text = str(e)
            # Timeout detection across common error types/messages (include client last error text)
            try:
                # requests timeouts
                try:
                    import requests  # type: ignore
                    if isinstance(e, getattr(requests, 'exceptions', object).__dict__.get('Timeout', ())) or \
                       isinstance(e, getattr(requests, 'exceptions', object).__dict__.get('ReadTimeout', ())) or \
                       isinstance(e, getattr(requests, 'exceptions', object).__dict__.get('ConnectTimeout', ())):
                        is_timeout = True
                except Exception:
                    pass
                # socket timeout or generic message; also check client.last_error_text if present
                client_last_err = None
                try:
                    client_last_err = getattr(self.client, 'last_error_text', None)
                except Exception:
                    client_last_err = None
                err_text_all = " ".join([t for t in [error_text, client_last_err] if t])
                if isinstance(e, socket.timeout) or ('timed out' in err_text_all.lower()) or ('timeout' in err_text_all.lower()):
                    is_timeout = True
            except Exception:
                pass
            # Always include a cURL command for errors to aid reproduction
            try:
                curl_cmd = self._build_curl_here(getattr(self, 'provider', None), prompt, getattr(self.client, 'model', None), mx, tp)
            except Exception:
                curl_cmd = None
            raise
        finally:
            log_entry = {**meta_req, "response": resp_text, "error": error_text, "elapsed_sec": (time.time() - started)}
            # Include provider-native reasoning content if present
            try:
                rc = getattr(self, 'last_reasoning_content', None)
            except Exception:
                rc = None
            if rc:
                log_entry["reasoning_content"] = rc
            if is_timeout:
                log_entry["timeout"] = True
            if error_text:
                log_entry["curl"] = curl_cmd
            self._log_llm(log_entry)

    def _build_curl_here(self, provider: Optional[str], prompt: str, model: Optional[str], max_tokens: int, temperature: float) -> str:
        prov = (provider or '').lower()
        if prov == 'ambient':
            try:
                base_url = getattr(self.client, 'base_url', 'https://api.ambient.xyz/v1')
            except Exception:
                base_url = 'https://api.ambient.xyz/v1'
            url = f"{base_url.rstrip('/')}/chat/completions"
            auth_env = "$AMBIENT_API_KEY"
        elif prov == 'together':
            url = 'https://api.together.xyz/v1/chat/completions'
            auth_env = "$TOGETHER_API_KEY"
        elif prov == 'openai':
            url = 'https://api.openai.com/v1/chat/completions'
            auth_env = "$OPENAI_API_KEY"
        else:
            url = 'https://api.together.xyz/v1/chat/completions'
            auth_env = "$TOGETHER_API_KEY"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        body = json.dumps(payload, ensure_ascii=False, indent=2)
        curl = (
            f"curl -sS -X POST \"{url}\" \\\n+  -H \"Authorization: Bearer {auth_env}\" \\\n+  -H \"Content-Type: application/json\" \\\n+  --data-binary @- <<'JSON'\n{body}\nJSON"
        )
        return curl

    def equation_to_story(self, spec: ProblemArtifacts, topic_path: Optional[List[str]] = None) -> str:
        var_semantics = "\n".join([f"- {k}: {v}" for k, v in spec.variables.items()])
        eqs = "\n".join([str(e) for e in spec.eqs])
        if topic_path:
            topic_str = " / ".join([t for t in topic_path if t])
            topic_block = (
                "Topic context (use this real-world domain; do NOT add numbers beyond the equations):\n"
                f"{topic_str}\n"
            )
        else:
            topic_block = "(No specific topic context; choose a realistic everyday setting.)"
        style = (getattr(self.cfg, 'style', 'natural') or 'natural').lower()
        if style == 'natural':
            user = NATURAL_STORY_USER_TEMPLATE.format(var_semantics=var_semantics, eqs=eqs, domain=spec.domain, topic_block=topic_block)
            # Family-specific guidance for logarithmic relations to avoid equality misinterpretation
            try:
                if spec.family == "log_equation" and spec.eqs:
                    eq0 = spec.eqs[0]
                    lhs = getattr(eq0, 'lhs', None)
                    rhs = getattr(eq0, 'rhs', None)
                    base_str = None
                    k_str = None
                    if lhs is not None and getattr(lhs, 'func', None) == sp.log:
                        args = list(getattr(lhs, 'args', []) or [])
                        if len(args) >= 2:
                            base_str = str(args[1])
                    if rhs is not None:
                        k_str = str(rhs)
                    if base_str and k_str:
                        user += (
                            "\nFamily-specific guidance (logarithmic):\n"
                            f"- Explicitly say 'logarithmic (base {base_str})' or 'log base {base_str}'.\n"
                            f"- Make the 'logarithmic reading' of the processed measure equal to {k_str}; do NOT say the processed measure equals {k_str}.\n"
                            f"- Example: 'On a logarithmic scale with base {base_str}, the logarithmic reading of this measure is {k_str}.'\n"
                        )
            except Exception:
                # Non-fatal: proceed without the extra guidance if introspection fails
                pass
            return self._complete(NATURAL_STORY_SYSTEM_PROMPT, user, tag="equation_to_story:natural").strip()
        else:
            user = STORY_USER_TEMPLATE.format(var_semantics=var_semantics, eqs=eqs, domain=spec.domain, topic_block=topic_block)
            # Inject similar guidance for facts-style prompts when logarithmic family is used
            try:
                if spec.family == "log_equation" and spec.eqs:
                    eq0 = spec.eqs[0]
                    lhs = getattr(eq0, 'lhs', None)
                    rhs = getattr(eq0, 'rhs', None)
                    base_str = None
                    k_str = None
                    if lhs is not None and getattr(lhs, 'func', None) == sp.log:
                        args = list(getattr(lhs, 'args', []) or [])
                        if len(args) >= 2:
                            base_str = str(args[1])
                    if rhs is not None:
                        k_str = str(rhs)
                    if base_str and k_str:
                        user += (
                            "\nAdditional constraint (logarithmic): explicitly say 'logarithmic (base {base})' or 'log base {base}', and state that the 'logarithmic reading' equals {k}; do NOT equate the processed measure to {k}.\n"
                        ).format(base=base_str, k=k_str)
            except Exception:
                pass
            return self._complete(STORY_SYSTEM_PROMPT, user, tag="equation_to_story:facts").strip()

    def reasoning_trace(self, equations: list[str], problem_text: str, numeric_answer: Dict[str, float]) -> str:
        if self.client is None:
            raise RuntimeError("Ambient client not provided; set use_llm=True with a configured client.")
        sys_prompt = (
            "You are a meticulous algebra tutor. Given a word problem, its governing equation(s), "
            "and the numeric answer, produce a clear step-by-step reasoning trace."
        )
        user = (
            "Problem:\n" + problem_text.strip() + "\n\n" +
            "Equations (SymPy):\n" + "\n".join(equations) + "\n\n" +
            "Numeric answer (variables to values):\n" + json.dumps(numeric_answer) + "\n\n" +
            "Instructions:\n"
            "- Produce numbered steps (1., 2., 3., ...).\n"
            "- For each step, state what you do AND why it is valid (e.g., properties of equality).\n"
            "- Show substitutions and simplifications explicitly in plain ASCII.\n"
            "- Refer to the provided equations when using them.\n"
            "- End by restating the final numeric answer exactly.\n"
            "- Plain text only; no headings, no Markdown, no LaTeX.\n"
        )
        return self._complete(
            sys_prompt,
            user,
            max_tokens=int(self.cfg.trace_max_tokens),
            temperature=float(self.cfg.trace_temperature),
            tag="reasoning_trace",
        ).strip()

    def story_to_equations(self, problem_text: str) -> tuple[Dict[str, Any], str | None, str | None]:
        """Ask the LLM to parse the story to JSON with retries on invalid JSON.

        Returns (parsed_json, raw_text, reasoning_think_block).
        """
        user = PARSE_USER_TEMPLATE.format(problem_text=problem_text)

        def _attempt() -> tuple[str, str | None]:
            raw = self._complete(PARSE_SYSTEM_PROMPT, user, tag="parse").strip()
            think, stripped = extract_think_blocks(raw)
            # Prefer provider-native reasoning content when available (Ambient)
            try:
                rc = getattr(self, 'last_reasoning_content', None)
                if rc and not think:
                    think = rc
            except Exception:
                pass
            raw2 = (stripped or raw).strip().strip("```").strip()
            if not (raw2.startswith("{") and raw2.endswith("}")) and "{" in raw2 and "}" in raw2:
                s = raw2.find("{")
                e = raw2.rfind("}")
                if s != -1 and e != -1 and e > s:
                    raw2 = raw2[s:e+1]
            return raw2, think

        raw, think = _attempt()
        for i in range(int(self.cfg.parse_retries) + 1):
            try:
                return json.loads(raw), raw, think
            except Exception:
                if i >= int(self.cfg.parse_retries):
                    break
                time.sleep(max(0.0, float(self.cfg.parse_retry_delay)))
                raw, think = _attempt()
        raise ValueError(f"Round-trip parser returned non-JSON after retries: {raw[:160]}...")

    def story_to_equations_strict(self, problem_text: str, allowed_vars: list[str]) -> tuple[Dict[str, Any], str | None, str | None]:
        """Stricter parser that enforces exact variable names.

        - allowed_vars: e.g., ['x_1','x_2','z'] must be used verbatim in JSON.
        Returns (parsed_json, raw_text, reasoning_think_block).
        """
        # Build a stricter system prompt with explicit constraints + example
        example_vars = {v: f"semantic for {v}" for v in allowed_vars}
        example_eq = "Eq(" + "+".join([f"1*{v}" for v in (allowed_vars[:2] or ['x','y'])]) + ", 0)"
        strict_system = (
            PARSE_SYSTEM_PROMPT
            + "\nIMPORTANT:\n"
              "- Use these variable names EXACTLY (verbatim), including underscores: "
              + ", ".join(allowed_vars)
              + ".\n"
              "- Do not drop underscores or rename variables (e.g., do not write x1 for x_1).\n"
              "- Output must include only these variables; do not invent new variable names.\n"
              "- Example JSON (structure only, keep underscores):\n"
              + json.dumps({"variables": example_vars, "equations": [example_eq]}, ensure_ascii=False)
        )
        user = PARSE_USER_TEMPLATE.format(problem_text=problem_text)

        def _attempt() -> tuple[str, str | None]:
            raw = self._complete(strict_system, user, tag="parse_strict").strip()
            think, stripped = extract_think_blocks(raw)
            try:
                rc = getattr(self, 'last_reasoning_content', None)
                if rc and not think:
                    think = rc
            except Exception:
                pass
            raw2 = (stripped or raw).strip().strip("```").strip()
            if not (raw2.startswith("{") and raw2.endswith("}")) and "{" in raw2 and "}" in raw2:
                s = raw2.find("{")
                e = raw2.rfind("}")
                if s != -1 and e != -1 and e > s:
                    raw2 = raw2[s:e+1]
            return raw2, think

        raw, think = _attempt()
        for i in range(int(self.cfg.parse_retries) + 1):
            try:
                return json.loads(raw), raw, think
            except Exception:
                if i >= int(self.cfg.parse_retries):
                    break
                time.sleep(max(0.0, float(self.cfg.parse_retry_delay)))
                raw, think = _attempt()
        raise ValueError(f"Round-trip parser (strict) returned non-JSON after retries: {raw[:160]}...")

    def student_solve(self, problem_text: str, max_tokens: int = 50000, temperature: float | None = None) -> str:
        """Ask the model to solve the problem step-by-step and end with 'Final Answer: ...'."""
        if self.client is None:
            raise RuntimeError("Ambient client not provided; set use_llm=True with a configured client.")
        sys_prompt = (
            "You are a careful algebra solver. Solve the following problem step-by-step. "
            "On the last line, output only 'Final Answer: ' followed by concise values for the unknowns. "
            "Use plain text only (no LaTeX, no code fences)."
        )
        user = (
            "Problem:\n" + problem_text.strip() + "\n\n" +
            "Instructions:\n"
            "- Show the steps clearly and keep arithmetic explicit.\n"
            "- Use exact arithmetic when possible.\n"
            "- Last line MUST be: Final Answer: x = ..., y = ... (as needed).\n"
        )
        return self._complete(
            sys_prompt,
            user,
            max_tokens=int(max_tokens),
            temperature=(self.cfg.temperature if temperature is None else float(temperature)),
            tag="student_solve",
        ).strip()


def extract_think_blocks(text: str) -> tuple[str | None, str]:
    """Extract <think>...</think> blocks and return (thinking, stripped_text).
    - Concatenates multiple blocks if present (including tags in the thinking output).
    - If an orphan '<think>' appears without a closing tag, treat everything from the first
      '<think>' to the end as thinking and strip it from the returned content.
    """
    if not text:
        return None, text
    # Find all paired think blocks
    blocks = re.findall(r"<think>[\s\S]*?</think>", text)
    thinking = "\n".join(blocks) if blocks else None
    stripped = text
    if blocks:
        for b in blocks:
            stripped = stripped.replace(b, "")
    # Handle orphan case
    if "<think>" in stripped and "</think>" not in stripped:
        s = stripped.find("<think>")
        if s != -1:
            orphan = stripped[s:]
            thinking = (thinking + "\n" + orphan).strip() if thinking else orphan
            stripped = stripped[:s]
    return (thinking.strip() if isinstance(thinking, str) else thinking), stripped.strip()

def sanitize_plaintext(text: str) -> str:
    """Best-effort cleanup to keep story text plain ASCII without LaTeX/markup wrappers.
    - Remove LaTeX delimiters ($, $$, \( \), \[ \])
    - Drop code fences and stray backticks
    - Unescape common backslash escapes before punctuation
    """
    if not text:
        return text
    s = str(text)
    # Remove LaTeX math delimiters
    s = s.replace("$$", "").replace("$", "")
    # Remove simple code fences/backticks
    s = s.replace("```", "").replace("`", "")
    # Remove LaTeX inline delimiters and brackets
    s = s.replace("\\(", "").replace("\\)", "")
    s = s.replace("\\[", "").replace("\\]", "")
    # Replace common LaTeX commands with ASCII
    s = s.replace("\\cdot", "*")
    s = s.replace("\\times", "*")
    s = s.replace("\\neq", "!=")
    s = s.replace("\\log", "log")
    s = s.replace("\\exp", "exp")
    # Remove leading backslashes before parentheses/brackets/braces and equals
    s = re.sub(r"\\([(){}\[\]=])", r"\1", s)
    # Collapse excessive whitespace
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


# -------------------------- Dataset Builder --------------------------

@dataclass
class DatasetRecord:
    id: str
    family: str
    problem_text: str
    variables: Dict[str, str]
    domain: str
    eq_system_str: List[str]
    eq_system_ast: List[str]
    sympy_src: str
    solution: Dict[str, Any]
    solution_eval: Dict[str, float]
    difficulty_meta: Dict[str, Any]
    roundtrip_verified: Optional[bool]
    verifier_feedback: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    seed: Optional[int] = None
    verify_mode: Optional[str] = None
    verification_error: Optional[float] = None
    story_numbers_ok: Optional[bool] = None
    trace_retry_count: Optional[int] = None
    # Optional reasoning captures for Together provider
    problem_text_reasoning: Optional[str] = None
    # Topic diversification (optional)
    topic_path: Optional[List[str]] = None
    topic_name: Optional[str] = None
    # Round-trip parser artifacts (optional)
    roundtrip_parser_json: Optional[Dict[str, Any]] = None
    roundtrip_parser_raw: Optional[str] = None
    roundtrip_parser_reasoning: Optional[str] = None
    # Numeric diagnostics (always emitted when LLM verification is attempted)
    verifier_orig_numeric: Optional[Dict[str, Any]] = None
    verifier_parsed_numeric: Optional[Any] = None
    verifier_best_numeric_error: Optional[float] = None
    verifier_vars: Optional[List[str]] = None
    # Post-hoc fields (optional)
    item_a_equations: Optional[List[str]] = None
    item_b_numeric_answer: Optional[Dict[str, float]] = None
    item_c_problem: Optional[str] = None
    item_d_reasoning: Optional[str] = None
    item_c_problem_reasoning: Optional[str] = None
    item_d_reasoning_reasoning: Optional[str] = None
    item_d_reasoning_steps: Optional[List[str]] = None
    # Student-solve attempts (optional)
    student_attempts: Optional[List[Dict[str, Any]]] = None
    student_success_count: Optional[int] = None
    # Optional random fact about the chosen encyclopedia topic
    random_fact: Optional[str] = None
    # Overall wall time to generate this record (seconds)
    duration_sec: Optional[float] = None


def _to_jsonable(obj):
    if isinstance(obj, (int, float, str, type(None))):
        return obj
    # SymPy numbers
    try:
        import sympy as sp
        if isinstance(obj, (sp.Integer, sp.Rational, sp.Float)):
            # Prefer string to preserve exactness
            return str(obj)
    except Exception:
        pass
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return str(obj)

class DatasetGenerator:
    def __init__(self, use_llm: bool = False, api_key: Optional[str] = None, temperature: float = 0.2,
                 provider: str = "ambient", model: str = "deepseek-ai/DeepSeek-R1",
                 verify_mode: str = "equivalence", numeric_tol: float = 1e-6, require_unique: bool = True,
                 parse_retries: int = 0, parse_retry_delay: float = 5.0,
                 trace_retries: int = 2,
                 save_roundtrip: bool = False,
                 student_solve: int = 0,
                 use_encyclopedia_topics: bool = False,
                 encyclopedia_file: Optional[str] = None,
                 story_mode: str = "natural"):
        self.use_llm = use_llm
        self.client = None
        self.provider = (provider or "ambient").lower()
        self.model = model or "deepseek-ai/DeepSeek-R1"
        self.verify_mode = verify_mode
        self.numeric_tol = float(numeric_tol)
        self.require_unique = bool(require_unique)
        self.trace_retries = int(trace_retries)
        self.save_roundtrip = bool(save_roundtrip)
        self.student_solve = int(student_solve)
        self.use_encyclopedia_topics = bool(use_encyclopedia_topics)
        self.encyclopedia_file = encyclopedia_file or str(Path.cwd() / "Encyclopedia70K_20250810_174254.json")
        self._topic_paths: Optional[List[List[str]]] = None
        # buffer for skipped attempts
        self.skipped: List[Dict[str, Any]] = []
        if use_llm:
            if self.provider == "ambient":
                if api_key is None:
                    api_key = os.environ.get("AMBIENT_API_KEY") or os.environ.get("AMBIENT_KEY")
                self.client = create_ambient_client(api_key=api_key)
                try:
                    self.client.model = self.model
                except Exception:
                    pass
                try:
                    self.client.stream_output = False
                except Exception:
                    pass
            elif self.provider == "together":
                if _TogetherClient is None:
                    raise RuntimeError("Together client not available. Install 'together' and ensure together_client.py is present.")
                # Key is loaded within TogetherClient (env or file)
                self.client = _TogetherClient(model=self.model)
            elif self.provider == "openai":
                if _OpenAIClient is None:
                    raise RuntimeError("OpenAI client not available. Ensure openai_client.py exists and 'openai' is installed.")
                # Key is loaded within OpenAIClient (env var or local file)
                self.client = _OpenAIClient(model=self.model, timeout=300)
            elif self.provider == "google":
                try:
                    from google_client import GoogleClient  # type: ignore
                except Exception as e:
                    raise RuntimeError("Google client not available. Ensure google_client.py is present.") from e
                self.client = GoogleClient(model=self.model)
            elif self.provider == "grok":
                if _GrokClient is None:
                    raise RuntimeError("Grok client not available. Ensure grok_client.py is present (optional xai-sdk).")
                self.client = _GrokClient(model=self.model, timeout=300)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
        self.rationalizer = WordProblemRationalizer(
            self.client,
            RationalizationConfig(
                temperature=temperature,
                parse_retries=int(parse_retries),
                parse_retry_delay=float(parse_retry_delay),
                style=(story_mode or "natural"),
            ),
        )
        # Configure LLM logging file under logs/
        try:
            ts = time.strftime("%Y%m%d_%H%M%S")
            log_dir = Path.cwd() / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"algebra_generator_llm_{ts}.jsonl"
            self.rationalizer.llm_log_path = str(log_file)
            self.rationalizer.provider = self.provider
        except Exception:
            pass

        # Preload topic paths if enabled
        if self.use_encyclopedia_topics:
            try:
                self._topic_paths = self._load_topic_paths(self.encyclopedia_file)
            except Exception:
                self._topic_paths = None

    def _load_topic_paths(self, path: str) -> List[List[str]]:
        with open(Path(path), 'r', encoding='utf-8') as f:
            data = json.load(f)
        tree = data.get('tree') or []
        paths: List[List[str]] = []
        def dfs(node: Dict[str, Any]):
            topic = node.get('topic')
            path_list = list(node.get('path') or [])
            # Some trees might not include the leaf topic in 'path'; ensure it ends with the node topic
            if topic and (len(path_list) == 0 or path_list[-1] != topic):
                full = path_list + [topic]
            else:
                full = path_list if path_list else ([topic] if topic else [])
            if full:
                paths.append(full)
            for ch in node.get('children', []) or []:
                dfs(ch)
        for n in tree:
            dfs(n)
        # Deduplicate and keep reasonable length
        uniq = []
        seen = set()
        for p in paths:
            t = tuple(p)
            if t not in seen:
                seen.add(t)
                uniq.append(p)
        return uniq

    def _choose_topic_path(self) -> Optional[List[str]]:
        if not self._topic_paths:
            return None
        # Bias towards deeper, specific topics by sampling a few and picking the longest
        import random as _rnd
        candidates = [_rnd.choice(self._topic_paths) for _ in range(5)]
        candidates.sort(key=len, reverse=True)
        return candidates[0]

    # ---- Generation entrypoints ----
    def generate_one(self, family: str) -> ProblemArtifacts:
        if family == "linear_1var":
            return generate_linear_1var()
        if family == "linear_2var_system":
            return generate_linear_2var_system()
        if family == "quadratic":
            return generate_quadratic(complete_square_required=False)
        if family == "quadratic_complete_square":
            return generate_quadratic(complete_square_required=True)
        if family == "rational_equation":
            return generate_rational_equation()
        if family == "logexp":
            return generate_logexp()
        raise ValueError(f"Unknown family: {family}")

    def _templated_story(self, art: ProblemArtifacts) -> str:
        # Offline templater: produce a self-contained problem statement with explicit numeric parameters.
        fam = art.family
        eqs_text = "; ".join(eqs_to_strings(art.eqs))
        if fam == "linear_1var":
            eq = art.eqs[0]
            return (
                "You sign up for a plan where the total cost C follows the linear relation "
                f"{eq}. Here x is the number of months. Find x."
            )
        if fam == "linear_2var_system":
            return (
                "Two unknowns x and y represent quantities in a purchase. The totals satisfy the system "
                f"{eqs_text}. Solve for x and y."
            )
        if fam.startswith("quadratic"):
            return (
                "A projectile’s height is modeled by a quadratic that equals zero at impact. Using "
                f"{eqs_text}, solve for x."
            )
        if fam == "rational_equation":
            return (
                "A mixing process yields the rational relation "
                f"{eqs_text}. Solve for x, avoiding values that zero denominators."
            )
        if fam in ["log_equation", "exp_equation", "logexp"]:
            return (
                "A growth process is modeled by "
                f"{eqs_text}. Solve for x while respecting real-domain constraints."
            )
        return f"Solve the problem defined by: {eqs_text}"

    def _roundtrip_verify(self, text: str, original: ProblemArtifacts) -> Tuple[bool, str, Optional[float], Dict[str, Any]]:
        """Verify parsed equations according to verify_mode: equivalence, numeric, robustOptimal, or none.

        Returns (ok, feedback, error_value, meta) where meta may include parser_json/raw/reasoning.
        """
        if not self.use_llm:
            return False, "Round-trip skipped (use_llm=False).", None, {}
        if self.verify_mode == "none":
            return True, "Verification disabled (mode=none).", None, {}
        # Ensure no intermingled reasoning tags leak into the parser
        try:
            _, stripped = extract_think_blocks(text)
            if stripped:
                text = stripped
        except Exception:
            pass
        try:
            # Sanitize and prefer strict variable set if available
            text_in = sanitize_plaintext(text)
            allowed_vars = list(original.variables.keys())
            if hasattr(self.rationalizer, "story_to_equations_strict"):
                parsed, parser_raw, parser_think = self.rationalizer.story_to_equations_strict(text_in, allowed_vars)
            else:
                parsed, parser_raw, parser_think = self.rationalizer.story_to_equations(text_in)
            eqs = []
            local_dict = {"Eq": Eq, "log": sp.log, "exp": sp.exp}
            for s in parsed.get("equations", []):
                try:
                    eqs.append(parse_expr(s, local_dict=local_dict, evaluate=False))
                except Exception:
                    continue
            if not eqs:
                # Prepare numeric meta for logging in skipped records
                orig_numeric_meta = {}
                try:
                    for k, v in original.solution.items():
                        if isinstance(v, (list, tuple)):
                            orig_numeric_meta[k] = [float(sp.N(t)) for t in v]
                        else:
                            orig_numeric_meta[k] = [float(sp.N(v))]
                except Exception:
                    orig_numeric_meta = {}
                return False, "Parser returned no equations.", None, {
                    "parser_json": parsed, "parser_raw": parser_raw, "parser_reasoning": parser_think,
                    "orig_numeric": orig_numeric_meta, "parsed_numeric": None, "best_numeric_error": None,
                    "vars": list(original.variables.keys()),
                }

            vars_sorted = list(sp.ordered([sp.Symbol(v) for v in original.variables.keys()]))

            # Allow per-item override: if original has multiple roots and mode==equivalence, escalate to robustOptimal
            mode = self.verify_mode
            try:
                if mode == "equivalence":
                    for v in original.solution.values():
                        if isinstance(v, (list, tuple)):
                            mode = "robustOptimal"
                            break
            except Exception:
                pass

            # Precompute numeric candidates for logging in case of skip
            orig_numeric_meta = {}
            try:
                for k, v in original.solution.items():
                    if isinstance(v, (list, tuple)):
                        orig_numeric_meta[k] = [float(sp.N(t)) for t in v]
                    else:
                        orig_numeric_meta[k] = [float(sp.N(v))]
            except Exception:
                orig_numeric_meta = {}
            parsed_numeric_meta = None
            best_numeric_error = None
            try:
                if len(vars_sorted) == 1:
                    var = vars_sorted[0]
                    # Solve parsed equations for var
                    if len(eqs) == 1:
                        solset = sp.solveset(eqs[0], var, domain=S.Reals)
                        parsed_numeric_meta = [float(sp.N(v)) for v in solset]
                    else:
                        sol = sp.solve(eqs, var)
                        if isinstance(sol, (list, tuple)):
                            parsed_numeric_meta = [float(sp.N(v)) for v in sol]
                        else:
                            parsed_numeric_meta = [float(sp.N(sol))]
                    # Best error vs original
                    try:
                        ov = next(iter(original.solution.values()))
                        orig_vals = ([float(sp.N(t)) for t in ov] if isinstance(ov, (list, tuple)) else [float(sp.N(ov))])
                        for a in (parsed_numeric_meta or []):
                            for b in orig_vals:
                                err = abs(a - b)
                                if best_numeric_error is None or (err < best_numeric_error):
                                    best_numeric_error = err
                    except Exception:
                        pass
                else:
                    sols = sp.solve(eqs, vars_sorted, dict=True)
                    if isinstance(sols, list) and sols:
                        parsed_numeric_meta = []
                        for cand in sols:
                            try:
                                parsed_numeric_meta.append({str(v): float(sp.N(cand.get(v))) for v in vars_sorted if v in cand})
                            except Exception:
                                continue
                        # Compute best max component error
                        try:
                            orig_map = {str(k): float(sp.N(v)) for k, v in original.solution.items()}
                            for cand in (parsed_numeric_meta or []):
                                errs = []
                                for v in [str(x) for x in vars_sorted]:
                                    if v in cand and v in orig_map:
                                        errs.append(abs(cand[v] - orig_map[v]))
                                if errs:
                                    m = max(errs)
                                    if best_numeric_error is None or (m < best_numeric_error):
                                        best_numeric_error = m
                        except Exception:
                            pass
            except Exception:
                parsed_numeric_meta = None

            if mode == "equivalence":
                # Build substitution map from original solution (must be unique unless allowed)
                sol_map: Dict[sp.Symbol, Any] = {}
                for k, v in original.solution.items():
                    if isinstance(v, (list, tuple)):
                        if self.require_unique:
                            return False, f"Original solution for {k} not unique.", None, {
                                "parser_json": parsed, "parser_raw": parser_raw, "parser_reasoning": parser_think,
                                "orig_numeric": orig_numeric_meta, "parsed_numeric": parsed_numeric_meta,
                                "best_numeric_error": best_numeric_error, "vars": [str(v) for v in vars_sorted],
                            }
                        else:
                            v = v[0]
                    try:
                        sol_map[sp.Symbol(k)] = v
                    except Exception:
                        return False, f"Invalid variable name in solution: {k}", None, {
                            "parser_json": parsed, "parser_raw": parser_raw, "parser_reasoning": parser_think,
                            "orig_numeric": orig_numeric_meta, "parsed_numeric": parsed_numeric_meta,
                            "best_numeric_error": best_numeric_error, "vars": [str(v) for v in vars_sorted],
                        }
                for eq in eqs:
                    if hasattr(eq, 'lhs') and hasattr(eq, 'rhs'):
                        diff = sp.simplify(eq.lhs.subs(sol_map) - eq.rhs.subs(sol_map))
                        if sp.simplify(diff) != 0:
                            return False, "Residual not zero under original solution.", None, {
                                "parser_json": parsed, "parser_raw": parser_raw, "parser_reasoning": parser_think,
                                "orig_numeric": orig_numeric_meta, "parsed_numeric": parsed_numeric_meta,
                                "best_numeric_error": best_numeric_error, "vars": [str(v) for v in vars_sorted],
                            }
                    else:
                        val = sp.simplify(eq.subs(sol_map))
                        if sp.simplify(val) != 0:
                            return False, "Expression did not reduce to zero.", None, {
                                "parser_json": parsed, "parser_raw": parser_raw, "parser_reasoning": parser_think,
                                "orig_numeric": orig_numeric_meta, "parsed_numeric": parsed_numeric_meta,
                                "best_numeric_error": best_numeric_error, "vars": [str(v) for v in vars_sorted],
                            }
                return True, "Round-trip equations consistent with original solution.", None, {
                    "parser_json": parsed, "parser_raw": parser_raw, "parser_reasoning": parser_think,
                    "orig_numeric": orig_numeric_meta, "parsed_numeric": parsed_numeric_meta,
                    "best_numeric_error": best_numeric_error, "vars": [str(v) for v in vars_sorted],
                }

            # Robust optimal mode: structural proportional matching with randomized identity testing,
            # record number-literal fidelity, then fall back to numeric check if needed.
            if mode == "robustOptimal":
                # Helper: extract numeric literals from equations via SymPy atoms
                def _numeric_literals(eq_list: List[sp.Eq]) -> Counter:
                    nums: list[str] = []
                    for eq in eq_list:
                        # residual expression
                        expr = (eq.lhs - eq.rhs) if hasattr(eq, 'lhs') and hasattr(eq, 'rhs') else eq
                        for a in expr.atoms(sp.Integer, sp.Rational, sp.Float):
                            try:
                                nums.append(str(sp.nsimplify(a)))
                            except Exception:
                                try:
                                    nums.append(str(a))
                                except Exception:
                                    pass
                    return Counter(nums)

                # Helper: check proportional identity eq1 <-> c * eq2 across random real points
                def _proportional(eq1: sp.Eq, eq2: sp.Eq, vars_syms: List[sp.Symbol], samples: int = 8,
                                  tol: float = 1e-8) -> bool:
                    f1 = (eq1.lhs - eq1.rhs) if hasattr(eq1, 'lhs') and hasattr(eq1, 'rhs') else eq1
                    f2 = (eq2.lhs - eq2.rhs) if hasattr(eq2, 'lhs') and hasattr(eq2, 'rhs') else eq2
                    # Find a non-zero evaluation to estimate c
                    c_val = None
                    tries = 0
                    import random as _rnd
                    while tries < 40 and c_val is None:
                        tries += 1
                        subs = {}
                        for v in vars_syms:
                            # Avoid singularities by sampling away from zero for denominators
                            subs[v] = _rnd.uniform(-7.0, 7.0)
                        try:
                            a = complex(f1.subs(subs).evalf())
                            b = complex(f2.subs(subs).evalf())
                        except Exception:
                            continue
                        if not (a == a and b == b):  # NaN guard
                            continue
                        if abs(b) > 1e-12:
                            c_val = a / b
                    if c_val is None:
                        return False
                    # Validate proportionality across more samples
                    ok_count = 0
                    for _ in range(samples):
                        subs = {}
                        for v in vars_syms:
                            subs[v] = _rnd.uniform(-9.0, 9.0)
                        try:
                            a = complex(f1.subs(subs).evalf())
                            b = complex(f2.subs(subs).evalf())
                        except Exception:
                            continue
                        if not (a == a and b == b):
                            continue
                        # Allow both nearly zero simultaneously as consistent
                        if abs(a) <= tol and abs(b) <= tol:
                            ok_count += 1
                            continue
                        # Check |a - c*b| <= tol * (1 + |a|)
                        if abs(a - c_val * b) <= tol * (1.0 + abs(a)):
                            ok_count += 1
                    return ok_count >= max(3, samples - 2)

                def _structural_match(orig_list: List[sp.Eq], parsed_list: List[sp.Eq], vars_syms: List[sp.Symbol]) -> bool:
                    if len(orig_list) != len(parsed_list):
                        return False
                    remaining = list(orig_list)
                    for peq in parsed_list:
                        matched = False
                        for i, oeq in enumerate(remaining):
                            try:
                                if _proportional(oeq, peq, vars_syms):
                                    matched = True
                                    remaining.pop(i)
                                    break
                            except Exception:
                                continue
                        if not matched:
                            return False
                    return True

                # Compute fidelity signal (not a hard gate)
                try:
                    num_fid = _numeric_literals(original.eqs) == _numeric_literals(eqs)
                except Exception:
                    num_fid = None  # unknown

                # Structural proportional match
                try:
                    if _structural_match(original.eqs, eqs, vars_sorted):
                        fb = "Structural proportional match (orderless); numbers_fidelity=" + ("true" if num_fid else ("false" if num_fid is False else "unknown"))
                        return True, fb, None, {"parser_json": parsed, "parser_raw": parser_raw, "parser_reasoning": parser_think,
                                                "orig_numeric": orig_numeric_meta, "parsed_numeric": parsed_numeric_meta,
                                                "best_numeric_error": best_numeric_error, "vars": [str(v) for v in vars_sorted]}
                except Exception:
                    pass
                # Fallback to numeric check below

            # Numeric mode (also used as fallback for robustOptimal)
            tol = float(self.numeric_tol)
            if len(vars_sorted) == 1:
                var = vars_sorted[0]
                # Original numeric solutions
                orig_vals: List[float] = []
                ov = list(original.solution.values())[0]
                if isinstance(ov, (list, tuple)):
                    for t in ov:
                        try:
                            orig_vals.append(float(sp.N(t)))
                        except Exception:
                            pass
                else:
                    try:
                        orig_vals.append(float(sp.N(ov)))
                    except Exception:
                        pass
                if not orig_vals:
                    return False, "Original numeric solution empty.", None, {"parser_json": parsed, "parser_raw": parser_raw, "parser_reasoning": parser_think}
                # Solve parsed equations for var
                try:
                    if len(eqs) == 1:
                        solset = sp.solveset(eqs[0], var, domain=S.Reals)
                        cand_vals = [float(sp.N(v)) for v in solset]
                    else:
                        sol = sp.solve(eqs, var)
                        if isinstance(sol, (list, tuple)):
                            cand_vals = [float(sp.N(v)) for v in sol]
                        else:
                            cand_vals = [float(sp.N(sol))]
                except Exception:
                    return False, "Parsed system unsolved.", None, {
                        "parser_json": parsed, "parser_raw": parser_raw, "parser_reasoning": parser_think,
                        "orig_numeric": orig_numeric_meta, "parsed_numeric": None, "best_numeric_error": None,
                        "vars": [str(v) for v in vars_sorted],
                    }
                best_err = None
                for a in cand_vals:
                    for b in orig_vals:
                        err = abs(a - b)
                        best_err = err if best_err is None or err < best_err else best_err
                        if err <= tol:
                            return True, "Numeric solutions matched.", err, {
                                "parser_json": parsed, "parser_raw": parser_raw, "parser_reasoning": parser_think,
                                "orig_numeric": orig_numeric_meta, "parsed_numeric": cand_vals,
                                "best_numeric_error": err, "vars": [str(v) for v in vars_sorted],
                            }
                return False, "No numeric match to ground truth.", best_err, {"parser_json": parsed, "parser_raw": parser_raw, "parser_reasoning": parser_think,
                                                                                "orig_numeric": orig_numeric_meta, "parsed_numeric": cand_vals,
                                                                                "best_numeric_error": best_err, "vars": [str(v) for v in vars_sorted]}
            else:
                try:
                    sols = sp.solve(eqs, vars_sorted, dict=True)
                except Exception:
                    return False, "Parsed system unsolved.", None, {
                        "parser_json": parsed, "parser_raw": parser_raw, "parser_reasoning": parser_think,
                        "orig_numeric": orig_numeric_meta, "parsed_numeric": None, "best_numeric_error": None,
                        "vars": [str(v) for v in vars_sorted],
                    }
                if not isinstance(sols, list) or not sols:
                    return False, "Parsed system returned no solutions.", None, {
                        "parser_json": parsed, "parser_raw": parser_raw, "parser_reasoning": parser_think,
                        "orig_numeric": orig_numeric_meta, "parsed_numeric": None, "best_numeric_error": None,
                        "vars": [str(v) for v in vars_sorted],
                    }
                try:
                    orig_map = {sp.Symbol(k): float(sp.N(v)) for k, v in original.solution.items()}
                except Exception:
                    return False, "Original solution not numeric.", None, {
                        "parser_json": parsed, "parser_raw": parser_raw, "parser_reasoning": parser_think,
                        "orig_numeric": orig_numeric_meta, "parsed_numeric": None, "best_numeric_error": None,
                        "vars": [str(v) for v in vars_sorted],
                    }
                best_err = None
                for cand in sols:
                    try:
                        ok = True
                        max_comp_err = 0.0
                        for var in vars_sorted:
                            if var not in cand:
                                ok = False
                                break
                            comp_err = abs(float(sp.N(cand[var])) - orig_map[var])
                            max_comp_err = max(max_comp_err, comp_err)
                            if comp_err > tol:
                                ok = False
                                break
                        if ok:
                            return True, "Numeric solutions matched.", max_comp_err, {
                                "parser_json": parsed, "parser_raw": parser_raw, "parser_reasoning": parser_think,
                                "orig_numeric": orig_numeric_meta, "parsed_numeric": [
                                    {str(v): float(sp.N(cand.get(v))) for v in vars_sorted if v in cand}
                                ],
                                "best_numeric_error": max_comp_err, "vars": [str(v) for v in vars_sorted],
                            }
                    except Exception:
                        continue
                return False, "No numeric match to ground truth.", best_err, {
                    "parser_json": parsed, "parser_raw": parser_raw, "parser_reasoning": parser_think,
                    "orig_numeric": orig_numeric_meta, "parsed_numeric": [
                        {str(v): float(sp.N(cand.get(v))) for v in vars_sorted if v in cand} for cand in (sols or [])
                    ],
                    "best_numeric_error": best_err, "vars": [str(v) for v in vars_sorted],
                }
        except Exception as e:
            return False, f"Round-trip parse failed: {e}", None, {}

    def build_records(self, families: List[str], n_per_family: int, seed: int = 42, progress: bool = False,
                      mode: str = "standard", jobs: int = 1) -> List[DatasetRecord]:
        # Parallel path: use a process pool when jobs > 1
        try:
            jobs = int(jobs)
        except Exception:
            jobs = 1
        if jobs and jobs > 1:
            return self._build_records_parallel(families, n_per_family, seed, progress, mode, jobs)
        random.seed(seed)
        records: List[DatasetRecord] = []
        total_expected = len(families) * n_per_family
        attempted = 0
        attempt_time_sum = 0.0
        accepted_time_sum = 0.0
        accepted_count = 0

        def _fmt_eta(sec: float) -> str:
            if sec <= 0 or sec != sec:  # nan/neg guard
                return "-"
            m, s = divmod(int(sec + 0.5), 60)
            if m >= 60:
                h, m = divmod(m, 60)
                return f"{h}h{m:02d}m"
            if m > 0:
                return f"{m}m{s:02d}s"
            return f"{s}s"

        def _print_progress(last_elapsed: float | None = None):
            if not progress:
                return
            width = 30
            fill = int(width * attempted / max(1, total_expected))
            bar = '[' + '#' * fill + '·' * (width - fill) + ']'
            avg_txt = '-' if accepted_count == 0 else f"{accepted_time_sum/accepted_count:.1f}s/ok"
            last_txt = '-' if last_elapsed is None else f"{last_elapsed:.1f}s"
            rem = max(0, total_expected - attempted)
            avg_attempt = attempt_time_sum/attempted if attempted else 0.0
            eta_txt = _fmt_eta(rem * avg_attempt)
            msg = (
                f"\r{bar} {attempted}/{total_expected} | ok:{accepted_count} "
                f"skip:{attempted-accepted_count} | avg:{avg_txt} | last:{last_txt} | eta:{eta_txt}"
            )
            sys.stdout.write(msg)
            sys.stdout.flush()

        _print_progress()
        for fam in families:
            for _ in range(n_per_family):
                t0 = time.time()
                art = self.generate_one(fam)
                # Produce story
                # Optional topic context for story diversification
                chosen_topic_path: Optional[List[str]] = None
                if self.use_encyclopedia_topics:
                    try:
                        chosen_topic_path = self._choose_topic_path()
                    except Exception:
                        chosen_topic_path = None

                if self.use_llm:
                    try:
                        story_raw = self.rationalizer.equation_to_story(art, topic_path=chosen_topic_path)
                    except Exception as e:
                        story_raw = self._templated_story(art) + f" [LLM error: {e}]"
                else:
                    story_raw = self._templated_story(art)
                # Always strip any <think> blocks from the problem text and keep them separately
                story_reasoning = None
                try:
                    story_reasoning, story = extract_think_blocks(story_raw)
                except Exception:
                    story_reasoning = None
                # For Ambient, reasoning may come separately; prefer it when <think> blocks absent
                if not story_reasoning:
                    try:
                        rc = getattr(self.rationalizer, 'last_reasoning_content', None)
                        if rc:
                            story_reasoning = rc
                    except Exception:
                        pass
                # Fallback: if story text is empty after stripping think blocks, reuse thinking content as story
                if not story or not story.strip():
                    try:
                        if story_reasoning and isinstance(story_reasoning, str):
                            import re as _re
                            story = _re.sub(r"</?think>", "", story_reasoning).strip()
                    except Exception:
                        pass
                    if not story or not story.strip():
                        # Last resort: use the raw content
                        story = (story_raw or "").strip()
                # Optional: one-off random fact for the chosen encyclopedia topic
                rand_fact: Optional[str] = None
                if self.use_encyclopedia_topics and chosen_topic_path and self.use_llm and self.client is not None:
                    try:
                        topic_str = " / ".join([t for t in chosen_topic_path if t])
                        sys_prompt = (
                            "You state concise, verifiable facts. Return one factual sentence about the topic. "
                            "No preface, no Markdown, no lists."
                        )
                        user_prompt = (
                            "Topic path:\n" + topic_str + "\n\n" +
                            "Write one random fact about this topic (<= 25 words)."
                        )
                        rf_raw = self.rationalizer._complete(sys_prompt, user_prompt, max_tokens=int(self.rationalizer.cfg.max_tokens), temperature=0.7, tag="random_fact")
                        think, rf_text = extract_think_blocks(rf_raw)
                        # Prefer stripped content even if empty; do not fall back to raw which may contain <think>
                        rand_fact = ((rf_text if rf_text is not None else rf_raw) or "").strip()
                        # Trim to one sentence and remove code fences/backticks if any
                        if rand_fact:
                            # Keep up to first sentence terminator
                            m = re.search(r"([\s\S]*?[\.!?])\s", rand_fact + " ")
                            if m:
                                rand_fact = m.group(1).strip()
                            rand_fact = rand_fact.strip('`').strip()
                    except Exception:
                        rand_fact = None
                # Round-trip verify (if LLM used)
                verified, fb, verr, rt_meta = self._roundtrip_verify(story, art)
                # If require_roundtrip, skip unverified items
                if self.use_llm and self.rationalizer.cfg.require_roundtrip and not verified:
                    attempts_meta: List[Dict[str, Any]] = []
                    # record first attempt
                    attempts_meta.append({
                        "problem_text": story,
                        "problem_text_reasoning": story_reasoning,
                        "feedback": fb,
                        "verification_error": verr,
                        "parser_json": (rt_meta.get("parser_json") if isinstance(rt_meta, dict) else None),
                        "parser_raw": (rt_meta.get("parser_raw") if isinstance(rt_meta, dict) else None),
                        "parser_reasoning": (rt_meta.get("parser_reasoning") if isinstance(rt_meta, dict) else None),
                        "orig_numeric": (rt_meta.get("orig_numeric") if isinstance(rt_meta, dict) else None),
                        "parsed_numeric": (rt_meta.get("parsed_numeric") if isinstance(rt_meta, dict) else None),
                        "best_numeric_error": (rt_meta.get("best_numeric_error") if isinstance(rt_meta, dict) else None),
                        "vars": (rt_meta.get("vars") if isinstance(rt_meta, dict) else None),
                    })
                    # Try one retry with a fresh story if LLM is available
                    try:
                        if self.client is not None:
                            story2_raw = self.rationalizer.equation_to_story(art, topic_path=chosen_topic_path)
                            # Strip <think> tags from retry story as well
                            story2_reasoning, story2 = extract_think_blocks(story2_raw)
                            if not story2_reasoning:
                                try:
                                    rc2 = getattr(self.rationalizer, 'last_reasoning_content', None)
                                    if rc2:
                                        story2_reasoning = rc2
                                except Exception:
                                    pass
                            verified2, fb2, verr2, rt_meta2 = self._roundtrip_verify(story2, art)
                            if verified2:
                                story, verified, fb, verr, rt_meta = story2, verified2, fb2, verr2, rt_meta2
                                # Keep reasoning for accepted retry
                                story_reasoning = story2_reasoning
                            else:
                                attempts_meta.append({
                                    "problem_text": story2,
                                    "problem_text_reasoning": story2_reasoning,
                                    "feedback": fb2,
                                    "verification_error": verr2,
                                    "parser_json": (rt_meta2.get("parser_json") if isinstance(rt_meta2, dict) else None),
                                    "parser_raw": (rt_meta2.get("parser_raw") if isinstance(rt_meta2, dict) else None),
                                    "parser_reasoning": (rt_meta2.get("parser_reasoning") if isinstance(rt_meta2, dict) else None),
                                    "orig_numeric": (rt_meta2.get("orig_numeric") if isinstance(rt_meta2, dict) else None),
                                    "parsed_numeric": (rt_meta2.get("parsed_numeric") if isinstance(rt_meta2, dict) else None),
                                    "best_numeric_error": (rt_meta2.get("best_numeric_error") if isinstance(rt_meta2, dict) else None),
                                    "vars": (rt_meta2.get("vars") if isinstance(rt_meta2, dict) else None),
                                })
                                # Append skipped item
                                self._record_skipped(art, attempts_meta, reason="verification_failed", topic_path=chosen_topic_path)
                                elapsed = time.time() - t0
                                attempted += 1
                                attempt_time_sum += elapsed
                                _print_progress(elapsed)
                                continue
                        else:
                            self._record_skipped(art, attempts_meta, reason="no_llm_client", topic_path=chosen_topic_path)
                            elapsed = time.time() - t0
                            attempted += 1
                            attempt_time_sum += elapsed
                            _print_progress(elapsed)
                            continue
                    except Exception as e:
                        self._record_skipped(art, attempts_meta, reason=f"exception: {e}", topic_path=chosen_topic_path)
                        elapsed = time.time() - t0
                        attempted += 1
                        attempt_time_sum += elapsed
                        _print_progress(elapsed)
                        continue
                eq_strs = eqs_to_strings(art.eqs)
                sol_eval = eval_solution(art.solution)
                item_a = None
                item_b = None
                item_c = None
                item_d = None
                item_c_reason = None
                item_d_reason = None
                trace_retry_count = 0
                verification_error: Optional[float] = None
                story_numbers_ok: Optional[bool] = None
                # Student-solve attempts (optional)
                student_attempts: Optional[List[Dict[str, Any]]] = None
                student_success_count: Optional[int] = None
                if mode == "posthoc":
                    # Choose a unique numeric answer per variable
                    uniq = {}
                    for k, v in sol_eval.items():
                        if isinstance(v, list) and v:
                            try:
                                uniq[k] = sorted(v, key=lambda x: (abs(x), x))[0]
                            except Exception:
                                uniq[k] = v[0]
                        else:
                            uniq[k] = v
                    item_a = list(eq_strs)
                    item_b = uniq
                    item_c = story
                    # Attach stripped reasoning (if any) regardless of provider
                    item_c_reason = story_reasoning
                    # simple number fidelity check for story: ensure all equation numbers appear in the story
                    try:
                        nums_in_eqs = re.findall(r"[-+]?\d+(?:\.\d+)?", " ".join(item_a)) if item_a else []
                        story_numbers_ok = all(n in (item_c or "") for n in nums_in_eqs)
                    except Exception:
                        story_numbers_ok = None
                    if self.use_llm:
                        try:
                            # Generate reasoning with numeric-consistency guard
                            for attempt_idx in range(self.trace_retries + 1):
                                item_d_raw = self.rationalizer.reasoning_trace(item_a, item_c, item_b)
                                if self.provider == "together":
                                    item_d_reason, item_d_candidate = extract_think_blocks(item_d_raw)
                                else:
                                    item_d_candidate = item_d_raw
                                    item_d_reason = None
                                # Check per-variable numeric equality patterns: var = value
                                bad = False
                                for var, ans in (item_b or {}).items():
                                    try:
                                        m = re.search(rf"\b{re.escape(var)}\s*=\s*([-+]?\d+(?:\.\d+)?)", item_d_candidate)
                                        if m:
                                            val = float(m.group(1))
                                            if abs(val - float(ans)) > self.numeric_tol:
                                                bad = True
                                                break
                                    except Exception:
                                        continue
                                if not bad:
                                    item_d = item_d_candidate
                                    trace_retry_count = attempt_idx
                                    break
                                trace_retry_count = attempt_idx
                            if item_d is None:
                                item_d = item_d_candidate
                        except Exception as e:
                            item_d = f"[reasoning generation error: {e}]"

                rec = DatasetRecord(
                    id=str(uuid.uuid4()),
                    family=art.family,
                    problem_text=story,
                    topic_path=chosen_topic_path,
                    topic_name=(chosen_topic_path[-1] if isinstance(chosen_topic_path, list) and chosen_topic_path else None),
                    variables=art.variables,
                    domain=art.domain,
                    eq_system_str=eq_strs,
                    eq_system_ast=serialize_eqs_ast(art.eqs),
                    sympy_src=art.sympy_src,
                    solution=art.solution,
                    solution_eval=sol_eval,
                    difficulty_meta=art.difficulty_meta,
                    roundtrip_verified=(None if ((self.verify_mode == 'none') or (not self.use_llm)) else verified),
                    verifier_feedback=fb,
                    verification_error=verr,
                    provider=self.provider,
                    model=self.model,
                    seed=seed,
                    verify_mode=self.verify_mode,
                    story_numbers_ok=story_numbers_ok,
                    trace_retry_count=(trace_retry_count if mode == 'posthoc' else None),
                    problem_text_reasoning=story_reasoning if mode != "posthoc" else None,
                    random_fact=rand_fact,
                    item_a_equations=item_a,
                    item_b_numeric_answer=item_b,
                    item_c_problem=item_c,
                    item_d_reasoning=item_d,
                    item_c_problem_reasoning=item_c_reason,
                    item_d_reasoning_reasoning=item_d_reason,
                    item_d_reasoning_steps=(
                        [ln.strip() for ln in (item_d or '').splitlines() if re.match(r"^\s*(?:\d+\.|Step\s+\d+|\-\s)", ln)]
                        if (mode == 'posthoc') else None
                    ),
                )
                # Attach numeric diagnostics (always when LLM is used)
                if self.use_llm:
                    try:
                        rec.verifier_orig_numeric = (rt_meta.get("orig_numeric") if isinstance(rt_meta, dict) else None)
                        rec.verifier_parsed_numeric = (rt_meta.get("parsed_numeric") if isinstance(rt_meta, dict) else None)
                        rec.verifier_best_numeric_error = (rt_meta.get("best_numeric_error") if isinstance(rt_meta, dict) else None)
                        rec.verifier_vars = (rt_meta.get("vars") if isinstance(rt_meta, dict) else None)
                        # Always attach parser reasoning (unconditional; prefer provider-native reasoning when present in rt_meta)
                        rec.roundtrip_parser_reasoning = (rt_meta.get("parser_reasoning") if isinstance(rt_meta, dict) else None)
                    except Exception:
                        pass
                # Optionally attach full round-trip parser artifacts (JSON/raw)
                if self.save_roundtrip and self.use_llm:
                    try:
                        rec.roundtrip_parser_json = (rt_meta.get("parser_json") if isinstance(rt_meta, dict) else None)
                        rec.roundtrip_parser_raw = (rt_meta.get("parser_raw") if isinstance(rt_meta, dict) else None)
                    except Exception:
                        pass

                # Optionally run student solves
                if self.use_llm and self.student_solve and int(self.student_solve) > 0:
                    student_attempts = []
                    success = 0
                    for _i in range(int(self.student_solve)):
                        try:
                            solve_raw = self.rationalizer.student_solve(story)
                            think, solve_text = extract_think_blocks(solve_raw)
                            # Extract final answer mapping
                            fa_map = {}
                            # Search for Final Answer line
                            m = None
                            for line in (solve_text or solve_raw).splitlines()[::-1]:
                                if line.strip().lower().startswith("final answer:"):
                                    m = line.strip()[len("final answer:"):].strip()
                                    break
                            # Parse variable assignments like x=..., y=...
                            if m:
                                for var in original.variables.keys():
                                    import re as _re
                                    rx = _re.compile(rf"\b{_re.escape(var)}\s*[:=]\s*([-+]?\d+(?:\.\d+)?)")
                                    mm = rx.search(m)
                                    if mm:
                                        try:
                                            fa_map[var] = float(mm.group(1))
                                        except Exception:
                                            pass
                            # Verify against ground truth numerically
                            match = None
                            verr_student = None
                            try:
                                tol = float(self.numeric_tol)
                                # Only verify if we have values for all vars
                                have_all = all(k in fa_map for k in original.variables.keys())
                                if have_all:
                                    # Build orig numeric map
                                    on = {k: float(sp.N(v if not isinstance(v, (list, tuple)) else v[0])) for k, v in original.solution.items()}
                                    max_err = 0.0
                                    for k, v in on.items():
                                        err = abs(float(fa_map.get(k)) - v)
                                        max_err = max(max_err, err)
                                    verr_student = max_err
                                    match = (max_err <= tol)
                            except Exception:
                                pass
                            student_attempts.append({
                                "text": solve_text or solve_raw,
                                "reasoning": think,
                                "final_answer": m,
                                "final_answer_map": fa_map if fa_map else None,
                                "numeric_match": match,
                                "verification_error": verr_student,
                            })
                            if match:
                                success += 1
                        except Exception as _e:
                            student_attempts.append({
                                "text": None,
                                "reasoning": None,
                                "final_answer": None,
                                "error": str(_e),
                            })
                    student_success_count = success

                rec.student_attempts = student_attempts
                rec.student_success_count = student_success_count

                # Compute and attach overall duration, then append
                elapsed = time.time() - t0
                try:
                    rec.duration_sec = elapsed
                except Exception:
                    pass
                records.append(rec)
                attempted += 1
                attempt_time_sum += elapsed
                accepted_time_sum += elapsed
                accepted_count += 1
                _print_progress(elapsed)

        if progress:
            sys.stdout.write("\n")
            sys.stdout.flush()
        return records

    def _build_records_parallel(self, families: List[str], n_per_family: int, seed: int, progress: bool, mode: str, jobs: int) -> List[DatasetRecord]:
        # Cap jobs at 10 per request
        max_workers = max(1, min(int(jobs), 10))
        total_expected = len(families) * n_per_family
        # Prepare single log file and queue
        run_ts = time.strftime("%Y%m%d_%H%M%S")
        log_dir = Path.cwd() / "logs"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        log_path = str(log_dir / f"algebra_generator_llm_{run_ts}.jsonl")
        # Drain log entries from subprocesses
        import multiprocessing as mp
        try:
            mp.set_start_method('spawn', force=True)
        except Exception:
            pass
        mgr = mp.Manager()
        log_queue = mgr.Queue()

        stop_sentinel = object()
        def _drain_logs(q, path, sentinel):
            try:
                p = Path(path)
                p.parent.mkdir(parents=True, exist_ok=True)
                with p.open('a', encoding='utf-8') as f:
                    while True:
                        try:
                            item = q.get()
                        except Exception:
                            break
                        if item is sentinel or (isinstance(item, dict) and item.get("__stop__")):
                            break
                        try:
                            f.write(json.dumps(item, ensure_ascii=False, indent=2) + "\n")
                        except Exception:
                            continue
            except Exception:
                pass

        import threading
        log_thread = threading.Thread(target=_drain_logs, args=(log_queue, log_path, stop_sentinel), daemon=True)
        log_thread.start()

        # Build job specs
        jobs_list: List[Dict[str, Any]] = []
        jid = 0
        for fam in families:
            for _ in range(n_per_family):
                jid += 1
                jobs_list.append({
                    "run_id": run_ts,
                    "job_id": jid,
                    "family": fam,
                    "seed": int(seed) + jid,
                    "use_llm": self.use_llm,
                    "provider": self.provider,
                    "model": self.model,
                    "verify_mode": self.verify_mode,
                    "numeric_tol": self.numeric_tol,
                    "require_unique": self.require_unique,
                    "parse_retries": self.rationalizer.cfg.parse_retries,
                    "parse_retry_delay": self.rationalizer.cfg.parse_retry_delay,
                    "trace_retries": self.trace_retries,
                    "save_roundtrip": self.save_roundtrip,
                    "student_solve": self.student_solve,
                    "use_encyclopedia_topics": self.use_encyclopedia_topics,
                    "encyclopedia_file": self.encyclopedia_file,
                    "story_mode": getattr(self.rationalizer.cfg, 'style', 'natural'),
                    "mode": mode,
                    "log_queue": log_queue,
                })

        # Progress helpers
        attempted = 0
        accepted_count = 0
        attempt_time_sum = 0.0
        accepted_time_sum = 0.0
        start_wall = time.time()
        def _fmt_eta(sec: float) -> str:
            if sec <= 0 or sec != sec:
                return "-"
            m, s = divmod(int(sec + 0.5), 60)
            if m >= 60:
                h, m = divmod(m, 60)
                return f"{h}h{m:02d}m"
            if m > 0:
                return f"{m}m{s:02d}s"
            return f"{s}s"
        def _print_progress(last_elapsed: float | None = None):
            if not progress:
                return
            width = 30
            fill = int(width * attempted / max(1, total_expected))
            bar = '[' + '#' * fill + '·' * (width - fill) + ']'
            # Show average wall-clock per completed item for clearer parallel progress
            avg_wall = ( (time.time() - start_wall) / attempted ) if attempted > 0 else 0.0
            avg_txt = '-' if attempted == 0 else f"{avg_wall:.1f}s/item"
            last_txt = '-' if last_elapsed is None else f"{last_elapsed:.1f}s"
            rem = max(0, total_expected - attempted)
            # ETA based on observed throughput (completions per wall-second)
            elapsed_total = max(1e-6, (time.time() - start_wall))
            throughput = attempted / elapsed_total if attempted > 0 else 0.0
            eta_sec = (rem / throughput) if throughput > 0 else 0.0
            eta_txt = _fmt_eta(eta_sec)
            msg = (
                f"\r{bar} {attempted}/{total_expected} | ok:{accepted_count} "
                f"skip:{attempted-accepted_count} | avg:{avg_txt} | last:{last_txt} | eta:{eta_txt}"
            )
            sys.stdout.write(msg)
            sys.stdout.flush()
        _print_progress()

        # Run jobs in a process pool
        from concurrent.futures import ProcessPoolExecutor, as_completed
        results_by_jid: Dict[int, Dict[str, Any]] = {}
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(_worker_run_job, job) for job in jobs_list]
                for fut in as_completed(futures):
                    try:
                        res = fut.result()
                    except Exception as e:
                        # Account failed job as a skip
                        attempted += 1
                        attempt_time_sum += 0.0
                        _print_progress(0.0)
                        continue
                    jid2 = int(res.get("job_id"))
                    results_by_jid[jid2] = res
                    attempted += 1
                    elapsed = float(res.get("elapsed") or 0.0)
                    attempt_time_sum += elapsed
                    if res.get("accepted"):
                        accepted_count += 1
                        accepted_time_sum += elapsed
                    else:
                        # accumulate skipped
                        try:
                            self.skipped.extend(list(res.get("skipped") or []))
                        except Exception:
                            pass
                    _print_progress(elapsed)
        finally:
            if progress:
                sys.stdout.write("\n")
                sys.stdout.flush()
            # Stop log drainer
            try:
                log_queue.put({"__stop__": True})
            except Exception:
                pass
            try:
                log_thread.join(timeout=5.0)
            except Exception:
                pass

        # Reconstruct DatasetRecord list in job order
        ordered_records: List[DatasetRecord] = []
        for j in sorted(results_by_jid.keys()):
            res = results_by_jid[j]
            if res.get("accepted") and res.get("record"):
                try:
                    rd = res.get("record")
                    ordered_records.append(DatasetRecord(**rd))
                except Exception:
                    # If dataclass reconstruction fails, skip this record
                    continue
        return ordered_records

    def _record_skipped(self, art: ProblemArtifacts, attempts: List[Dict[str, Any]], reason: str, topic_path: Optional[List[str]] = None, duration_sec: Optional[float] = None) -> None:
        try:
            # Normalize think blocks for Together provider attempts
            norm_attempts: List[Dict[str, Any]] = []
            for a in attempts:
                pt = a.get("problem_text") or ""
                # Always strip any <think> blocks and carry them alongside
                think, content = extract_think_blocks(pt)
                norm_attempts.append({
                    "problem_text": content,
                    "problem_text_reasoning": (a.get("problem_text_reasoning") if a.get("problem_text_reasoning") is not None else think),
                    "feedback": a.get("feedback"),
                    "verification_error": a.get("verification_error"),
                    "parser_json": a.get("parser_json"),
                    "parser_raw": a.get("parser_raw"),
                    "parser_reasoning": a.get("parser_reasoning"),
                    # Proposed vs actual numeric details if present
                    "orig_numeric": a.get("orig_numeric"),
                    "parsed_numeric": a.get("parsed_numeric"),
                    "best_numeric_error": a.get("best_numeric_error"),
                    "vars": a.get("vars"),
                })
            self.skipped.append({
                "family": art.family,
                "eq_system_str": eqs_to_strings(art.eqs),
                "eq_system_ast": serialize_eqs_ast(art.eqs),
                "variables": art.variables,
                "domain": art.domain,
                "difficulty_meta": art.difficulty_meta,
                "sympy_src": art.sympy_src,
                "provider": self.provider,
                "model": self.model,
                "verify_mode": getattr(self, 'verify_mode', 'equivalence'),
                "numeric_tol": getattr(self, 'numeric_tol', 1e-6),
                "require_unique": getattr(self, 'require_unique', True),
                "parse_retries": getattr(self.rationalizer.cfg, 'parse_retries', 0),
                "parse_retry_delay": getattr(self.rationalizer.cfg, 'parse_retry_delay', 5.0),
                "reason": reason,
                "trace_retries": getattr(self, 'trace_retries', 0),
                "topic_path": topic_path,
                "topic_name": (topic_path[-1] if isinstance(topic_path, list) and topic_path else None),
                "attempts": norm_attempts,
                "duration_sec": duration_sec,
            })
        except Exception:
            pass

    def save_records(self, path: str, records: List[DatasetRecord], fmt: str = "jsonl", indent: int = 2) -> None:
        fmt = (fmt or "jsonl").lower()
        # Ensure parent directory exists
        try:
            Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        if fmt == "json":
            data = [_to_jsonable(asdict(r)) for r in records]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=indent)
            return
        # default: jsonl (one object per line)
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(_to_jsonable(asdict(r)), ensure_ascii=False, indent=indent))
                f.write("\n")

    # Back-compat helper
    def save_jsonl(self, path: str, records: List[DatasetRecord]) -> None:
        self.save_records(path, records, fmt="jsonl")


# -------------------------- Parallel worker --------------------------

def _worker_run_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """Worker entrypoint to generate and verify one record.

    Returns a dict with keys: job_id, accepted (bool), record (as dict or None),
    skipped (list), elapsed (float).
    """
    t0 = time.time()
    job_id = job.get("job_id")
    try:
        # Per-job determinism
        try:
            random.seed(int(job.get("seed", 0)))
        except Exception:
            pass
        # Instantiate a fresh generator in the subprocess
        gen = DatasetGenerator(
            use_llm=bool(job.get("use_llm", False)),
            provider=str(job.get("provider", "ambient")),
            model=str(job.get("model", "deepseek-ai/DeepSeek-R1")),
            verify_mode=str(job.get("verify_mode", "equivalence")),
            numeric_tol=float(job.get("numeric_tol", 1e-6)),
            require_unique=bool(job.get("require_unique", True)),
            parse_retries=int(job.get("parse_retries", 0)),
            parse_retry_delay=float(job.get("parse_retry_delay", 5.0)),
            trace_retries=int(job.get("trace_retries", 2)),
            save_roundtrip=bool(job.get("save_roundtrip", False)),
            student_solve=int(job.get("student_solve", 0)),
            use_encyclopedia_topics=bool(job.get("use_encyclopedia_topics", False)),
            encyclopedia_file=job.get("encyclopedia_file"),
            story_mode=str(job.get("story_mode", "natural")),
        )
        # Configure queue-based logging in parallel mode
        try:
            gen.rationalizer.llm_log_path = None  # avoid per-process file writes
            gen.rationalizer.llm_log_queue = job.get("log_queue")
            gen.rationalizer.log_extra = {"run_id": job.get("run_id"), "job_id": job_id}
        except Exception:
            pass

        fam = str(job.get("family"))
        mode = str(job.get("mode", "standard"))
        seed_val = int(job.get("seed", 0))

        art = gen.generate_one(fam)
        # Optional topic context
        chosen_topic_path: Optional[List[str]] = None
        if gen.use_encyclopedia_topics:
            try:
                chosen_topic_path = gen._choose_topic_path()
            except Exception:
                chosen_topic_path = None

        # Build story
        if gen.use_llm:
            try:
                story_raw = gen.rationalizer.equation_to_story(art, topic_path=chosen_topic_path)
            except Exception as e:
                story_raw = gen._templated_story(art) + f" [LLM error: {e}]"
        else:
            story_raw = gen._templated_story(art)
        # Strip think blocks
        story_reasoning = None
        try:
            story_reasoning, story = extract_think_blocks(story_raw)
        except Exception:
            story_reasoning = None
        if not story_reasoning:
            try:
                rc = getattr(gen.rationalizer, 'last_reasoning_content', None)
                if rc:
                    story_reasoning = rc
            except Exception:
                pass
        # Fallback for empty story after stripping
        if not story or not story.strip():
            try:
                if story_reasoning and isinstance(story_reasoning, str):
                    import re as _re
                    story = _re.sub(r"</?think>", "", story_reasoning).strip()
            except Exception:
                pass
            if not story or not story.strip():
                story = (story_raw or "").strip()

        # Optional random fact
        rand_fact: Optional[str] = None
        if gen.use_encyclopedia_topics and chosen_topic_path and gen.use_llm and gen.client is not None:
            try:
                topic_str = " / ".join([t for t in chosen_topic_path if t])
                sys_prompt = (
                    "You state concise, verifiable facts. Return one factual sentence about the topic. "
                    "No preface, no Markdown, no lists."
                )
                user_prompt = (
                    "Topic path:\n" + topic_str + "\n\n" +
                    "Write one random fact about this topic (<= 25 words)."
                )
                rf_raw = gen.rationalizer._complete(sys_prompt, user_prompt, max_tokens=int(gen.rationalizer.cfg.max_tokens), temperature=0.7, tag="random_fact")
                think, rf_text = extract_think_blocks(rf_raw)
                rand_fact = ((rf_text if rf_text is not None else rf_raw) or "").strip()
                if rand_fact:
                    m = re.search(r"([\s\S]*?[\.!?])\s", rand_fact + " ")
                    if m:
                        rand_fact = m.group(1).strip()
                    rand_fact = rand_fact.strip('`').strip()
            except Exception:
                rand_fact = None

        # Round-trip verify (if LLM used)
        verified, fb, verr, rt_meta = gen._roundtrip_verify(story, art)
        if gen.use_llm and gen.rationalizer.cfg.require_roundtrip and not verified:
            attempts_meta: List[Dict[str, Any]] = []
            attempts_meta.append({
                "problem_text": story,
                "problem_text_reasoning": story_reasoning,
                "feedback": fb,
                "verification_error": verr,
                "parser_json": (rt_meta.get("parser_json") if isinstance(rt_meta, dict) else None),
                "parser_raw": (rt_meta.get("parser_raw") if isinstance(rt_meta, dict) else None),
                "parser_reasoning": (rt_meta.get("parser_reasoning") if isinstance(rt_meta, dict) else None),
                "orig_numeric": (rt_meta.get("orig_numeric") if isinstance(rt_meta, dict) else None),
                "parsed_numeric": (rt_meta.get("parsed_numeric") if isinstance(rt_meta, dict) else None),
                "best_numeric_error": (rt_meta.get("best_numeric_error") if isinstance(rt_meta, dict) else None),
                "vars": (rt_meta.get("vars") if isinstance(rt_meta, dict) else None),
            })
            try:
                if gen.client is not None:
                    story2_raw = gen.rationalizer.equation_to_story(art, topic_path=chosen_topic_path)
                    story2_reasoning, story2 = extract_think_blocks(story2_raw)
                    if not story2_reasoning:
                        try:
                            rc2 = getattr(gen.rationalizer, 'last_reasoning_content', None)
                            if rc2:
                                story2_reasoning = rc2
                        except Exception:
                            pass
                    verified2, fb2, verr2, rt_meta2 = gen._roundtrip_verify(story2, art)
                    if verified2:
                        story, verified, fb, verr, rt_meta = story2, verified2, fb2, verr2, rt_meta2
                        story_reasoning = story2_reasoning
                    else:
                        attempts_meta.append({
                            "problem_text": story2,
                            "problem_text_reasoning": story2_reasoning,
                            "feedback": fb2,
                            "verification_error": verr2,
                            "parser_json": (rt_meta2.get("parser_json") if isinstance(rt_meta2, dict) else None),
                            "parser_raw": (rt_meta2.get("parser_raw") if isinstance(rt_meta2, dict) else None),
                            "parser_reasoning": (rt_meta2.get("parser_reasoning") if isinstance(rt_meta2, dict) else None),
                            "orig_numeric": (rt_meta2.get("orig_numeric") if isinstance(rt_meta2, dict) else None),
                            "parsed_numeric": (rt_meta2.get("parsed_numeric") if isinstance(rt_meta2, dict) else None),
                            "best_numeric_error": (rt_meta2.get("best_numeric_error") if isinstance(rt_meta2, dict) else None),
                            "vars": (rt_meta2.get("vars") if isinstance(rt_meta2, dict) else None),
                        })
                        elapsed = time.time() - t0
                        gen._record_skipped(art, attempts_meta, reason="verification_failed", topic_path=chosen_topic_path, duration_sec=elapsed)
                        return {"job_id": job_id, "accepted": False, "record": None, "skipped": gen.skipped, "elapsed": elapsed}
                else:
                    elapsed = time.time() - t0
                    gen._record_skipped(art, attempts_meta, reason="no_llm_client", topic_path=chosen_topic_path, duration_sec=elapsed)
                    return {"job_id": job_id, "accepted": False, "record": None, "skipped": gen.skipped, "elapsed": elapsed}
            except Exception as e:
                elapsed = time.time() - t0
                gen._record_skipped(art, attempts_meta, reason=f"exception: {e}", topic_path=chosen_topic_path, duration_sec=elapsed)
                return {"job_id": job_id, "accepted": False, "record": None, "skipped": gen.skipped, "elapsed": elapsed}

        # Build record
        eq_strs = eqs_to_strings(art.eqs)
        sol_eval = eval_solution(art.solution)
        item_a = None
        item_b = None
        item_c = None
        item_d = None
        item_c_reason = None
        item_d_reason = None
        trace_retry_count = 0
        story_numbers_ok: Optional[bool] = None
        student_attempts: Optional[List[Dict[str, Any]]] = None
        student_success_count: Optional[int] = None
        if mode == "posthoc":
            uniq = {}
            for k, v in sol_eval.items():
                if isinstance(v, list) and v:
                    try:
                        uniq[k] = sorted(v, key=lambda x: (abs(x), x))[0]
                    except Exception:
                        uniq[k] = v[0]
                else:
                    uniq[k] = v
            item_a = list(eq_strs)
            item_b = uniq
            item_c = story
            item_c_reason = story_reasoning
            try:
                nums_in_eqs = re.findall(r"[-+]?\d+(?:\.\d+)?", " ".join(item_a)) if item_a else []
                story_numbers_ok = all(n in (item_c or "") for n in nums_in_eqs)
            except Exception:
                story_numbers_ok = None
            if gen.use_llm:
                for attempt_idx in range(int(gen.trace_retries) + 1):
                    try:
                        item_d_raw = gen.rationalizer.reasoning_trace(item_a, item_c, item_b)
                        think_d, item_d_text = extract_think_blocks(item_d_raw)
                        if item_d_text:
                            item_d_candidate = item_d_text
                            item_d_reason = think_d
                        else:
                            item_d_candidate = item_d_raw
                            item_d_reason = None
                        bad = False
                        for var, ans in (item_b or {}).items():
                            try:
                                m = re.search(rf"\b{re.escape(var)}\s*=\s*([-+]?\d+(?:\.\d+)?)", item_d_candidate)
                                if m:
                                    val = float(m.group(1))
                                    if abs(val - float(ans)) > gen.numeric_tol:
                                        bad = True
                                        break
                            except Exception:
                                continue
                        if not bad:
                            item_d = item_d_candidate
                            trace_retry_count = attempt_idx
                            break
                        trace_retry_count = attempt_idx
                    except Exception as e:
                        item_d = f"[reasoning generation error: {e}]"

        rec = DatasetRecord(
            id=str(uuid.uuid4()),
            family=art.family,
            problem_text=story,
            topic_path=chosen_topic_path,
            topic_name=(chosen_topic_path[-1] if isinstance(chosen_topic_path, list) and chosen_topic_path else None),
            variables=art.variables,
            domain=art.domain,
            eq_system_str=eq_strs,
            eq_system_ast=serialize_eqs_ast(art.eqs),
            sympy_src=art.sympy_src,
            solution=art.solution,
            solution_eval=sol_eval,
            difficulty_meta=art.difficulty_meta,
            roundtrip_verified=(None if ((gen.verify_mode == 'none') or (not gen.use_llm)) else verified),
            verifier_feedback=fb,
            verification_error=verr,
            provider=gen.provider,
            model=gen.model,
            seed=seed_val,
            verify_mode=gen.verify_mode,
            story_numbers_ok=story_numbers_ok,
            trace_retry_count=(trace_retry_count if mode == 'posthoc' else None),
            problem_text_reasoning=story_reasoning if mode != "posthoc" else None,
            random_fact=rand_fact,
            item_a_equations=item_a,
            item_b_numeric_answer=item_b,
            item_c_problem=item_c,
            item_d_reasoning=item_d,
            item_c_problem_reasoning=item_c_reason,
            item_d_reasoning_reasoning=item_d_reason,
            item_d_reasoning_steps=(
                [ln.strip() for ln in (item_d or '').splitlines() if re.match(r"^\s*(?:\d+\.|Step\s+\d+|\-\s)", ln)]
                if (mode == 'posthoc') else None
            ),
        )
        # Attach diagnostics
        if gen.use_llm:
            try:
                rec.verifier_orig_numeric = (rt_meta.get("orig_numeric") if isinstance(rt_meta, dict) else None)
                rec.verifier_parsed_numeric = (rt_meta.get("parsed_numeric") if isinstance(rt_meta, dict) else None)
                rec.verifier_best_numeric_error = (rt_meta.get("best_numeric_error") if isinstance(rt_meta, dict) else None)
                rec.verifier_vars = (rt_meta.get("vars") if isinstance(rt_meta, dict) else None)
                rec.roundtrip_parser_reasoning = (rt_meta.get("parser_reasoning") if isinstance(rt_meta, dict) else None)
            except Exception:
                pass
        if gen.save_roundtrip and gen.use_llm:
            try:
                rec.roundtrip_parser_json = (rt_meta.get("parser_json") if isinstance(rt_meta, dict) else None)
                rec.roundtrip_parser_raw = (rt_meta.get("parser_raw") if isinstance(rt_meta, dict) else None)
            except Exception:
                pass

        if gen.use_llm and gen.student_solve and int(gen.student_solve) > 0:
            student_attempts = []
            success = 0
            for _i in range(int(gen.student_solve)):
                try:
                    solve_raw = gen.rationalizer.student_solve(story)
                    think, solve_text = extract_think_blocks(solve_raw)
                    fa_map = {}
                    m = None
                    for line in (solve_text or solve_raw).splitlines()[::-1]:
                        if line.strip().lower().startswith("final answer:"):
                            m = line.strip()[len("final answer:"):].strip()
                            break
                    if m:
                        for var in art.variables.keys():
                            rx = re.compile(rf"\b{re.escape(var)}\s*[:=]\s*([-+]?\d+(?:\.\d+)?)")
                            mm = rx.search(m)
                            if mm:
                                try:
                                    fa_map[var] = float(mm.group(1))
                                except Exception:
                                    pass
                    match = None
                    verr_student = None
                    try:
                        tol = float(gen.numeric_tol)
                        have_all = all(k in fa_map for k in art.variables.keys())
                        if have_all:
                            on = {k: float(sp.N(v if not isinstance(v, (list, tuple)) else v[0])) for k, v in art.solution.items()}
                            max_err = 0.0
                            for k, v in on.items():
                                err = abs(float(fa_map.get(k)) - v)
                                max_err = max(max_err, err)
                            verr_student = max_err
                            match = (max_err <= tol)
                    except Exception:
                        pass
                    student_attempts.append({
                        "text": solve_text or solve_raw,
                        "reasoning": think,
                        "final_answer": m,
                        "final_answer_map": fa_map if fa_map else None,
                        "numeric_match": match,
                        "verification_error": verr_student,
                    })
                    if match:
                        success += 1
                except Exception as _e:
                    student_attempts.append({
                        "text": None,
                        "reasoning": None,
                        "final_answer": None,
                        "error": str(_e),
                    })
            rec.student_attempts = student_attempts
            rec.student_success_count = success

        elapsed = time.time() - t0
        try:
            rec.duration_sec = elapsed
        except Exception:
            pass
        return {"job_id": job_id, "accepted": True, "record": asdict(rec), "skipped": [], "elapsed": elapsed}
    except Exception as e:
        elapsed = time.time() - t0
        # Build minimal skip artifact
        try:
            fam = str(job.get("family"))
        except Exception:
            fam = "unknown"
        skip = {
            "family": fam,
            "provider": job.get("provider"),
            "model": job.get("model"),
            "verify_mode": job.get("verify_mode"),
            "reason": f"exception: {e}",
            "attempts": [],
            "duration_sec": elapsed,
        }
        return {"job_id": job_id, "accepted": False, "record": None, "skipped": [skip], "elapsed": elapsed}


# --------- CLI helper for quick preview ----------
if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_family", type=int, default=3)
    parser.add_argument("--families", type=str, default=None, help="Families to generate (e.g., 'linear_1var|quadratic|logexp'). Use '|' or ',' as separators.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--use_llm", action="store_true")
    parser.add_argument("--provider", type=str, default="ambient", choices=["ambient", "together", "openai", "google", "grok"], help="LLM provider when --use_llm is set")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1", help="Provider model name")
    parser.add_argument("--verify-mode", type=str, default="equivalence", choices=["equivalence", "numeric", "robustOptimal", "none"], help="Verification strategy for round-trip")
    parser.add_argument("--story-mode", type=str, default="natural", choices=["natural", "facts"], help="Story prompt mode: natural (no algebraic phrasing) or facts (equation-anchored)")
    parser.add_argument("--numeric-tol", type=float, default=1e-6, help="Tolerance for numeric verification")
    parser.add_argument("--allow-nonunique", action="store_true", help="Allow non-unique original solutions in equivalence mode")
    parser.add_argument("--parse-retries", type=int, default=0, help="Times to retry JSON parse call on invalid JSON")
    parser.add_argument("--parse-retry-delay", type=float, default=5.0, help="Seconds to wait between parse retries")
    parser.add_argument("--out", type=str, default="algebraTest/algebra_synth_dataset.jsonl")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar output")
    parser.add_argument("--format", type=str, default="jsonl", choices=["jsonl", "json"], help="Output format: jsonl or pretty json array")
    parser.add_argument("--indent", type=int, default=2, help="Indent level for pretty JSON when --format json")
    parser.add_argument("--mode", type=str, default="standard", choices=["standard", "posthoc"], help="Generation mode: standard or post hoc rationalization")
    parser.add_argument("--jobs", type=int, default=1, help="Max parallel worker processes (up to 10)")
    parser.add_argument("--trace-max-tokens", type=int, default=18000, help="Max tokens for reasoning trace generation")
    parser.add_argument("--trace-temperature", type=float, default=0.2, help="Temperature for reasoning trace generation")
    parser.add_argument("--trace-retries", type=int, default=2, help="Max attempts to regenerate reasoning when numeric mismatch is detected")
    parser.add_argument("--save-roundtrip", action="store_true", help="Save story→equations parser JSON/raw output into each record")
    parser.add_argument("--student-solve", type=int, default=0, help="Ask the LLM to solve each problem N times and record attempts")
    parser.add_argument("--use-encyclopedia-topics", action="store_true", help="Diversify scenarios with topics from the encyclopedia file")
    parser.add_argument("--encyclopedia-file", type=str, default=str(Path.cwd() / "Encyclopedia70K_20250810_174254.json"), help="Path to encyclopedia topic tree JSON")
    args = parser.parse_args()

    # Families to generate
    if getattr(args, 'families', None):
        raw = (args.families or "").replace(',', '|')
        fams = [t.strip() for t in raw.split('|') if t.strip()]
        # Fallback to defaults if parsing yields empty list
        if not fams:
            fams = [
                "linear_1var",
                "linear_2var_system",
                "quadratic",
                "quadratic_complete_square",
                "rational_equation",
                "logexp",
            ]
    else:
        fams = [
            "linear_1var",
            "linear_2var_system",
            "quadratic",
            "quadratic_complete_square",
            "rational_equation",
            "logexp",
        ]
    # Establish mode early for downstream defaults
    mode = getattr(args, 'mode', 'standard') if hasattr(args, 'mode') else 'standard'
    # Default verification relaxation in posthoc mode
    if mode == 'posthoc' and args.verify_mode == 'equivalence':
        args.verify_mode = 'none'

    gen = DatasetGenerator(
        use_llm=args.use_llm,
        provider=args.provider,
        model=args.model,
        verify_mode=args.verify_mode,
        numeric_tol=args.numeric_tol,
        require_unique=not args.allow_nonunique,
        parse_retries=args.parse_retries,
        parse_retry_delay=args.parse_retry_delay,
        trace_retries=args.trace_retries,
        save_roundtrip=args.save_roundtrip,
        student_solve=args.student_solve,
        use_encyclopedia_topics=args.use_encyclopedia_topics,
        encyclopedia_file=args.encyclopedia_file,
        story_mode=getattr(args, 'story_mode', 'natural'),
    )
    # Apply trace config
    try:
        gen.rationalizer.cfg.trace_max_tokens = int(args["trace_max_tokens"])  # type: ignore
    except Exception:
        try:
            gen.rationalizer.cfg.trace_max_tokens = int(getattr(args, 'trace_max_tokens'))
        except Exception:
            pass
    try:
        gen.rationalizer.cfg.trace_temperature = float(getattr(args, 'trace_temperature'))
    except Exception:
        pass
    # Default to pretty JSON and a posthoc filename when mode=posthoc and user didn't override
    if mode == 'posthoc' and args.format == 'jsonl':
        args.format = 'json'
    if mode == 'posthoc' and args.out == 'algebraTest/algebra_synth_dataset.jsonl':
        args.out = 'algebraTest/posthoc_dataset.json'

    # Resolve output path: if user passed a directory, append a timestamped default filename
    def _resolve_out_path(out_str: str, fmt: str, provider: str, verify_mode: str, mode: str) -> str:
        fmt_l = (fmt or "jsonl").lower()
        ext = ".json" if fmt_l == "json" else ".jsonl"
        p = Path(out_str).expanduser()
        # Treat as directory if endswith path separator or path exists and is a directory
        if out_str.endswith(os.sep) or (p.exists() and p.is_dir()):
            ts = time.strftime("%Y%m%d_%H%M%S")
            prefix = "posthoc" if mode == "posthoc" else "algebra_synth"
            fname = f"{prefix}_{provider}_{verify_mode}_{ts}{ext}"
            return str(p / fname)
        return str(p)

    out_path_str = _resolve_out_path(args.out, args.format, args.provider, args.verify_mode, mode)
    recs = gen.build_records(fams, n_per_family=args.per_family, seed=args.seed, progress=not args.no_progress, mode=mode, jobs=args.jobs)
    gen.save_records(out_path_str, recs, fmt=args.format, indent=args.indent)
    print(f"Wrote {len(recs)} items to {out_path_str}")

    # Always write skipped attempts (if any) to a sibling file with _skipped suffix
    try:
        from pathlib import Path
        out_path = Path(out_path_str)
        skipped_path = out_path.with_name(out_path.stem + "_skipped" + (".json" if out_path.suffix.lower() not in (".json", ".jsonl") else out_path.suffix))
        stats_path = out_path.with_name(out_path.stem + "_skipped_stats.json")
        if gen.skipped:
            # Save skipped items as pretty JSON
            skipped_path.parent.mkdir(parents=True, exist_ok=True)
            with open(skipped_path, "w", encoding="utf-8") as f:
                json.dump(gen.skipped, f, ensure_ascii=False, indent=args.indent)
            # Build stats grouped by family
            by_family = {}
            reasons = {}
            for it in gen.skipped:
                fam = it.get("family", "unknown")
                by_family.setdefault(fam, {"count": 0, "reasons": {}})
                by_family[fam]["count"] += 1
                r = it.get("reason", "unknown")
                by_family[fam]["reasons"][r] = by_family[fam]["reasons"].get(r, 0) + 1
                reasons[r] = reasons.get(r, 0) + 1
            run_params = {
                "provider": gen.provider,
                "model": gen.model,
                "verify_mode": getattr(gen, 'verify_mode', 'equivalence'),
                "numeric_tol": getattr(gen, 'numeric_tol', 1e-6),
                "require_unique": getattr(gen, 'require_unique', True),
                "parse_retries": getattr(gen.rationalizer.cfg, 'parse_retries', 0),
                "parse_retry_delay": getattr(gen.rationalizer.cfg, 'parse_retry_delay', 5.0),
                "mode": mode,
                "per_family": args.per_family,
                "seed": args.seed,
            }
            stats = {
                "skipped_count": len(gen.skipped),
                "reasons": reasons,
                "by_family": by_family,
                "run_params": run_params,
            }
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=args.indent)
            # Print stats JSON to console
            print("Skipped stats:")
            print(json.dumps(stats, ensure_ascii=False, indent=args.indent))
            print(f"Saved skipped items to {skipped_path} and stats to {stats_path}")
        else:
            print("No skipped items.")
    except Exception as e:
        print(f"Failed writing skipped files: {e}")
