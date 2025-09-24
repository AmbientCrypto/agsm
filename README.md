# agsm
Contains the Ambient GSM, a dynamically generated algebra word problem benchmark for LLMs. See also: reproducibleCommandIllustrations.txt


### AGSM (Ambient Grade School Math) Overview

- Generation: `algebra_dataset_generator.py` and `algebra_dataset_generatorV2.py` programmatically build solvable equations with SymPy, realize them into word problems with an LLM, and verify by round‑trip parsing (story → equations → solution).
- Difficulty: `algebra_dataset_generator_deterministic_composite.py` composes n verified leaf problems into a single multi‑step question whose final answer is a linear expression of the n sub‑answers. Level n ≡ n sub‑problems.
- Sieve (two‑stage):
  - Stage 1 (tractable leaves): `algebra_dataset_problem_siever.py` keeps problems that at least one model solves (beyond round‑trip parsing).
  - Stage 2 (easy‑filter): `algebra_dataset_problem_siever_easy_filter.py` culls problems a panel solves too easily; use `--successThreshold 50` with a weak panel to match the paper’s >50% criterion.
- Benchmark: `algebra_dataset_benchmarking_tool.py` runs the standardized AGSM harness: strict JSON `{"final_answer": ...}` with tolerant parsing fallbacks, percent‑error grading, per‑difficulty graphs, consensus regrade, and “expected vs actual” reference lines (naïve d=1 and WLS‑fit p^d).
- Narrative perturbations: when topic diversification is enabled, a short `random_fact` can be inserted before/after sub‑problems. This is semantics‑independent and does not change variables/constants.

### Generator (`algebra_dataset_generator.py`)

- Purpose: Create verified algebra word problems (linear, systems, quadratic, rational, log/exp).
- Verification modes:
  - equivalence: exact symbolic residual check with ground‑truth solution.
  - numeric: numeric solution matching (tolerance).
  - robustOptimal: structural proportional match with randomized identity testing + numeric fallback.
  - none: skip verification.
- Topic diversification: `--use-encyclopedia-topics` uses a hierarchical topic path from Encyclopedia70K to guide realistic settings; saved as `topic_path`/`topic_name`.
- Random topic fact (optional): when both `--use-encyclopedia-topics` and LLM usage are enabled, the generator makes a one‑off call to produce a concise, single‑sentence `random_fact` about the chosen topic and stores it in each record (field: `random_fact`).
- Story modes: `--story-mode` selects between two prompt styles:
  - `natural` (default): Converts relations into realistic narrative facts without algebraic phrasing.
  - `facts`: A more equation‑anchored style that stays close to the symbolic form while remaining natural.
- Logarithmic phrasing (tightened): For logarithmic families the generator now explicitly instructs stories to say “logarithmic (base k)”/“log base k” and to make the “logarithmic reading” equal to r. This avoids ambiguous phrasing like “on a base‑k scale it is r,” which the parser could misread as a direct equality.
- Round‑trip parsing (enhanced): Input stories are sanitized to plain text and parsed with a strict mode that enforces exact variable names (including underscores) when available. Verification can auto‑escalate to `robustOptimal` for multi‑root originals.
- Output formatting: `--format json` writes a pretty JSON array; `--format jsonl` writes one pretty‑printed JSON object per line.
- Families selection: `--families` lets you pick a subset (e.g., `linear_1var|quadratic|logexp`) instead of generating all defaults.
- Parallelism: `--jobs N` runs up to N processes in parallel (max 10). Parent process is the sole writer for output and logs, ensuring coherent JSON/JSONL artifacts. Results are ordered deterministically by job id.
- LLM logging: Every LLM request/response is appended to a JSONL log file under `logs/` (e.g., `logs/algebra_generator_llm_YYYYMMDD_HHMMSS.jsonl`). In parallel mode, workers push log entries to a queue and the parent writes a single log file. Each entry includes provider, model, max_tokens, temperature, full prompt (system/user), and the raw response or error. Errors include a reproducible `curl` command and set `timeout: true` when the error was a timeout.
- Optional captures:
  - `--save-roundtrip` saves parser JSON/raw/reasoning (story→equations).
  - `--student-solve N` runs N solution attempts and records reasoning, extracted answers, and numeric match.
- Directory‑friendly output: passing a directory to `--out` auto‑creates a timestamped filename.

Example:

```
python algebra_dataset_generator.py --per_family 3 --seed 42 --use_llm --jobs 8 \
  --provider together --model deepseek-ai/DeepSeek-R1 \
  --use-encyclopedia-topics --encyclopedia-file Encyclopedia70K_20250810_174254.json \
  --verify-mode robustOptimal --format json --out algebraTest/
```

Example (with random facts attached via topics):

```
python algebra_dataset_generator.py --per_family 5 --seed 7 --use_llm \
  --provider together --model deepseek-ai/DeepSeek-R1 \
  --use-encyclopedia-topics --format json --out algebraTest/dataset_with_facts.json
```
Each record may include `"random_fact": "…"` summarizing a short fact about the topic.

Example (facts story style with strict parsing/verification):

```
python algebra_dataset_generator.py --per_family 4 --seed 11 --use_llm \
  --provider together --model deepseek-ai/DeepSeek-R1 \
  --story-mode facts --verify-mode equivalence --format jsonl \
  --out algebraTest/algebra_synth_facts.jsonl

Example (restrict to selected families):

```
python algebra_dataset_generator.py --per_family 5 --families linear_1var|quadratic|logexp \
  --use_llm --provider together --model deepseek-ai/DeepSeek-R1 --format jsonl \
  --out algebraTest/algebra_synth_selected.jsonl
```
```



#### Logging, Progress, Streaming

The composer CLI now supports structured logging and a progress indicator:

- `--log-level LEVEL`: Set log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). Default: `INFO`.
- `--log-file PATH`: Optional file to write logs; still logs to stdout unless you redirect.
- `--no-progress`: Disable progress output (useful for CI logs).

If `tqdm` is installed, a live progress bar is shown. Otherwise, a simple "[i/N] ok:X skip:Y" line prints each iteration.

- Ambient client timeout: default 600s when used by the composer and dashboard server.
- LLM calls prefer streaming with a non‑streaming fallback. Use `--verbose-llm` to log prompts/responses/errors.

Optional install for nicer progress bar:
```bash
pip install tqdm
```

Examples:
```bash
# Verbose run with progress bar
python algebra_composer.py \
  --base-dataset algebraTest/robustOptimalTestrunAmbientV2.json \
  --composites 10 --subs-per-composite 2 --composition-mode fork_join \
  --operators affine,coef --log-level INFO

# Debug logs to a file, no progress output
python algebra_composer.py \
  --base-dataset algebraTest/robustOptimalTestrunAmbientV2.json \
  --composites 5 --composition-mode chain --operators log,ratio \
  --log-level DEBUG --log-file composer.log --no-progress

#### Symbolic Composition (LLM Path)

When `--use_llm` is enabled, the composer can build the parent equation symbolically over r1..rK (answers to sub‑problems) and generate a numberless story that references those sub‑answers by object names. The pipeline is:

- Map objects (LLM, JSON-only): each sub‑problem answer r_i is mapped to a short singular noun aligned to the topic, e.g., `{index:1,name:"apples",count:26}`.
- Compose symbolic equation: e.g., `Eq(r1*y + r2, r3)` or `Eq(y, r1 + r2 + r3)`.
- Generate story (LLM, numberless): writes a short paragraph that refers to “the number of <object> from Sub‑problem i” instead of stating constants/coefficients.
- Append original sub‑problems verbatim as context.
- Verify round‑trip (LLM): supplies the story and explicit numeric assignments for r_i; requests JSON-only SymPy equations; checks structural/numeric match to ground truth.

Retries & validation:
- Mapping: 1 retry if JSON invalid or counts don’t match; otherwise falls back to `item1`, `item2`, …
- Story: 2 retries if digits/number words leak, references are missing, or equation markers appear. Enforced constraints: no digits (except in “Sub-problem i”), no number words, no `Eq(…)`/operators in text, all referenced r_i used.
- Parser: 1 retry if JSON invalid or extraneous symbols occur.

Verbose LLM logging:
- Add `--verbose-llm` to log full LLM prompts/responses/errors for mapping, story, and parser steps.
  - Combine with `--log-file` to persist these logs for audits.

Example (LLM path):
```bash
python algebra_composer.py \
  --base-dataset algebraTest/robustOptimalTestrunAmbientV2.json \
  --composites 5 --subs-per-composite 3 --composition-mode fork_join \
  --operators affine_multi,coef,ratio \
  --use_llm --provider together --model deepseek-ai/DeepSeek-R1 \
  --verify-mode numeric --parse-retries 6 --parse-retry-delay 2.0 \
  --use-encyclopedia-topics --format json --out algebraTest/ \
  --log-level INFO --verbose-llm --save-roundtrip

```

### Deterministic Composites (No LLM)

- Purpose: Build composite linear problems that depend on the “single most positive” numeric answers from existing sub‑problems.
- Script: `algebra_dataset_generator_deterministic_composite.py`
- Input: Existing dataset file (`.json` or `.jsonl`) with `solution_eval` values.
- Output: JSON array of composites with fields `question`, `equation_template`, `deterministic_answer`, and `sub_components` (includes coefficients, chosen answers, and original sub‑problem JSON).

Notes
- The original sub‑problem JSON is preserved under `sub_components[*].source`. If your base dataset was generated with topics + LLM, the `random_fact` field (when present) will also be included there and can be used by downstream tools.

Usage
- Basic: `python algebra_dataset_generator_deterministic_composite.py --input algebraTest/algebra_synth_together_robustOptimal_20250907_135541.json --runs 50 --subproblems 3 --seed 42 --output algebraTest/deterministic_composites.json`
- With replacement: add `--with-replacement` to allow repeated sub‑problems within one composite.
- Coefficients: control with `--coeff-min` and `--coeff-max` (inclusive).
- Output format: `--format jsonl` for JSONL (default: pretty JSON array), `--indent` for JSON indentation.

Key Behavior
- Selection: For each sub‑problem’s `solution_eval`, consider all numeric values (including lists for multi‑root problems) and pick the numerically largest (most positive). If none positive exist, pick the largest overall (least negative).
- Coefficients: Random integers in the inclusive range `[coeff_min, coeff_max]`.
- Question: A linear expression over `sub_1..sub_K` followed by each sub‑problem’s text.


## Algebra Benchmarking Tool

Benchmarks multiple providers/models across the algebra datasets by difficulty and generates rich graphs.

Usage
- Basic (ground truth grading):
  - `python3 algebra_dataset_benchmarking_tool.py --n-per-file 25 --error-pct 1.0 --graph --run-name smoke_benchmark`
- Parallel run (throughput tuning):
  - `--workers-per-provider N` and `--provider-max-parallel M`
- Providers/models list:
  - `--providers-file externalProvidersAndModelsV6.json`
- Selection filter (regex):
  - `--select "^(openai|together):"`

Graph customization (mirrors finalize_run_outputs)
- `--sort-bars-desc`: sort bars left→right highest→lowest.
- `--graph-provider-rename ambient:zai`: rename providers in labels/titles (repeatable).
- `--graph-omit-run-name-in-titles`: cleaner titles without the run directory.
- `--graph-show-grading-type-in-titles`: show Ground truth vs Consensus in titles.
- `--graph-randomize-provider-line-colors`: deterministic random colors per provider’s model series.
- `--graph-filtered-difficulties 1,10`: grouped bars per model for specific difficulties (defaults to d1,d10).
- `--graph-consensus`: also compute consensus regrade and emit `_consensusGrade` graphs.

Example (matches prior manual finalize settings)
```
python3 algebra_dataset_benchmarking_tool.py \
  --base-name AGSM8K-V3-prod --n-per-file 50 --error-pct 0.5 \
  --workers-per-provider 10 --provider-max-parallel 10 \
  --request-timeout 640 --max-tokens 31000 \
  --run-name general_runV7 --providers-file externalProvidersAndModelsV6.json \
  --graph --sort-bars-desc \
  --graph-provider-rename ambient:zai \
  --graph-omit-run-name-in-titles \
  --graph-show-grading-type-in-titles \
  --graph-randomize-provider-line-colors \
  --graph-filtered-difficulties 1,10
```

Outputs (algebraTest/benchmark_runs/<run_name>/)
- All-model line: `graph_<run>*.png`
- Per-provider lines and per-model series
- Providers/model overall bars, providers overall bar
- Grouped d1 vs d10 bars: `graph_<run>_providers_models_levels_grouped_d1_d10.png`
- Missed-by-all: `graph_<run>_missed_by_all*.png` (requires `missed_by_all.json` to exist)
- Missed-by-all JSON (optional; built via finalize): `missed_by_all.json` and `missed_by_all_detailed.json` (adds `equation_template`, `eq_system_str`, `solution_eval` when present)
- Zero-incorrect by difficulty: `graph_<run>_zero_incorrect_by_difficulty*.png`
- Consensus variants (when enabled): suffix `_consensusGrade`
  - Missed-by-all JSON consensus: `missed_by_all_consensus.json` and `missed_by_all_consensus_detailed.json`
- Run metadata: `run_meta.json` (records `base_name`, sampling, provider file). Finalize uses this to scope dataset lookups to the correct base.

Per‑Model Reference Lines
- For each provider/model, we render additional "ref" graphs that overlay reference lines on top of actual series (per reasoning level):
  - Ground truth: `graph_<run>_provider_<provider>_model_<model>_ref.png`
  - Consensus: `graph_<run>_provider_<provider>_model_<model>_ref_consensusGrade.png`
  - Per‑reasoning variants: add `_reasoning_<reasoning>` before the suffix.
- Reference lines per series:
  - Naive(d1) baseline (slate gray dashed): pass@d = 100 × (pass1^d), where pass1 is the pass rate at difficulty 1.
  - Fit(all d) baseline (orange dashed): estimate p by fitting log(Pd) ≈ d·log(p) using all difficulties (weighted by counts with smoothing Pd=(Sd+0.5)/(Nd+1)), then pass@d = 100 × (p^d). We also render a light orange 95% confidence band using a weighted residual variance estimate (delta method on the exp transform).
  - A short caption is included beneath each ref chart: “Baselines: naive(d1) pass(d)=100·(pass1^d); fit(all d) via WLS on log(Pd) with smoothing, 95% band by delta method.”
  - Both baselines appear on the same ref graphs alongside the actual series (consistent provider/model colors).

Regenerate graphs without rerunning models (safe defaults)
```
python3 - <<'PY'
from algebra_dataset_benchmarking_tool import finalize_run_outputs
finalize_run_outputs(
    'algebraTest/benchmark_runs/general_runV7',
    sort_bars_desc=True,
    provider_renames={'ambient':'zai'},
    omit_run_name_in_titles=True,
    show_grading_type_in_titles=True,
    randomize_provider_line_colors=True,
    filtered_difficulties=[1,10],
)
PY
```

Build missed-by-all safely (streaming, base-limited)
```
python3 - <<'PY'
from algebra_dataset_benchmarking_tool import finalize_run_outputs
finalize_run_outputs(
    'algebraTest/benchmark_runs/general_runV7',
    build_missed=True,
    build_missed_detailed=True,
    dataset_backfill=True,
    prefer_lite=True,
    # optional if run_meta.json is present
    dataset_base_name='AGSM8K-V3-prod',
    # plus any graph options you like...
    do_consensus=True,
    fallback_mode='exclude',
    filtered_difficulties=[3,6,9],
)
PY
```