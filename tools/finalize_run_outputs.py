#!/usr/bin/env python3
"""
Finalize Run Outputs (Graphs + Summaries)
=========================================

Small CLI wrapper around `algebra_dataset_benchmarking_tool.finalize_run_outputs`.

Usage examples:
- Regenerate graphs only (safe, no missed-by-all build):
  python3 tools/finalize_run_outputs.py --run-dir algebraTest/benchmark_runs/general_runV7

- Build missed-by-all files (streaming, lite-first, base-limited):
  python3 tools/finalize_run_outputs.py \
    --run-dir algebraTest/benchmark_runs/general_runV7 \
    --build-missed --build-missed-detailed --prefer-lite-only

- Override dataset base for older runs without run_meta.json:
  python3 tools/finalize_run_outputs.py \
    --run-dir algebraTest/benchmark_runs/general_runV6 \
    --dataset-base AGSM8K-V3-prod --build-missed

- Exclude providers/models (regex; tests against `provider:model:reasoning` and plain `model`):
  python3 tools/finalize_run_outputs.py \
    --run-dir algebraTest/benchmark_runs/general_runV7 \
    --exclude "^ambient:" --exclude "^openai:.*o3"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import sys
from pathlib import Path as _Path

# Ensure project root is importable when running from tools/
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from algebra_dataset_benchmarking_tool import finalize_run_outputs


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Regenerate merged summaries and graphs for a run directory.")
    ap.add_argument(
        "--run-dir",
        required=True,
        help="Path to run directory (e.g., algebraTest/benchmark_runs/<RUN_NAME>)",
    )
    ap.add_argument(
        "--exclude",
        action="append",
        default=None,
        help=(
            "Regex of specs to exclude from graphs/merge. Can be repeated. "
            "Patterns test against 'provider:model:reasoning' and plain 'model'."
        ),
    )
    ap.add_argument(
        "--sort-bars-desc",
        action="store_true",
        help="Order bar charts by descending values (left→right highest to lowest)",
    )
    ap.add_argument(
        "--build-missed",
        action="store_true",
        help="Also (re)build missed_by_all.json (opt-in to avoid heavy dataset reads)",
    )
    ap.add_argument(
        "--build-missed-detailed",
        action="store_true",
        help="Also build missed_by_all_detailed.json (requires --build-missed)",
    )
    ap.add_argument(
        "--no-dataset-backfill",
        action="store_true",
        help="Skip dataset backfill when building missed_by_all.json (use records only)",
    )
    ap.add_argument(
        "--prefer-lite-only",
        action="store_true",
        help="When building missed files, only use *_lite.json sources for dataset lookup",
    )
    ap.add_argument(
        "--dataset-base",
        default=None,
        help="Restrict dataset lookup to this base prefix (e.g., AGSM8K-V3-prod). Overrides run_meta.json if set.",
    )
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_dir = Path(args.run_dir)
    patterns = args.exclude if args.exclude else None
    finalize_run_outputs(
        run_dir,
        exclude=patterns,
        sort_bars_desc=bool(args.sort_bars_desc),
        build_missed=bool(args.build_missed),
        build_missed_detailed=bool(args.build_missed_detailed),
        dataset_backfill=not bool(args.no_dataset_backfill),
        prefer_lite=bool(args.prefer_lite_only) or True,
        dataset_base_name=args.dataset_base,
    )
    print(f"Finalized outputs under: {run_dir}")
    if patterns:
        print(f"Excluded patterns: {patterns}")
    if args.build_missed:
        print("Rebuilt missed_by_all.json")
        if args.build_missed_detailed:
            print("Rebuilt missed_by_all_detailed.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
