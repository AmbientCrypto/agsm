#!/usr/bin/env python3
"""
AGSM Lite Regenerator
=====================

Converts full AGSM composite files (e.g., AGSM8K_diff1.json) into enriched lite
files (AGSM8K_diff1_lite.json) that contain:

- id
- question
- deterministic_answer
- sub_components[sub_i]:
  - eq_system_str
  - eq_system_ast
  - sympy_src
  - solution
  - solution_eval

Usage:
  python3 tools/agsm_lite_regenerator.py \
    --base-dir algebraTest --base-name AGSM8K --first 1 --last 10

Notes:
- This reads the full files as pretty JSON arrays (not JSONL).
- It is safe to re-run; output files are overwritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _lite_sub(sub_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Project a sub-component to the lite schema from its `source` field."""
    src = sub_obj.get("source") or {}
    if not isinstance(src, dict):
        src = {}
    return {
        "eq_system_str": list(src.get("eq_system_str") or []),
        "eq_system_ast": list(src.get("eq_system_ast") or []),
        "sympy_src": src.get("sympy_src"),
        "solution": src.get("solution"),
        "solution_eval": src.get("solution_eval"),
    }


def regenerate_one(in_path: Path, out_path: Path) -> None:
    data = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {in_path}")
    out_items = []
    for rec in data:
        if not isinstance(rec, dict):
            continue
        subs_in = rec.get("sub_components") or {}
        subs_out: Dict[str, Any] = {}
        # Keep sub_i order by numeric index if available
        for key in sorted(subs_in.keys(), key=lambda k: (int(k.split("_")[-1]) if k and k.split("_")[-1].isdigit() else 1_000_000, k)):
            subs_out[key] = _lite_sub(subs_in[key] or {})
        out_items.append({
            "id": rec.get("id"),
            "question": rec.get("question"),
            "deterministic_answer": rec.get("deterministic_answer"),
            "sub_components": subs_out,
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_items, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Regenerate AGSM lite files from full composite JSON arrays.")
    ap.add_argument("--base-dir", default="algebraTest", help="Directory containing AGSM files (default: algebraTest)")
    ap.add_argument("--base-name", default="AGSM8K", help="Base filename prefix (default: AGSM8K)")
    ap.add_argument("--first", type=int, default=1, help="First difficulty index (default: 1)")
    ap.add_argument("--last", type=int, default=10, help="Last difficulty index (default: 10)")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    any_written = False
    for diff in range(int(args.first), int(args.last) + 1):
        in_path = base_dir / f"{args.base_name}_diff{diff}.json"
        out_path = base_dir / f"{args.base_name}_diff{diff}_lite.json"
        if not in_path.exists():
            print(f"[WARN] Missing input: {in_path}")
            continue
        try:
            regenerate_one(in_path, out_path)
            print(f"[OK] Wrote {out_path}")
            any_written = True
        except Exception as e:
            print(f"[ERROR] Failed {in_path} -> {out_path}: {e}")
    return 0 if any_written else 1


if __name__ == "__main__":
    raise SystemExit(main())

