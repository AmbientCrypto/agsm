#!/usr/bin/env bash
set -euo pipefail

# Live provider smoke tests for streaming. Opt-in via LIVE_TESTS=1.
# Usage examples:
#   bash tools/run_live_tests.sh            # run all live tests (ambient+openai+together)
#   bash tools/run_live_tests.sh ambient    # only Ambient
#   bash tools/run_live_tests.sh openai     # only OpenAI
#   bash tools/run_live_tests.sh together   # only Together

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="${SCRIPT_DIR%/tools}"
cd "$ROOT_DIR"

provider="${1:-all}"

# Prefer project venv if present
if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export LIVE_TESTS=1

# Load keys from local files if env not already set
if [ -z "${AMBIENT_API_KEY:-}" ]; then
  for f in "${ROOT_DIR}/ambient_api_key.txt" "$HOME/.ambient_api_key"; do
    if [ -f "$f" ]; then export AMBIENT_API_KEY="$(tr -d '\r\n' < "$f")"; break; fi
  done
fi
if [ -z "${OPENAI_API_KEY:-}" ] && [ -f "${ROOT_DIR}/openai_api_key.txt" ]; then
  export OPENAI_API_KEY="$(tr -d '\r\n' < "${ROOT_DIR}/openai_api_key.txt")"
fi
if [ -z "${TOGETHER_API_KEY:-}" ] && [ -f "${ROOT_DIR}/togetherai_api_key.txt" ]; then
  export TOGETHER_API_KEY="$(tr -d '\r\n' < "${ROOT_DIR}/togetherai_api_key.txt")"
fi

# Print key lengths without exposing values (safe with set -u)
_amb_show="${AMBIENT_API_KEY:-}"
_oai_show="${OPENAI_API_KEY:-}"
_tog_show="${TOGETHER_API_KEY:-}"
echo "LIVE_TESTS=$LIVE_TESTS  AMBIENT=${#_amb_show}  OPENAI=${#_oai_show}  TOGETHER=${#_tog_show}"

case "$provider" in
  ambient)
    target=(tests/test_live_streaming.py::test_live_ambient_streaming)
    ;;
  openai)
    target=(tests/test_live_streaming.py::test_live_openai_responses_streaming)
    ;;
  together)
    target=(tests/test_live_streaming.py::test_live_together_streaming)
    ;;
  all|*)
    # Run all live-marked tests; missing keys will cause individual tests to skip
    target=(-m live)
    ;;
esac

# Prefer pytest if available, otherwise use python -m pytest
if command -v pytest >/dev/null 2>&1; then
  PYTEST=pytest
else
  PYTEST="python -m pytest"
fi

set -x
$PYTEST "${target[@]}" -s -vv -ra
