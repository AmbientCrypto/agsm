#!/usr/bin/env python3
"""
Standalone manual tests for Ambient API Client V3.

Run directly, not via pytest:
  AMBIENT_API_KEY=... python tools/ambient_client_standalone.py

This file was renamed to avoid pytest collection and accidental network calls
during CI. The logic is unchanged from the prior *_test.py variant.
"""

import sys
import time
from pathlib import Path

from ambient_api_client_v3 import AmbientAPIClientV3


def test_basic_completion():
    print("=" * 60)
    print("TEST 1: Basic Non-Streaming Completion")
    print("=" * 60)
    try:
        client = AmbientAPIClientV3(stream_output=False)
        prompt = "What is 2 + 2? Please respond with just the number."
        print(f"\nPrompt: {prompt}")
        print("\nGetting response...")
        start = time.time()
        response = client.complete(prompt, max_tokens=100, temperature=0.1)
        duration = time.time() - start
        print(f"\nResponse: {response}")
        print(f"Duration: {duration:.2f} seconds")
        print("\n✅ Non-streaming test PASSED")
        return True
    except Exception as e:
        print(f"\n❌ Non-streaming test FAILED: {e}")
        return False


def test_streaming_completion():
    print("\n" + "=" * 60)
    print("TEST 2: Streaming Completion")
    print("=" * 60)
    try:
        client = AmbientAPIClientV3(stream_output=True)
        prompt = "Count from 1 to 5, one number per line."
        print(f"\nPrompt: {prompt}")
        print("\nStreaming response:")
        print("-" * 30)
        start = time.time()
        response = client.complete(prompt, max_tokens=100, temperature=0.1, stream=True)
        duration = time.time() - start
        print("-" * 30)
        print(f"\nFull response collected: {response}")
        print(f"Duration: {duration:.2f} seconds")
        print("\n✅ Streaming test PASSED")
        return True
    except Exception as e:
        print(f"\n❌ Streaming test FAILED: {e}")
        return False


def test_execute_single_query():
    print("\n" + "=" * 60)
    print("TEST 3: Execute Single Query (Original Interface)")
    print("=" * 60)
    try:
        client = AmbientAPIClientV3(stream_output=False)
        query = "What is the capital of France? Answer in one word."
        print(f"\nQuery: {query}")
        result = client.execute_single_query(query, max_tokens=50, temperature=0.1)
        print(f"\nResult success: {result['success']}")
        print(f"Response: {result['result']}")
        print(f"Duration: {result['duration']:.2f} seconds")
        print(f"Error: {result['error']}")
        if result['success']:
            print("\n✅ Execute single query test PASSED")
            return True
        else:
            print(f"\n❌ Execute single query test FAILED: {result['error']}")
            return False
    except Exception as e:
        print(f"\n❌ Execute single query test FAILED: {e}")
        return False


def main():
    print("\n" + "🧪" * 30)
    print("AMBIENT API CLIENT V3 - STANDALONE TEST SUITE")
    print("🧪" * 30)
    key_file = Path.cwd() / "ambient_api_key.txt"
    if key_file.exists():
        print(f"\n✅ Using API key from: {key_file}")
    else:
        print(f"\nℹ️ Provide AMBIENT_API_KEY or a key file for real calls.")
    results = []
    print("\nStarting tests...\n")
    results.append(("Basic Completion", test_basic_completion()))
    results.append(("Streaming Completion", test_streaming_completion()))
    results.append(("Execute Single Query", test_execute_single_query()))
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    failed = sum(1 for _, r in results if r is False)
    if failed == 0:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

