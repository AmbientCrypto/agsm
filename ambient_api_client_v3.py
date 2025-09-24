#!/usr/bin/env python3
"""
Ambient API Client V3
~~~~~~~~~~~~~~~~~~~

Simplified Ambient API client for V3 system.
No threading, no async, just sequential calls with simple retry logic.
"""

import os
import sys
import time
import json
import requests
from pathlib import Path
from typing import Dict, Any, Generator


class AmbientAPIClientV3:
    """
    Simple Ambient API client with basic retry logic and streaming support.
    
    This is a simplified version of the original ambient_api.py that removes
    all async/threading complexity while maintaining the same functionality.
    """
    
    def __init__(self, api_key: str = None, model: str = "deepseek-ai/DeepSeek-R1",
                 base_url: str = "https://api.ambient.xyz/v1", stream_output: bool = False,
                 timeout: int = 580):
        """
        Initialize Ambient API client.
        
        Args:
            api_key: API key (defaults to file or hardcoded)
            model: Model to use
            base_url: API base URL
            stream_output: Whether to stream output to console (default: False)
            timeout: Request timeout in seconds (default: 180 = 3 minutes)
        """
        self.api_key = api_key or self._load_api_key()
        self.model = model
        self.base_url = base_url
        self.stream_output = stream_output
        
        # Simple retry configuration
        self.backoff_sequence = [5, 10, 20]  # seconds
        
        # Request timeout
        self.timeout = timeout  # Default 180 seconds (3 minutes) for large prompts
        
        # Last usage metadata if provided by API
        self.last_usage = None
        # Last raw JSON response and reasoning content if provided by API
        self.last_response_json = None
        self.last_reasoning_content = None
        # Last error details (text/type) for external logging/inspection
        self.last_error_text = None
        self.last_error_type = None
        
        print("🚀 AmbientAPIClientV3 initialized")
        print(f"   Model: {model}")
        print(f"   Base URL: {base_url}")
        if stream_output:
            print("   Streaming: Enabled (output will be displayed as generated)")
    
    def _load_api_key(self) -> str:
        """Load API key from env or file; no hardcoded defaults."""
        # Prefer environment variable
        env_key = os.getenv("AMBIENT_API_KEY")
        if env_key and env_key.strip():
            return env_key.strip()

        # Optional explicit file path via env
        env_file = os.getenv("AMBIENT_API_KEY_FILE")
        if env_file:
            p = Path(env_file).expanduser()
            if p.exists():
                key = p.read_text(encoding="utf-8").strip()
                if key:
                    return key

        # Common fallback locations (outside repo preferred)
        candidates = [
            Path.cwd() / "ambient_api_key.txt",
            Path(__file__).resolve().parent.parent.parent / "ambient_api_key.txt",
            Path.home() / ".ambient_api_key",
        ]
        for p in candidates:
            try:
                if p.exists():
                    key = p.read_text(encoding="utf-8").strip()
                    if key:
                        return key
            except Exception:
                continue

        raise RuntimeError(
            "AMBIENT_API_KEY not found. Set env var or provide an ambient_api_key.txt (outside repo)."
        )
    
    def complete(self, prompt: str, max_tokens: int = 34000, temperature: float = 0.7,
                 stream: bool = False) -> str:
        """
        Get completion from Ambient API with simple retry logic.
        
        Args:
            prompt: The prompt to send
            max_tokens: Maximum tokens in response
            temperature: Temperature for generation
            stream: Whether to stream the response
            
        Returns:
            The completion text
        """
        if stream:
            # For streaming, collect all chunks
            chunks = []
            for chunk in self.stream_complete(prompt, max_tokens, temperature):
                chunks.append(chunk)
            
            # Validate we got content
            result = "".join(chunks)
            if not result or not result.strip():
                raise ValueError("API returned empty response in streaming mode")
                
            return result
        else:
            return self._complete_non_streaming(prompt, max_tokens, temperature)
    
    def _complete_non_streaming(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Non-streaming completion request."""
        attempt = 0
        max_attempts = 60  # Match the retry count from article_worker
        
        while attempt < max_attempts:
            try:
                # Prepare request
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                # Use OpenAI-compatible format
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False
                }
                
                # Make request
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                # Check response
                if response.status_code == 200:
                    data = response.json()
                    # Debug prints removed; use caller-side logging if desired
                    # Save raw response and reasoning_content (if present)
                    try:
                        self.last_response_json = data
                        self.last_reasoning_content = None
                        ch0 = (data.get("choices") or [{}])[0]
                        msg = ch0.get("message") or {}
                        rc = msg.get("reasoning_content")
                        if isinstance(rc, str) and rc.strip():
                            self.last_reasoning_content = rc
                    except Exception:
                        self.last_response_json = None
                        self.last_reasoning_content = None
                    # Try capture usage if present
                    try:
                        self.last_usage = data.get("usage")
                    except Exception:
                        self.last_usage = None
                    if "choices" in data and len(data["choices"]) > 0:
                        content = data["choices"][0]["message"]["content"]
                        # Validate content is not empty
                        if not content or not content.strip():
                            print("Content is: ")
                            print(content)
                            raise ValueError("API returned empty content in response")
                        return content
                    else:
                        raise ValueError(f"Invalid response format: {data}")
                else:
                    raise ValueError(f"API error {response.status_code}: {response.text}")
                    
            except Exception as e:
                # Get backoff time
                backoff = self.backoff_sequence[attempt % len(self.backoff_sequence)]
                
                # Log error
                self.last_error_text = str(e)
                self.last_error_type = type(e).__name__
                print(f"[Ambient API] Error: {self.last_error_text}")
                # Emit a copy-paste cURL (here-doc form) for reproduction
                try:
                    body = json.dumps(payload, ensure_ascii=False, indent=2)
                    url = f"{self.base_url.rstrip('/')}/chat/completions"
                    curl_hd = (
                        f"curl -sS -X POST \"{url}\" \\\n+  -H \"Authorization: Bearer $AMBIENT_API_KEY\" \\\n+  -H \"Content-Type: application/json\" \\\n+  --data-binary @- <<'JSON'\n{body}\nJSON"
                    )
                    print("[Ambient API] cURL (env key; here-doc):")
                    print(curl_hd)
                    # Placeholder variant
                    curl_hd_ph = curl_hd.replace("$AMBIENT_API_KEY", "YOUR_API_KEY")
                    print("[Ambient API] cURL (replace YOUR_API_KEY if needed):")
                    print(curl_hd_ph)
                except Exception:
                    pass
                sys.stdout.flush()
                print(f"[Ambient API] Retrying in {backoff} seconds...")
                
                # Sleep and retry
                time.sleep(backoff)
                attempt += 1
        
        # If we've exhausted all attempts, raise an error
        # Preserve last error context in final raised message
        last = f" Last error: {self.last_error_type}: {self.last_error_text}" if self.last_error_text else ""
        raise RuntimeError(f"Failed to get completion after {max_attempts} attempts.{last}")
    
    def stream_complete(self, prompt: str, max_tokens: int = 34000, 
                       temperature: float = 0.7) -> Generator[str, None, None]:
        """
        Stream completion from Ambient API.
        
        Args:
            prompt: The prompt to send
            max_tokens: Maximum tokens in response
            temperature: Temperature for generation
            
        Yields:
            Chunks of completion text
        """
        attempt = 0
        max_attempts = 60  # Match the retry count from article_worker
        
        while attempt < max_attempts:
            try:
                # Prepare request
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                }
                
                # Use OpenAI-compatible format with streaming
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": True
                }
                
                # Make streaming request
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                    stream=True
                )
                
                # Check response
                if response.status_code != 200:
                    raise ValueError(f"API error {response.status_code}: {response.text}")
                
                # Process streaming response
                # Reset usage/last response for streaming (usually not provided)
                self.last_usage = None
                self.last_response_json = None
                self.last_reasoning_content = None
                emitted_any = False
                for line in response.iter_lines():
                    if not line:
                        continue
                    
                    # Decode line
                    line_text = line.decode('utf-8')
                    
                    # Skip empty lines
                    if not line_text.strip():
                        continue
                    
                    # Handle SSE format
                    if line_text.startswith("data: "):
                        data_str = line_text[6:]  # Remove "data: " prefix
                        
                        # Check for end of stream
                        if data_str.strip() == "[DONE]":
                            return
                        
                        try:
                            # Parse JSON
                            data = json.loads(data_str)
                            
                            # Extract content
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                # Some models emit reasoning tokens under a different field; prefer content when available
                                content = delta.get("content", "") or delta.get("text", "")
                                if content:
                                    # Print to console if stream_output is enabled
                                    if self.stream_output:
                                        print(content, end="", flush=True)
                                    emitted_any = True
                                    yield content
                                    
                        except json.JSONDecodeError:
                            # Skip invalid JSON
                            continue

                # Stream completed successfully
                if self.stream_output:
                    print()  # Add newline after streaming
                # Fallback: if no content was emitted during streaming, try non-streaming once
                if not emitted_any:
                    try:
                        content = self._complete_non_streaming(prompt, max_tokens, temperature)
                        if content and content.strip():
                            yield content
                            return
                    except Exception:
                        pass
                return
                    
            except Exception as e:
                # Get backoff time
                backoff = self.backoff_sequence[attempt % len(self.backoff_sequence)]
                
                # Log error
                self.last_error_text = str(e)
                self.last_error_type = type(e).__name__
                print(f"[Ambient API] Stream error: {self.last_error_text}")
                # Emit a copy-paste cURL for reproduction (stream=false fallback, here-doc form)
                try:
                    payload = {
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "stream": False,
                    }
                    body = json.dumps(payload, ensure_ascii=False, indent=2)
                    url = f"{self.base_url.rstrip('/')}/chat/completions"
                    curl_hd = (
                        f"curl -sS -X POST \"{url}\" \\\n+  -H \"Authorization: Bearer $AMBIENT_API_KEY\" \\\n+  -H \"Content-Type: application/json\" \\\n+  --data-binary @- <<'JSON'\n{body}\nJSON"
                    )
                    print("[Ambient API] cURL (env key; here-doc):")
                    print(curl_hd)
                    curl_hd_ph = curl_hd.replace("$AMBIENT_API_KEY", "YOUR_API_KEY")
                    print("[Ambient API] cURL (replace YOUR_API_KEY if needed):")
                    print(curl_hd_ph)
                except Exception:
                    pass
                sys.stdout.flush()
                print(f"[Ambient API] Retrying in {backoff} seconds...")
                
                # Sleep and retry
                time.sleep(backoff)
                attempt += 1
        
        # If we've exhausted all attempts, raise an error
        raise RuntimeError(f"Failed to get streaming completion after {max_attempts} attempts")
    
    def execute_single_query(self, query: str, max_tokens: int = 34000,
                           temperature: float = 0.7) -> Dict[str, Any]:
        """
        Execute a single query and return structured result.
        Compatible with original ambient_api interface.
        
        Args:
            query: The query to execute
            max_tokens: Maximum tokens in response
            temperature: Temperature for generation
            
        Returns:
            Dictionary with query result
        """
        start_time = time.time()
        
        try:
            # Get completion with streaming
            print("\n📝 Processing query...")
            result_text = ""
            
            for chunk in self.stream_complete(query, max_tokens, temperature):
                print(chunk, end="", flush=True)
                result_text += chunk
            
            print("\n")  # New line after streaming
            
            # Calculate stats
            duration = time.time() - start_time
            
            return {
                "success": True,
                "result": result_text,
                "duration": duration,
                "error": None
            }
            
        except Exception as e:
            # This should never happen due to infinite retry
            # but included for completeness
            duration = time.time() - start_time
            return {
                "success": False,
                "result": None,
                "duration": duration,
                "error": str(e)
            }


# Convenience function for compatibility
def create_ambient_client(api_key: str = None) -> AmbientAPIClientV3:
    """Create an Ambient API client with default settings."""
    return AmbientAPIClientV3(api_key=api_key)
