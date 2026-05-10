"""
Complexity classifier — sends the user's prompt to DeepSeek (via Ollama)
to get a complexity score from 1-10.
"""

import json
import re
import requests
from config import OLLAMA_CHAT_ENDPOINT, CLASSIFIER_MODEL, ROUTER_SYSTEM_PROMPT_PATH


def load_prompt(filepath: str) -> str:
    """Read a system prompt from a text file and return it as a string."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def classify_task(user_prompt: str) -> int:
    """
    Send the user's prompt to DeepSeek acting as a complexity evaluator.
    Returns an integer score from 1-10.
    Defaults to 6 (routes to Claude) if anything goes wrong — fail safe.
    """
    # Load the router system prompt that tells DeepSeek how to score tasks
    system_prompt = load_prompt(ROUTER_SYSTEM_PROMPT_PATH)

    # Build the request payload for the Ollama chat API
    payload = {
        "model": CLASSIFIER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": True,
        "options": {
            "num_predict": 256,  # Small model doesn't need much room for a JSON response
        },
    }

    try:
        # Call the Ollama API with streaming
        response = requests.post(OLLAMA_CHAT_ENDPOINT, json=payload, timeout=120, stream=True)
        response.raise_for_status()

        # Read streamed chunks, filtering out DeepSeek R1's <think> reasoning
        raw_reply = ""
        inside_think = False  # Track whether we're inside <think>...</think> tags
        print("Router thinking...", end="", flush=True)
        for line in response.iter_lines():
            if line:
                chunk_data = json.loads(line)
                chunk_text = chunk_data.get("message", {}).get("content", "")
                raw_reply += chunk_text

                # Check if we entered or exited a <think> block
                if "<think>" in chunk_text:
                    inside_think = True
                if "</think>" in chunk_text:
                    inside_think = False
                    continue  # Skip the chunk containing </think>

                # Only print text that's outside <think> tags (the final answer)
                if not inside_think:
                    print(chunk_text, end="", flush=True)
        print()  # Newline after streaming finishes

        # Strip out everything between <think> and </think> to get the actual answer
        reply = re.sub(r"<think>.*?</think>", "", raw_reply, flags=re.DOTALL).strip()

        # Debug: show what we're working with
        print(f"(Debug) Raw length: {len(raw_reply)}, Filtered reply: '{reply[:200]}'")

        # Method 1: Try to find {"score": N} anywhere in the raw or filtered response
        # This handles cases where DeepSeek wraps the JSON in extra text
        score_match = re.search(r'\{\s*"score"\s*:\s*(\d+)\s*\}', raw_reply)
        if score_match:
            score = int(score_match.group(1))
            return max(1, min(10, score))

        # Method 2: Try to parse the filtered reply as pure JSON
        try:
            data = json.loads(reply)
            score = int(data["score"])
            return max(1, min(10, score))
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        # Method 3: Look for any standalone number 1-10 in the filtered reply
        numbers = re.findall(r"\b(\d{1,2})\b", reply)
        for num_str in numbers:
            num = int(num_str)
            if 1 <= num <= 10:
                return num

        # If we still can't find a score, default to 6 (sends to Claude — safe choice)
        print("Warning: Could not parse complexity score. Defaulting to 6 (routing to Claude).")
        return 6

    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to Ollama. Is it running at localhost:11434?")
        print("Defaulting to Claude Code for safety.")
        return 6
    except requests.exceptions.Timeout:
        print("Error: Ollama request timed out.")
        print("Defaulting to Claude Code for safety.")
        return 6
    except Exception as e:
        print(f"Error during classification: {e}")
        print("Defaulting to Claude Code for safety.")
        return 6
