"""
Testing agent — sends DeepSeek's generated code to a second DeepSeek call
acting as a code reviewer. Returns a pass/fail verdict with explanation.
Also includes technique tagging for the example bank.
"""

import json
import re
import requests
from config import (
    OLLAMA_CHAT_ENDPOINT,
    REVIEWER_MODEL,
    REVIEWER_SYSTEM_PROMPT_PATH,
    TAGGER_MODEL,
    TAGGER_SYSTEM_PROMPT_PATH,
)
from classifier import load_prompt


def review_code(task_prompt: str, generated_code: str) -> dict:
    """
    Send the original task and DeepSeek's generated code to a reviewer model.
    Returns a dict with 'verdict' ('pass' or 'fail'), 'issues' (list), and 'explanation'.

    Args:
        task_prompt: The original coding task the user asked for
        generated_code: The code that DeepSeek generated

    Returns:
        Dict like {"verdict": "pass", "issues": [], "explanation": "..."}
        Defaults to pass if the reviewer itself fails (don't block the user).
    """
    print("Running code review...")

    # Load the reviewer system prompt
    system_prompt = load_prompt(REVIEWER_SYSTEM_PROMPT_PATH)

    # Build the user message with both the task and the generated code
    review_message = (
        f"TASK:\n{task_prompt}\n\n"
        f"GENERATED CODE:\n{generated_code}"
    )

    # Build the request payload for the Ollama chat API
    payload = {
        "model": REVIEWER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": review_message},
        ],
        "stream": True,
        "options": {
            "num_predict": 1024,  # Reviewer only needs enough for JSON + thinking
        },
    }

    try:
        # Call the Ollama API with streaming
        response = requests.post(OLLAMA_CHAT_ENDPOINT, json=payload, timeout=120, stream=True)
        response.raise_for_status()

        # Read streamed chunks, filtering out <think> reasoning
        raw_reply = ""
        inside_think = False
        print("Reviewer thinking...", end="", flush=True)
        for line in response.iter_lines():
            if line:
                chunk_data = json.loads(line)
                chunk_text = chunk_data.get("message", {}).get("content", "")
                raw_reply += chunk_text

                # Track <think> blocks
                if "<think>" in chunk_text:
                    inside_think = True
                if "</think>" in chunk_text:
                    inside_think = False
                    continue
        print(" done.")

        # Strip out thinking tags to get the actual answer
        reply = re.sub(r"<think>.*?</think>", "", raw_reply, flags=re.DOTALL).strip()

        # Try to parse as JSON
        try:
            result = json.loads(reply)
            # Make sure it has the expected fields
            verdict = result.get("verdict", "pass").lower()
            issues = result.get("issues", [])
            explanation = result.get("explanation", "")
            return {"verdict": verdict, "issues": issues, "explanation": explanation}
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fallback: look for "pass" or "fail" in the text
        if "fail" in reply.lower():
            return {"verdict": "fail", "issues": ["Could not parse detailed issues"], "explanation": reply[:200]}

        # Default to pass if we can't parse anything (don't block the user)
        print("Warning: Could not parse review result. Defaulting to pass.")
        return {"verdict": "pass", "issues": [], "explanation": "Review parsing failed."}

    except requests.exceptions.ConnectionError:
        print("Warning: Cannot connect to Ollama for review. Skipping review.")
        return {"verdict": "pass", "issues": [], "explanation": "Reviewer unavailable."}
    except requests.exceptions.Timeout:
        print("Warning: Review timed out. Skipping review.")
        return {"verdict": "pass", "issues": [], "explanation": "Review timed out."}
    except Exception as e:
        print(f"Warning: Review error ({e}). Skipping review.")
        return {"verdict": "pass", "issues": [], "explanation": f"Error: {e}"}


def tag_techniques(generated_code: str) -> list[str]:
    """
    Send generated code to a small local model to identify coding techniques used.
    Returns a list of technique tags like ["for_loop", "input_validation", "early_return"].

    These tags are stored in the example bank so we can find relevant examples
    across different projects based on coding patterns, not just prompt keywords.

    Args:
        generated_code: The code to analyze for techniques.

    Returns:
        List of technique tag strings. Returns empty list if tagging fails.
    """
    print("Tagging techniques...", end="", flush=True)

    # Load the tagger system prompt
    system_prompt = load_prompt(TAGGER_SYSTEM_PROMPT_PATH)

    # Build the request payload
    payload = {
        "model": TAGGER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"CODE:\n{generated_code}"},
        ],
        "stream": True,
        "options": {
            "num_predict": 256,  # Tags are short, don't need much output
        },
    }

    try:
        response = requests.post(OLLAMA_CHAT_ENDPOINT, json=payload, timeout=60, stream=True)
        response.raise_for_status()

        # Read streamed chunks, filtering out <think> reasoning
        raw_reply = ""
        inside_think = False
        for line in response.iter_lines():
            if line:
                chunk_data = json.loads(line)
                chunk_text = chunk_data.get("message", {}).get("content", "")
                raw_reply += chunk_text

                if "<think>" in chunk_text:
                    inside_think = True
                if "</think>" in chunk_text:
                    inside_think = False
                    continue
        print(" done.")

        # Strip out thinking tags to get the actual answer
        reply = re.sub(r"<think>.*?</think>", "", raw_reply, flags=re.DOTALL).strip()

        # Parse the JSON array of tags
        tags = json.loads(reply)
        if isinstance(tags, list):
            print(f"Techniques found: {', '.join(tags)}")
            return tags

        print("Warning: Tagger returned unexpected format. Skipping tags.")
        return []

    except (json.JSONDecodeError, AttributeError):
        print(" could not parse tags. Skipping.")
        return []
    except requests.exceptions.ConnectionError:
        print(" cannot connect to Ollama. Skipping tags.")
        return []
    except requests.exceptions.Timeout:
        print(" tagger timed out. Skipping tags.")
        return []
    except Exception as e:
        print(f" tagger error ({e}). Skipping tags.")
        return []
