"""
DeepSeek code generation agent — sends the user's coding task to DeepSeek
via the Ollama API and returns the generated code/response.
"""

import json
import re
import time
import requests
from config import OLLAMA_CHAT_ENDPOINT, CODER_MODEL, CODER_SYSTEM_PROMPT_PATH
from classifier import load_prompt
from example_bank import get_similar_examples


def _build_prompt_with_examples(user_prompt: str) -> str:
    """
    Search the example bank for similar past tasks and build an enriched prompt.
    If similar examples are found, they're added as reference code so DeepSeek
    can learn from what worked before. If none are found, returns the original prompt.
    """
    examples = get_similar_examples(user_prompt, limit=2)

    if not examples:
        return user_prompt

    # Build a section showing past successful examples
    examples_text = "\n\nHere are similar tasks you handled successfully before — use them as reference:\n"
    for i, ex in enumerate(examples, 1):
        examples_text += f"\n--- Example {i} ---\n"
        examples_text += f"Task: {ex['user_prompt']}\n"
        examples_text += f"Solution:\n{ex['generated_code']}\n"

    print(f"Found {len(examples)} similar example(s) in the bank.")
    return user_prompt + examples_text


def generate_with_deepseek(user_prompt: str, conversation_history: list = None) -> str:
    """
    Send the user's coding task to DeepSeek and return its response.
    Returns the generated code/text, or an error message if something fails.

    Args:
        user_prompt: The coding task or follow-up feedback
        conversation_history: Optional list of prior messages for multi-turn conversations.
                             If provided, user_prompt is appended and the full history is sent.
                             If None, a fresh conversation is started with the system prompt.
    """
    if conversation_history is not None:
        print("DeepSeek is revising code...")
    else:
        print("DeepSeek is generating code...")

    # Build the messages list — either fresh or continuing a conversation
    if conversation_history is not None:
        # Add the new user message to the existing conversation
        conversation_history.append({"role": "user", "content": user_prompt})
        messages = conversation_history
    else:
        # First message: load the system prompt and start a new conversation
        system_prompt = load_prompt(CODER_SYSTEM_PROMPT_PATH)

        # Search the example bank for similar past tasks to use as few-shot context
        enriched_prompt = _build_prompt_with_examples(user_prompt)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": enriched_prompt},
        ]

    # Build the request payload for the Ollama chat API
    payload = {
        "model": CODER_MODEL,
        "messages": messages,
        "stream": True,
        "options": {
            "num_predict": 1024,  # Cap output length to keep simple tasks fast
        },
    }

    try:
        # Call the Ollama API with streaming (15 min timeout — DeepSeek R1 reasoning can be slow)
        start_time = time.time()
        response = requests.post(OLLAMA_CHAT_ENDPOINT, json=payload, timeout=900, stream=True)
        response.raise_for_status()

        # Read streamed chunks, filtering out DeepSeek R1's <think> reasoning
        raw_reply = ""
        inside_think = False  # Track whether we're inside <think>...</think> tags
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

                # Only print text that's outside <think> tags (the actual code)
                if not inside_think:
                    print(chunk_text, end="", flush=True)
        print()  # Newline after streaming finishes

        # Strip out thinking tags to get the clean answer
        reply = re.sub(r"<think>.*?</think>", "", raw_reply, flags=re.DOTALL).strip()

        # Save the assistant's reply to conversation history for follow-ups
        if conversation_history is not None:
            conversation_history.append({"role": "assistant", "content": reply})

        elapsed = time.time() - start_time
        print(f"\nDeepSeek responded in {elapsed:.1f} seconds.")
        return reply

    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to Ollama. Is it running at localhost:11434?"
    except requests.exceptions.Timeout:
        return "Error: DeepSeek request timed out. The task might be too complex for local generation."
    except Exception as e:
        return f"Error during DeepSeek generation: {e}"
