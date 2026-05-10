"""
OpenSource Router — Web UI entry point.

Replaces the CLI (router.py) with a browser-based chat interface.
Runs a Flask server on localhost that routes coding tasks to DeepSeek or Claude
based on complexity scoring. Uses Server-Sent Events (SSE) for real-time streaming.

Usage:
    python router_app.py
    Then open http://localhost:5000 in your browser.
"""

import json
import os
import re
import time
import uuid
import threading
import requests as http_requests
from flask import Flask, render_template, request, jsonify, Response, stream_with_context

from config import (
    COMPLEXITY_THRESHOLD,
    LAST_SESSION_PATH,
    OLLAMA_CHAT_ENDPOINT,
    CODER_MODEL,
    CODER_SYSTEM_PROMPT_PATH,
    WEB_PORT,
)
from classifier import classify_task, load_prompt
from claude_agent import generate_with_claude
from testing_agent import review_code, tag_techniques
from example_bank import init_db, save_example, get_similar_examples, get_all_examples

# --- Flask app setup ---
app = Flask(__name__)

# --- Session state ---
# Tracks the current conversation so follow-ups go to the right model.
# In a single-user local app like this, a simple dict works fine.
session_state = {
    "session_id": None,       # Claude session UUID
    "using_claude": False,    # True if current conversation is with Claude
    "project_dir": None,      # Path to the project Claude should work in
    "deepseek_history": [],   # Conversation history for DeepSeek multi-turn
    "original_prompt": None,  # First prompt in this conversation (for escalation)
}


def save_session(session_id: str, project_dir: str) -> None:
    """Save the current Claude session ID and project dir for --resume support."""
    try:
        with open(LAST_SESSION_PATH, "w") as f:
            f.write(f"{session_id}\n{project_dir}")
    except Exception as e:
        print(f"Warning: Could not save session: {e}")


def load_session() -> tuple[str, str] | None:
    """Load the last saved Claude session ID and project dir."""
    try:
        with open(LAST_SESSION_PATH, "r") as f:
            lines = f.read().strip().split("\n")
            if len(lines) >= 2:
                return lines[0], lines[1]
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Warning: Could not load session: {e}")
    return None


def reset_session() -> None:
    """Clear the session state for a new conversation."""
    session_state["session_id"] = None
    session_state["using_claude"] = False
    session_state["deepseek_history"] = []
    session_state["original_prompt"] = None


# --- SSE streaming helpers ---

def sse_event(event: str, data: str) -> str:
    """Format a Server-Sent Event message.

    SSE is how the server pushes real-time updates to the browser.
    Each event has a type (like 'token', 'status', 'done') and data payload.
    """
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def stream_deepseek_generation(user_prompt: str, conversation_history: list = None):
    """
    Generator that streams DeepSeek code generation token-by-token via SSE.

    This is similar to generate_with_deepseek() in deepseek_agent.py, but instead
    of printing to terminal, it yields SSE events that the browser receives in real-time.
    """
    # Build the messages list — either fresh or continuing a conversation
    if conversation_history is not None and len(conversation_history) > 0:
        yield sse_event("status", "DeepSeek is revising code...")
        conversation_history.append({"role": "user", "content": user_prompt})
        messages = conversation_history
    else:
        yield sse_event("status", "DeepSeek is generating code...")
        system_prompt = load_prompt(CODER_SYSTEM_PROMPT_PATH)

        # Search example bank for similar past tasks (few-shot injection)
        examples = get_similar_examples(user_prompt, limit=2)
        enriched_prompt = user_prompt
        if examples:
            examples_text = "\n\nHere are similar tasks you handled successfully before — use them as reference:\n"
            for i, ex in enumerate(examples, 1):
                examples_text += f"\n--- Example {i} ---\n"
                examples_text += f"Task: {ex['user_prompt']}\n"
                examples_text += f"Solution:\n{ex['generated_code']}\n"
            enriched_prompt = user_prompt + examples_text
            yield sse_event("status", f"Found {len(examples)} similar example(s) in the bank.")

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
            "num_predict": 1024,
        },
    }

    try:
        start_time = time.time()
        response = http_requests.post(OLLAMA_CHAT_ENDPOINT, json=payload, timeout=900, stream=True)
        response.raise_for_status()

        # Read streamed chunks, filtering out <think> reasoning tags
        raw_reply = ""
        inside_think = False
        thinking_started = False

        for line in response.iter_lines():
            if line:
                chunk_data = json.loads(line)
                chunk_text = chunk_data.get("message", {}).get("content", "")
                raw_reply += chunk_text

                # Check if we entered or exited a <think> block
                if "<think>" in chunk_text:
                    inside_think = True
                    thinking_started = True
                    yield sse_event("status", "DeepSeek is thinking...")
                if "</think>" in chunk_text:
                    inside_think = False
                    yield sse_event("status", "DeepSeek is writing code...")
                    continue

                # Only send tokens that are outside <think> tags (the actual code)
                if not inside_think:
                    yield sse_event("token", chunk_text)

        # Clean up the reply — strip out thinking tags
        reply = re.sub(r"<think>.*?</think>", "", raw_reply, flags=re.DOTALL).strip()

        # Save to conversation history for follow-ups
        if conversation_history is not None:
            conversation_history.append({"role": "assistant", "content": reply})

        elapsed = time.time() - start_time
        yield sse_event("status", f"DeepSeek responded in {elapsed:.1f} seconds.")
        yield sse_event("done", reply)

    except http_requests.exceptions.ConnectionError:
        yield sse_event("error", "Cannot connect to Ollama. Is it running at localhost:11434?")
    except http_requests.exceptions.Timeout:
        yield sse_event("error", "DeepSeek request timed out. The task might be too complex.")
    except Exception as e:
        yield sse_event("error", f"DeepSeek generation error: {e}")


# --- Flask routes ---

@app.route("/")
def index():
    """Serve the main chat page."""
    return render_template("index.html")


@app.route("/send", methods=["POST"])
def send_message():
    """
    Main route — receives a prompt from the chat UI, classifies it, and streams
    the response back via SSE. Handles the full router flow:
    classify → route to DeepSeek or Claude → review (if DeepSeek) → respond.
    """
    data = request.json
    prompt = data.get("prompt", "").strip()
    project_dir = data.get("project_dir", "").strip() or None

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    # Update project dir if provided
    if project_dir:
        session_state["project_dir"] = project_dir

    def generate():
        """Generator function that yields SSE events as the router processes the task."""

        # Check if this is a follow-up in an existing conversation
        is_followup = session_state["session_id"] is not None

        if is_followup:
            # Follow-up message — send to whichever model is active
            if session_state["using_claude"]:
                yield sse_event("status", "Sending follow-up to Claude Code...")
                yield sse_event("model", "claude")
                try:
                    result = generate_with_claude(
                        prompt,
                        session_id=session_state["session_id"],
                        is_followup=True,
                        project_dir=session_state["project_dir"],
                    )
                    yield sse_event("done", result)
                except Exception as e:
                    yield sse_event("error", f"Claude error: {e}")
            else:
                # Follow-up to DeepSeek — stream it
                yield sse_event("model", "deepseek")
                yield from stream_deepseek_generation(
                    prompt,
                    conversation_history=session_state["deepseek_history"],
                )
            return

        # --- New conversation: classify and route ---
        session_state["original_prompt"] = prompt

        yield sse_event("status", "Analyzing task complexity...")
        score = classify_task(prompt)
        yield sse_event("score", str(score))

        # Generate a session ID for Claude conversation continuity
        session_state["session_id"] = str(uuid.uuid4())

        if score > COMPLEXITY_THRESHOLD:
            # --- Route to Claude ---
            session_state["using_claude"] = True
            yield sse_event("model", "claude")
            yield sse_event("status", f"Complexity: {score}/10 — Routing to Claude Code")

            proj = session_state["project_dir"]
            try:
                result = generate_with_claude(
                    prompt,
                    session_id=session_state["session_id"],
                    is_followup=False,
                    project_dir=proj,
                )
                # Tag techniques and save to example bank
                tags = tag_techniques(result)
                save_example(prompt, score, "claude", result, None, None, False, tags)
                save_session(session_state["session_id"], proj or "")
                yield sse_event("done", result)
            except Exception as e:
                yield sse_event("error", f"Claude error: {e}")

        else:
            # --- Route to DeepSeek ---
            session_state["using_claude"] = False
            yield sse_event("model", "deepseek")
            yield sse_event("status", f"Complexity: {score}/10 — Routing to DeepSeek (local)")

            # Stream DeepSeek generation — collect the final result from the 'done' event
            final_result = None
            for event in stream_deepseek_generation(prompt):
                yield event
                # Capture the final result from the done event
                if event.startswith("event: done"):
                    # Parse the data from the SSE event
                    data_line = event.split("data: ", 1)[1].split("\n")[0]
                    final_result = json.loads(data_line)

            if final_result is None:
                return  # Error already sent via SSE

            # Run the testing agent to review DeepSeek's output
            yield sse_event("status", "Running code review...")
            review = review_code(prompt, final_result)

            if review["verdict"] == "pass":
                yield sse_event("review", "passed")
                tags = tag_techniques(final_result)
                save_example(prompt, score, "deepseek", final_result, review["verdict"], review["issues"], False, tags)
            else:
                # Failed review — retry once
                yield sse_event("review", f"failed: {', '.join(review['issues'])}")
                yield sse_event("status", "Retrying with feedback...")

                retry_prompt = (
                    f"Your previous code had these issues: {', '.join(review['issues'])}. "
                    f"Please fix them. Original task: {prompt}"
                )

                # Stream the retry
                retry_result = None
                for event in stream_deepseek_generation(retry_prompt):
                    yield event
                    if event.startswith("event: done"):
                        data_line = event.split("data: ", 1)[1].split("\n")[0]
                        retry_result = json.loads(data_line)

                if retry_result is None:
                    return

                # Review the retry
                yield sse_event("status", "Reviewing retry...")
                review2 = review_code(prompt, retry_result)

                if review2["verdict"] == "pass":
                    yield sse_event("review", "passed on retry")
                    tags = tag_techniques(retry_result)
                    save_example(prompt, score, "deepseek", retry_result, review2["verdict"], review2["issues"], False, tags)
                else:
                    # Failed twice — escalate to Claude
                    yield sse_event("review", f"failed again: {', '.join(review2['issues'])}")
                    yield sse_event("status", "Escalating to Claude Code...")
                    yield sse_event("model", "claude")
                    session_state["using_claude"] = True

                    escalation_prompt = (
                        f"Original task: {prompt}\n\n"
                        f"A local model attempted this twice but failed code review. "
                        f"Issues found: {', '.join(review2['issues'])}. "
                        f"Please generate a correct solution."
                    )
                    try:
                        result = generate_with_claude(
                            escalation_prompt,
                            session_id=session_state["session_id"],
                            is_followup=False,
                            project_dir=session_state["project_dir"],
                        )
                        tags = tag_techniques(result)
                        save_example(prompt, score, "claude", result, review2["verdict"], review2["issues"], True, tags)
                        save_session(session_state["session_id"], session_state["project_dir"] or "")
                        yield sse_event("done", result)
                    except Exception as e:
                        yield sse_event("error", f"Claude error: {e}")

            # Initialize DeepSeek conversation history for follow-ups (if still on DeepSeek)
            if not session_state["using_claude"]:
                system_prompt = load_prompt(CODER_SYSTEM_PROMPT_PATH)
                session_state["deepseek_history"] = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": final_result},
                ]

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Prevents nginx from buffering SSE
        },
    )


@app.route("/escalate", methods=["POST"])
def escalate():
    """Manually escalate from DeepSeek to Claude mid-conversation."""
    if session_state["using_claude"]:
        return jsonify({"error": "Already using Claude"}), 400

    session_state["using_claude"] = True
    session_state["session_id"] = str(uuid.uuid4())

    original = session_state.get("original_prompt", "the current task")
    escalation_prompt = (
        f"Original task: {original}\n\n"
        f"A local model attempted this but the result needs improvement. "
        f"Please generate a better solution."
    )

    try:
        result = generate_with_claude(
            escalation_prompt,
            session_id=session_state["session_id"],
            is_followup=False,
            project_dir=session_state["project_dir"],
        )
        save_session(session_state["session_id"], session_state["project_dir"] or "")
        return jsonify({"result": result, "model": "claude"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/new", methods=["POST"])
def new_conversation():
    """Start a fresh conversation (clears session state)."""
    reset_session()
    return jsonify({"status": "ok"})


@app.route("/history", methods=["GET"])
def history():
    """Return recent entries from the example bank."""
    limit = request.args.get("limit", 50, type=int)
    examples = get_all_examples(limit=limit)
    return jsonify(examples)


@app.route("/settings", methods=["GET"])
def get_settings():
    """Return current settings so the UI can display them."""
    return jsonify({
        "project_dir": session_state["project_dir"] or "",
        "threshold": COMPLEXITY_THRESHOLD,
        "using_claude": session_state["using_claude"],
        "has_session": session_state["session_id"] is not None,
    })


@app.route("/settings", methods=["POST"])
def update_settings():
    """Update project directory from the UI."""
    data = request.json
    if "project_dir" in data:
        session_state["project_dir"] = data["project_dir"] or None
    return jsonify({"status": "ok"})


# --- Start the server ---

if __name__ == "__main__":
    # Initialize the example bank database
    init_db()
    print(f"\nOpenSource Router is running at http://localhost:{WEB_PORT}")
    print("Open that URL in your browser to start.\n")
    app.run(host="127.0.0.1", port=WEB_PORT, debug=False)
