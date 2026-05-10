"""
OpenSource Router — Main CLI entry point.

Routes coding tasks to either DeepSeek (local, free) or Claude Code (powerful, paid)
based on a complexity score from 1-10. After the first response, enters an interactive
feedback loop so you can iterate on the code.

Usage:
    python router.py "your coding task here"
    python router.py --project-dir "C:\\path\\to\\project" "refactor the auth module"
    python router.py --resume                  (reconnect to your last Claude session)
"""

import argparse
import os
import sys
import uuid
from config import COMPLEXITY_THRESHOLD, LAST_SESSION_PATH
from classifier import classify_task
from deepseek_agent import generate_with_deepseek
from claude_agent import generate_with_claude
from testing_agent import review_code, tag_techniques
from example_bank import init_db, save_example


def save_session(session_id: str, project_dir: str) -> None:
    """Save the current Claude session ID and project dir to a file so we can resume later."""
    try:
        with open(LAST_SESSION_PATH, "w") as f:
            f.write(f"{session_id}\n{project_dir}")
    except Exception as e:
        print(f"Warning: Could not save session: {e}")


def load_session() -> tuple[str, str] | None:
    """Load the last saved Claude session ID and project dir.

    Returns a (session_id, project_dir) tuple, or None if no saved session exists.
    """
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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Route coding tasks to DeepSeek (simple) or Claude Code (complex)."
    )
    parser.add_argument(
        "prompt",
        nargs="?",  # Make prompt optional so --resume can work without it
        default=None,
        help="The coding task to route (e.g. \"write a function that reverses a string\")"
    )
    parser.add_argument(
        "--project-dir",
        default=os.getcwd(),
        help="Directory of the project Claude should work in (default: current directory)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume your last Claude Code session (skips classification)"
    )
    return parser.parse_args()


def main() -> None:
    """Main function — parses the user's prompt, routes it, then enters feedback loop."""

    args = parse_args()
    project_dir = args.project_dir

    # Initialize the example bank database (creates table if first run)
    init_db()

    # Handle --resume: reconnect to the last Claude session
    if args.resume:
        saved = load_session()
        if saved is None:
            print("No saved session found. Run a task first, then use --resume.")
            sys.exit(1)

        session_id, saved_project_dir = saved
        # Use saved project dir unless the user explicitly passed a different one
        if args.project_dir == os.getcwd():
            project_dir = saved_project_dir

        print(f"Resuming Claude session: {session_id[:8]}...")
        print(f"Project directory: {project_dir}")
        using_claude = True
        deepseek_history = []

        # If the user also passed a prompt, send it as a follow-up right away
        if args.prompt:
            result = generate_with_claude(args.prompt, session_id=session_id, is_followup=True, project_dir=project_dir)
            print(result)

        # Jump straight to the feedback loop (skip classification and routing)
        # (The feedback loop code is below)

    else:
        # Normal flow: classify and route
        if args.prompt is None:
            print("Error: Please provide a coding task, or use --resume to continue a session.")
            sys.exit(1)

        user_prompt = args.prompt
        print(f"Project directory: {project_dir}")

        # Step 1: Classify the task complexity using DeepSeek as a judge
        print("Analyzing task complexity...")
        score = classify_task(user_prompt)

        # Generate a unique session ID for Claude conversation continuity
        session_id = str(uuid.uuid4())

        # Track which agent we're using and keep DeepSeek conversation history
        using_claude = score > COMPLEXITY_THRESHOLD
        deepseek_history = []  # Stores message history for DeepSeek multi-turn conversations

        # Step 2: Route based on the score
        if using_claude:
            print(f"Complexity: {score}/10 — Routing to Claude Code")
            print("-" * 50)
            result = generate_with_claude(user_prompt, session_id=session_id, is_followup=False, project_dir=project_dir)
            print(result)
            tags = tag_techniques(result)
            save_example(user_prompt, score, "claude", result, None, None, False, tags)
        else:
            print(f"Complexity: {score}/10 — Routing to DeepSeek (local)")
            print("-" * 50)
            # Generate code with DeepSeek
            result = generate_with_deepseek(user_prompt)

            # Run the testing agent to review DeepSeek's output
            print("\n" + "-" * 50)
            review = review_code(user_prompt, result)

            if review["verdict"] == "pass":
                # Code passed review — show it to the user
                print("Review passed!")
                tags = tag_techniques(result)
                save_example(user_prompt, score, "deepseek", result, review["verdict"], review["issues"], False, tags)
            else:
                # Code failed review — show issues and retry once
                print(f"Review failed: {', '.join(review['issues'])}")
                print("Retrying with feedback...")
                print("-" * 50)

                # Build a retry prompt that includes the reviewer's feedback
                retry_prompt = (
                    f"Your previous code had these issues: {', '.join(review['issues'])}. "
                    f"Please fix them. Original task: {user_prompt}"
                )
                result = generate_with_deepseek(retry_prompt)

                # Review the retry attempt
                print("\n" + "-" * 50)
                review2 = review_code(user_prompt, result)

                if review2["verdict"] == "pass":
                    print("Review passed!")
                    tags = tag_techniques(result)
                    save_example(user_prompt, score, "deepseek", result, review2["verdict"], review2["issues"], False, tags)
                else:
                    # Failed twice — escalate to Claude
                    print(f"Review failed again: {', '.join(review2['issues'])}")
                    print("Escalating to Claude Code...")
                    print("-" * 50)
                    using_claude = True
                    escalation_prompt = (
                        f"Original task: {user_prompt}\n\n"
                        f"A local model attempted this twice but failed code review. "
                        f"Issues found: {', '.join(review2['issues'])}. "
                        f"Please generate a correct solution."
                    )
                    result = generate_with_claude(escalation_prompt, session_id=session_id, is_followup=False, project_dir=project_dir)
                    print(result)
                    tags = tag_techniques(result)
                    save_example(user_prompt, score, "claude", result, review2["verdict"], review2["issues"], True, tags)

            # Initialize conversation history for follow-ups (if still on DeepSeek)
            if not using_claude:
                from config import CODER_SYSTEM_PROMPT_PATH
                from classifier import load_prompt
                system_prompt = load_prompt(CODER_SYSTEM_PROMPT_PATH)
                deepseek_history.append({"role": "system", "content": system_prompt})
                deepseek_history.append({"role": "user", "content": user_prompt})
                deepseek_history.append({"role": "assistant", "content": result})

        # Save the session so it can be resumed later with --resume
        if using_claude:
            save_session(session_id, project_dir)

    # Step 3: Interactive feedback loop — let the user iterate on the code
    print("\n" + "=" * 50)
    print("Type feedback to iterate on the code.")
    print("Commands: 'quit' to exit, 'escalate' to switch to Claude")
    print("=" * 50)

    while True:
        try:
            feedback = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            # User pressed Ctrl+C or Ctrl+D — exit gracefully
            print("\nDone. Goodbye!")
            break

        # Skip empty input
        if not feedback:
            continue

        # Check for exit commands
        if feedback.lower() in ("quit", "exit", "q"):
            print("Done. Goodbye!")
            break

        # Check for escalation from DeepSeek to Claude
        if feedback.lower() == "escalate" and not using_claude:
            print("\nEscalating to Claude Code...")
            using_claude = True
            session_id = str(uuid.uuid4())
            # Give Claude the original task + context about the escalation
            escalation_prompt = (
                f"Original task: {args.prompt}\n\n"
                f"A local model attempted this but the result needs improvement. "
                f"Please generate a better solution."
            )
            result = generate_with_claude(escalation_prompt, session_id=session_id, is_followup=False, project_dir=project_dir)
            print(result)
            # Save session so it can be resumed later
            save_session(session_id, project_dir)
            continue

        # Send follow-up feedback to whichever agent is active
        if using_claude:
            result = generate_with_claude(feedback, session_id=session_id, is_followup=True, project_dir=project_dir)
            print(result)
        else:
            result = generate_with_deepseek(feedback, conversation_history=deepseek_history)
            # (deepseek_history is updated inside the function automatically)


if __name__ == "__main__":
    main()
