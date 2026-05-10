"""
Claude Code agent — dispatches the user's coding task to Claude Code CLI
via subprocess and returns the result. Supports conversation continuity
using --session-id flag (same ID across calls).

Claude runs with file tools enabled (Read, Write, Edit, Bash) so it can
actually read and modify files in the user's project directory.
"""

import subprocess
from config import CLAUDE_COMMAND, CLAUDE_CODER_PROMPT_PATH
from classifier import load_prompt


def generate_with_claude(
    user_prompt: str,
    session_id: str = None,
    is_followup: bool = False,
    project_dir: str = None,
) -> str:
    """
    Send the user's coding task to Claude Code CLI using the -p flag.
    Claude runs with file access so it can read, write, and edit project files.

    Uses the same --session-id for all calls so Claude remembers the conversation.

    Args:
        user_prompt: The coding task or follow-up feedback
        session_id: UUID for conversation continuity (same ID used every call)
        is_followup: True if this is a follow-up message in an existing conversation
        project_dir: Path to the project directory Claude should work in
    """
    if is_followup:
        print("Sending follow-up to Claude Code...")
    else:
        print("Sending to Claude Code...")

    try:
        escaped_prompt = _escape_for_shell(user_prompt)

        # Base command — same flags for first call and follow-ups
        # --session-id keeps all calls in the same conversation
        # --dangerously-skip-permissions lets Claude edit files without prompting
        # --allowedTools gives Claude access to file and shell tools
        command = (
            f'"{CLAUDE_COMMAND}"'
            f' -p "{escaped_prompt}"'
            f" --session-id {session_id}"
            f' --allowedTools "Read,Write,Edit,Bash"'
            f" --dangerously-skip-permissions"
        )

        # On the first call, add the system prompt that tells Claude how to behave
        if not is_followup:
            system_prompt = load_prompt(CLAUDE_CODER_PROMPT_PATH)
            command += f' --append-system-prompt "{_escape_for_shell(system_prompt)}"'

        # Give Claude access to the project directory
        if project_dir:
            command += f' --add-dir "{project_dir}"'

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            shell=True,
            timeout=600,  # 10 minute timeout for complex tasks
            cwd=project_dir,  # Run Claude inside the project directory
        )

        # If Claude returned an error exit code, show stderr
        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            return f"Error: Claude Code returned exit code {result.returncode}\n{error_msg}"

        # Return Claude's response
        return result.stdout.strip()

    except FileNotFoundError:
        return (
            "Error: 'claude' command not found. "
            "Make sure Claude Code CLI is installed and on your PATH."
        )
    except subprocess.TimeoutExpired:
        return "Error: Claude Code request timed out after 10 minutes."
    except Exception as e:
        return f"Error running Claude Code: {e}"


def _escape_for_shell(text: str) -> str:
    """Escape double quotes in text so it's safe inside a shell double-quoted string."""
    return text.replace('"', '\\"')
