"""
Configuration for the OpenSource Router.
All settings live here — nothing is hardcoded in other files.
"""

import os

# --- Complexity threshold ---
# Tasks scoring at or below this number go to DeepSeek.
# Tasks scoring above this number go to Claude Code.
COMPLEXITY_THRESHOLD = 5

# --- Ollama settings ---
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_CHAT_ENDPOINT = f"{OLLAMA_BASE_URL}/api/chat"

# --- Model assignments ---
# Small fast model for simple JSON tasks (classification and review)
CLASSIFIER_MODEL = "qwen2.5:3b"
REVIEWER_MODEL = "qwen2.5:3b"
# Full reasoning model for actual code generation
CODER_MODEL = "deepseek-r1:14b"

# --- Claude Code CLI command ---
CLAUDE_COMMAND = "claude"

# --- File paths for system prompts ---
_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
ROUTER_SYSTEM_PROMPT_PATH = os.path.join(_PROMPTS_DIR, "router_system.txt")
CODER_SYSTEM_PROMPT_PATH = os.path.join(_PROMPTS_DIR, "coder_system.txt")
CLAUDE_CODER_PROMPT_PATH = os.path.join(_PROMPTS_DIR, "claude_coder_system.txt")
REVIEWER_SYSTEM_PROMPT_PATH = os.path.join(_PROMPTS_DIR, "reviewer_system.txt")
TAGGER_SYSTEM_PROMPT_PATH = os.path.join(_PROMPTS_DIR, "tagger_system.txt")
# Model for tagging code techniques (same fast model as classifier/reviewer)
TAGGER_MODEL = "qwen2.5:3b"

# --- Example bank (SQLite) ---
# Database file is stored in the project root alongside the other files
EXAMPLE_BANK_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_bank.db")

# --- Web UI settings ---
WEB_PORT = 5000

# --- Session resume ---
# Stores the last Claude session ID so you can reconnect with --resume
LAST_SESSION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".last_session")
