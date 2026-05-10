# OpenSource_Router

## What this project is
A Python CLI tool that acts as an intelligent task router between a local DeepSeek R1 14B model (via Ollama) and Claude Code. The user types a natural language coding prompt, the router classifies its complexity, and dispatches it to the appropriate model — DeepSeek for simple tasks, Claude for complex ones. This saves Claude Code tokens/usage by offloading easy work to the free local model.

## Architecture

### Core flow
1. User types a coding task prompt into the CLI
2. Router agent (DeepSeek via Ollama API at localhost:11434) evaluates complexity on a 1-10 scale
3. Score <= 5 → DeepSeek generates the code locally via Ollama API
4. Score > 5 → Task is forwarded to Claude Code via subprocess call
5. Result is displayed back to the user in the terminal

### Complexity scoring criteria
The router evaluates these dimensions:
- **Scope**: Single function vs multi-file changes (1-10)
- **Reasoning**: Boilerplate vs novel algorithmic logic (1-10)
- **Debugging**: Generating new code vs diagnosing existing bugs (1-10)
- **Architecture**: Isolated task vs requires understanding broad codebase structure (1-10)
- **Domain complexity**: Simple CRUD vs complex domains like auth, concurrency, ML (1-10)

Final score = weighted average. Threshold is configurable (default: 5).

### Built components
- **Testing agent**: After DeepSeek generates code, a reviewer model evaluates the output. If it fails review twice, auto-escalate to Claude.
- **Example bank**: A local SQLite database (`example_bank.db`, auto-created on first run) that logs every interaction — prompt, complexity score, which model handled it, generated code, and review results. Retrieval function (`get_similar_examples`) finds related past examples via keyword matching.
- **Few-shot injection**: When DeepSeek starts a fresh task, it searches the example bank for similar past prompts. If found, successful past solutions are injected into the prompt as reference examples so DeepSeek can learn from what worked before.

### Future components (not built yet)
- **DeepSeek file editing**: Instead of printing code to terminal, give DeepSeek the ability to read/write files directly.

## Tech stack
- Python 3.x
- Ollama API (localhost:11434) for DeepSeek R1 14B
- Claude Code CLI (subprocess) for complex task dispatch
- SQLite for example bank (stores all interactions)
- No external frameworks needed — keep dependencies minimal

## Project structure
```
OpenSource_Router/
├── CLAUDE.md
├── router.py          # Main CLI entry point
├── classifier.py      # Complexity scoring logic (calls DeepSeek)
├── deepseek_agent.py  # DeepSeek code generation via Ollama API
├── claude_agent.py    # Claude Code dispatch via subprocess
├── config.py          # Threshold, model name, API URLs, prompts
├── example_bank.py    # SQLite storage for all interactions (save/retrieve)
├── router_app.py      # Web UI entry point (Flask server, replaces router.py)
├── router.py          # Original CLI entry point (still works, but web UI is primary)
├── example_bank.db    # Auto-created SQLite database (not in git)
├── .last_session      # Auto-created file storing last Claude session ID (not in git)
├── templates/
│   └── index.html     # Chat UI page (single file with embedded CSS/JS)
├── prompts/
│   ├── router_system.txt    # System prompt for the router/classifier
│   └── coder_system.txt     # System prompt for DeepSeek coding mode
├── requirements.txt   # Python dependencies (flask, requests)
└── README.md
```

## Conventions
- Matthew is a beginner-level Python coder. Write clean, well-commented code with clear variable names. Avoid overly clever patterns.
- Use type hints on function signatures
- Use f-strings for string formatting
- Print clear status messages so the user knows what's happening at each step (e.g. "Analyzing task complexity...", "Routing to DeepSeek (score: 3/10)...")
- Keep it simple — no async, no classes unless truly needed, just clean functions
- All config (thresholds, model names, URLs) lives in config.py, not hardcoded

## How to run
```bash
# Web UI (primary)
python router_app.py                       # Starts server at http://localhost:5000

# CLI (still works)
python router.py "write a function that validates email addresses"
python router.py --resume                  # Reconnect to last Claude session
python router.py --resume "add error handling"  # Resume and send a new prompt
```

## Key API details
- Ollama chat endpoint: POST http://localhost:11434/api/chat
- Request body: {"model": "deepseek-r1:14b", "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}], "stream": false}
- Response: JSON with message.content field containing the model's reply
