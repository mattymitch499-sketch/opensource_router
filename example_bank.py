"""
Example Bank — SQLite storage for all router interactions.

Saves every coding task (prompt, score, output, review result) to a local database.
This data can later be used as few-shot examples to improve DeepSeek's code generation.

The database file is created automatically the first time the router runs.
"""

import json
import sqlite3
from datetime import datetime
from config import EXAMPLE_BANK_DB_PATH


def init_db() -> None:
    """Create the examples table if it doesn't exist yet.

    Called once at the start of each router session.
    If anything goes wrong, prints a warning but never crashes the router.
    """
    try:
        with sqlite3.connect(EXAMPLE_BANK_DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS examples (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       TEXT NOT NULL,
                    user_prompt     TEXT NOT NULL,
                    complexity_score INTEGER NOT NULL,
                    routed_to       TEXT NOT NULL,
                    generated_code  TEXT,
                    review_verdict  TEXT,
                    review_issues   TEXT,
                    was_escalated   INTEGER NOT NULL DEFAULT 0,
                    techniques      TEXT
                )
            """)

            # If the table already exists without the techniques column, add it
            # (this handles upgrading from the old schema)
            try:
                conn.execute("ALTER TABLE examples ADD COLUMN techniques TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists, that's fine

            conn.commit()
        print("Example bank ready.")
    except Exception as e:
        print(f"Warning: Could not initialize example bank: {e}")


def save_example(
    user_prompt: str,
    complexity_score: int,
    routed_to: str,
    generated_code: str,
    review_verdict: str | None,
    review_issues: list | None,
    was_escalated: bool,
    techniques: list[str] | None = None
) -> None:
    """Save one interaction to the database.

    Args:
        user_prompt: The original coding task from the user.
        complexity_score: The 1-10 complexity rating from the classifier.
        routed_to: Which agent handled it — "deepseek" or "claude".
        generated_code: The final code output shown to the user.
        review_verdict: "pass", "fail", or None (Claude path skips review).
        review_issues: List of issues found by reviewer, or None.
        was_escalated: True if DeepSeek failed twice and we escalated to Claude.
        techniques: List of coding technique tags like ["for_loop", "input_validation"].
    """
    try:
        # Convert lists to JSON strings for storage
        issues_json = json.dumps(review_issues) if review_issues else None
        techniques_json = json.dumps(techniques) if techniques else None

        with sqlite3.connect(EXAMPLE_BANK_DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO examples
                    (timestamp, user_prompt, complexity_score, routed_to,
                     generated_code, review_verdict, review_issues, was_escalated,
                     techniques)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(),
                    user_prompt,
                    complexity_score,
                    routed_to,
                    generated_code,
                    review_verdict,
                    issues_json,
                    int(was_escalated),
                    techniques_json,
                )
            )
            conn.commit()
        print("Example saved to bank.")
    except Exception as e:
        print(f"Warning: Could not save example: {e}")


def get_similar_examples(prompt: str, techniques: list[str] | None = None, limit: int = 3) -> list[dict]:
    """Find past examples using keyword matching on prompts AND technique tags.

    Searches both the user_prompt text and the stored technique tags.
    This means examples can match across totally different projects if they
    use similar coding patterns (e.g. "for_loop", "input_validation").

    Only returns examples where the review passed (we want good examples
    as few-shot context, not broken ones).

    Args:
        prompt: The current user prompt to find matches for.
        techniques: Optional list of technique tags to search for.
        limit: Max number of examples to return (default 3).

    Returns:
        List of dicts with keys: user_prompt, generated_code, review_verdict, routed_to, techniques.
    """
    try:
        # Split the prompt into keywords (skip short words like "a", "the", etc.)
        words = [w.lower() for w in prompt.split() if len(w) > 3]

        if not words and not techniques:
            return []

        # Build WHERE conditions — match on prompt keywords OR technique tags
        where_parts = []
        params = []

        # Match prompt keywords against user_prompt text
        for word in words:
            where_parts.append("LOWER(user_prompt) LIKE ?")
            params.append(f"%{word}%")

        # Match technique tags against the stored techniques JSON
        if techniques:
            for tag in techniques:
                where_parts.append("LOWER(techniques) LIKE ?")
                params.append(f"%{tag.lower()}%")

        where_clause = " OR ".join(where_parts)

        # Only return examples that passed review (good examples for few-shot)
        query = f"""
            SELECT user_prompt, generated_code, review_verdict, routed_to, techniques
            FROM examples
            WHERE ({where_clause})
              AND (review_verdict = 'pass' OR review_verdict IS NULL)
            ORDER BY timestamp DESC
            LIMIT ?
        """
        params.append(limit)

        with sqlite3.connect(EXAMPLE_BANK_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

        # Convert sqlite3.Row objects to plain dicts
        return [dict(row) for row in rows]

    except Exception as e:
        print(f"Warning: Could not search example bank: {e}")
        return []


def get_all_examples(limit: int = 50) -> list[dict]:
    """Get the most recent examples from the database.

    Useful for debugging or browsing your interaction history.

    Args:
        limit: Max number of rows to return (default 50).

    Returns:
        List of dicts, one per row, newest first.
    """
    try:
        with sqlite3.connect(EXAMPLE_BANK_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM examples ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()

        return [dict(row) for row in rows]

    except Exception as e:
        print(f"Warning: Could not read example bank: {e}")
        return []
