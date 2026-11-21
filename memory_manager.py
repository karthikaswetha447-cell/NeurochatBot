"""
memory_manager.py  -  Short-term + Long-term memory for NeuroBot
=================================================================

This module has NO Gradio code, so it can be unit-tested on its own.

SHORT-TERM MEMORY  = the recent chat messages (kept by Gradio's chat history in app.py).
LONG-TERM MEMORY   = useful facts about the student (name, level, language, interests, goal)
                     saved in a database AFTER the student gives consent.

Database:
  * All environments    : SQLite file (set MEMORY_SQLITE_PATH to choose its location)
  * Render / production : mount a persistent disk and store the SQLite file on that disk

Main functions (used by app.py):
  classify_memory / classify_memory_all   -> what kind of memory is this message?
  decide_memory_action                    -> save? ask consent? ignore?
  MemoryStore.save_memory                 -> store a fact (upsert, no duplicates)
  MemoryStore.get_memories                -> read saved facts of ONE user
  MemoryStore.delete_memory / delete_all_memories
  select_relevant_memories / build_memory_prompt -> text added to the LLM prompt
"""

import os
import re
import json
import hmac
import hashlib
import secrets
import sqlite3
import threading
from datetime import datetime, timezone

# ==========================================
# 1. SETTINGS
# ==========================================

LONG_TERM_CONFIDENCE = float(os.environ.get("MEMORY_MIN_CONFIDENCE", "0.80"))
MAX_VALUE_CHARS = 200
PBKDF2_ROUNDS = 200_000          # password hashing strength (tests lower this)

MEMORY_TYPES = ("LONG_TERM_MEMORY", "SHORT_TERM_MEMORY", "NO_PERMANENT_MEMORY")

CATEGORIES = ("name", "learning_level", "language_preference",
              "learning_interest", "learning_goal")

CATEGORY_LABELS = {
    "name": "Name",
    "learning_level": "Learning level",
    "language_preference": "Preferred language",
    "learning_interest": "Learning interests",
    "learning_goal": "Learning goal",
}

LEVELS = ("Beginner", "Intermediate", "Advanced")
LEVEL_WORDS = {
    "beginner": "Beginner", "newbie": "Beginner", "fresher": "Beginner",
    "intermediate": "Intermediate",
    "advanced": "Advanced", "expert": "Advanced",
}

LANGUAGE_WORDS = ("tanglish", "tamil", "hindi", "telugu", "malayalam",
                  "kannada", "english", "hinglish")


# ==========================================
# 2. SMALL HELPERS
# ==========================================

def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def parse_json_block(text):
    """Return a dict parsed from the first {...} block in text, or None."""
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


# ==========================================
# 3. PRIVACY: SENSITIVE DATA + PROMPT-INJECTION CHECKS
# ==========================================

_SENSITIVE_SOURCES = [
    r"\b(?:password|passcode|passwd|pwd)\b\s*(?:is|are|was|:|=)",
    r"\bmy\s+(?:password|passcode|passwd|pin|otp|cvv)\b",
    r"\botp\b\s*(?:is|:|=)?\s*\d",
    r"\bapi[\s_-]?key\b\s*(?:is|:|=)",
