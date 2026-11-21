import os
import re
import json
import math
import time
import hmac
import difflib
from collections import Counter
from datetime import date

import gradio as gr
from groq import Groq

import memory_manager as mm   # short-term / long-term memory logic
import multimodal_engine as multi  # image, audio, video helpers


# ==========================================
# 1. GROQ API CONNECTION
# ==========================================

client = Groq(api_key=os.environ["GROQ_API_KEY"])
model_name = "openai/gpt-oss-20b"

# Optional administrator settings (set them in Render > Environment)
#   ADMIN_PASSWORD     -> enables editing of the additional system instructions
#   SHOW_SYSTEM_PROMPT -> set to "false" to hide system instructions in the UI
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
SHOW_SYSTEM_PROMPT = os.environ.get("SHOW_SYSTEM_PROMPT", "true").strip().lower() not in ("false", "0", "no")
MAX_EXTRA_INSTRUCTION_CHARS = 800
MAX_MANUAL_INSTRUCTION_CHARS = 600   # user's own (manual) instructions

# RAG settings
RAG_MAX_FILE_MB = 5
RAG_MAX_CHARS = 300_000       # text read from one uploaded file
RAG_CHUNK_SIZE = 800          # characters per chunk
RAG_CHUNK_OVERLAP = 150       # characters shared between neighbouring chunks
RAG_TOP_K = 3                 # chunks sent to the model
TRUSTED_MIN_SCORE = 1.0       # minimum BM25 score for a trusted source (plus a strict keyword-match rule)
RAG_RELATIVE_CUTOFF = 0.40    # drop chunks scoring below 40% of the best chunk
# Allowed upload types. To accept ONLY PDF files, set the Render environment
# variable RAG_ALLOWED_TYPES=.pdf  (default: all six types)
_allowed = os.environ.get("RAG_ALLOWED_TYPES", ".txt,.md,.csv,.json,.pdf,.docx")
RAG_FILE_TYPES = [("." + t.strip().lower().lstrip(".")) for t in _allowed.split(",") if t.strip()]


# ==========================================
# 2. HELPERS
# ==========================================

def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return ""


def parse_json_block(text):
    """Return a dict parsed from the first {...} block in text, or None."""
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None


def validate_json(text):
    """Strict JSON validation. Returns (is_valid, parsed_or_None, error_or_None)."""
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE).strip()
    try:
        return True, json.loads(t), None
    except Exception as error:
        return False, None, str(error)


def safe_error(error):
    """Error text without any chance of leaking the API key."""
    text = f"{type(error).__name__}: {error}"
    key = os.environ.get("GROQ_API_KEY", "")
    if key:
        text = text.replace(key, "***")
    db_url = os.environ.get("DATABASE_URL", "")
    if db_url:
        text = text.replace(db_url, "***")
