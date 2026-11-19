import os
import re
import json
import time
import hmac
import difflib
import gradio as gr
from groq import Groq


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
    return text[:400]


def short(text, limit=400):
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + " ..."


# ==========================================
# 3. COURSE QUESTION DETECTION (CONTEXT FILTERING)
# ==========================================

COURSE_KEYWORDS = [
    "course", "fee", "fees", "price", "cost",
    "payment", "amount", "certificate", "certificates",
    "viva", "eligibility", "duration", "class", "classes",
    "completion", "issue date", "exam", "applied ai",
    "foundation", "admission", "enrollment", "enrolment"
]


def is_course_question(message):
    question = message.lower()
    return any(word in question for word in COURSE_KEYWORDS)
