import os
import json
import time
import gradio as gr
from groq import Groq


# ==========================================
# 1. GROQ API CONNECTION
# ==========================================

client = Groq(
    api_key=os.environ["GROQ_API_KEY"]
)

model_name = "openai/gpt-oss-20b"


# ==========================================
# 2. EXTRACT TEXT FROM HISTORY
# ==========================================

def extract_text(content):

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        text_parts = []

        for block in content:

            if isinstance(block, str):
                text_parts.append(block)

            elif isinstance(block, dict):

                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))

        return "\n".join(text_parts)

    return ""


# ==========================================
# 3. COURSE QUESTION DETECTION
# ==========================================

def is_course_question(message):

    keywords = [
        "course", "fee", "fees", "price", "cost",
        "payment", "amount", "certificate",
        "certificates", "viva", "eligibility",
        "duration", "class", "classes",
        "completion", "issue date", "exam",
        "applied ai", "foundation", "admission",
        "enrollment", "enrolment"
    ]

    question = message.lower()

    return any(word in question for word in keywords)


# ==========================================
# 4. RECENT USER MESSAGES
# ==========================================

def get_recent_user_messages(history):

    recent_messages = []

    for item in history or []:

        if isinstance(item, dict):

            if item.get("role") == "user":

                content = extract_text(
                    item.get("content")
                )

                if content.strip():
                    recent_messages.append(content)

        elif isinstance(item, (list, tuple)) and len(item) == 2:

            user_message = item[0]

            if isinstance(user_message, str):
                recent_messages.append(user_message)

    return recent_messages[-4:]
