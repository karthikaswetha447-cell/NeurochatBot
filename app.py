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


# ==========================================
# 5. OFFICIAL COURSE INFORMATION
# ==========================================

def build_relevant_context(message, history):

    recent_messages = get_recent_user_messages(history)

    course_related = is_course_question(message)

    if not course_related:

        course_related = any(
            is_course_question(old_message)
            for old_message in recent_messages
        )

    if not course_related:
        return None

    try:

        base_dir = os.path.dirname(os.path.abspath(__file__))

        course_info_path = os.path.join(
            base_dir,
            "course_info.json"
        )

        with open(
            course_info_path,
            "r",
            encoding="utf-8"
        ) as file:

            course_data = json.load(file)

        question = message.lower()

        if any(word in question for word in [
            "fee", "fees", "price", "cost",
            "payment", "amount"
        ]):

            relevant_data = {
                "course_name": course_data.get("course_name"),
                "course_fee": course_data.get("course_fee"),
                "currency": course_data.get("currency"),
                "updated_on": course_data.get("updated_on")
            }

        elif any(word in question for word in [
            "certificate", "certificates",
            "viva", "issue date", "eligibility"
        ]):

            relevant_data = {
                "course_name": course_data.get("course_name"),
                "certificate": course_data.get("certificate"),
                "certificate_eligibility": course_data.get(
                    "certificate_eligibility"
                ),
