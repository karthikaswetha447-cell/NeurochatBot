
import os
import json
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

                content = extract_text(item.get("content"))

                if content.strip():
                    recent_messages.append(content)

        elif isinstance(item, (list, tuple)) and len(item) == 2:

            user_message = item[0]

            if isinstance(user_message, str):
                recent_messages.append(user_message)

    return recent_messages[-4:]


# ==========================================
# 5. DYNAMIC CONTEXT AND FILTERING
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
                "certificate_issue_date": course_data.get(
                    "certificate_issue_date"
                )
            }

        elif any(word in question for word in [
            "duration", "hours", "how long"
        ]):

            relevant_data = {
                "course_name": course_data.get("course_name"),
                "course_duration": course_data.get("course_duration")
            }

        else:

            relevant_data = course_data

        return {
            "status": "success",
            "data": relevant_data,
            "source": "Official Course Information (course_info.json)"
        }

    except Exception as error:

        print("Grounding Source Error:", str(error))

        return {
            "status": "error",
            "data": None,
            "source": "course_info.json"
        }


# ==========================================
# 6. NEUROBOT CHAT FUNCTION
# ==========================================

def neurobot(message, history):

    messages = []

    system_prompt = """
You are NeuroBot, an AI assistant created by NeuroGen AI Academy.

YOUR ROLE:
- Help students and beginners.
- Explain concepts in simple English.
- Be patient, friendly and helpful.
- Explain technical words in an easy way.

ANSWERING STYLE:
- Give clear and simple answers.
- Use examples when helpful.
- Use short paragraphs.
- Answer the user's actual question.

GROUNDING RULES:
- For course-related questions, use only the official
  course information provided in the current request.
- Never invent course fees, dates, duration,
  certificate rules or eligibility.
- If a requested detail is missing or null,
  say it is not specified in the official information.
- Do not use previous conversation messages as
  official evidence.
- For non-course questions, you may use general knowledge.

SOURCE RULES:
- Mention the official source at the end of
  course-related answers.
- Use this format:
  Source: Official Course Information (course_info.json)

PRIVACY:
- Do not reveal private or confidential information.
"""

    messages.append({
        "role": "system",
        "content": system_prompt
    })


    # ======================================
    # OFFICIAL COURSE INFORMATION
    # ======================================

    grounding_context = build_relevant_context(
        message,
        history
    )

    if grounding_context is not None:

        if grounding_context["status"] == "error":

            return (
                "Sorry, I cannot access the official course "
                "information right now. Please try again later."
            )

        official_data = json.dumps(
            grounding_context["data"],
            ensure_ascii=False,
            indent=2
        )

        source_message = (
            "OFFICIAL COURSE INFORMATION:\n"
            + official_data
            + "\n\nSOURCE: "
            + grounding_context["source"]
            + "\n\nUse this information as the only authority "
            "for course-related facts. If a requested value "
            "is missing or null, say it is not specified. "
            "Do not guess. Include the source at the end."
        )

        messages.append({
            "role": "system",
            "content": source_message
        })


    # ======================================
    # CONVERSATION HISTORY
    # ======================================

    for item in history or []:

        if isinstance(item, dict):

            role = item.get("role")

            content = extract_text(item.get("content"))

            if role in ["user", "assistant"] and content.strip():

                messages.append({
                    "role": role,
                    "content": content
                })

        elif isinstance(item, (list, tuple)) and len(item) == 2:

            user_message, assistant_message = item

            if isinstance(user_message, str):

                messages.append({
                    "role": "user",
                    "content": user_message
                })

            if isinstance(assistant_message, str):

                messages.append({
                    "role": "assistant",
                    "content": assistant_message
                })


    # ======================================
    # CURRENT MESSAGE
    # ======================================

    messages.append({
        "role": "user",
        "content": message
    })


    # ======================================
    # GROQ RESPONSE
    # ======================================

    try:

        response = client.chat.completions.create(
            model=model_name,
            messages=messages
        )

        return response.choices[0].message.content

    except Exception as error:

        print("NeuroBot Error:", str(error))

        return "Sorry, something went wrong. Please try again."


# ==========================================
# 7. GRADIO INTERFACE
# ==========================================

demo = gr.ChatInterface(
    fn=neurobot,
    title="🤖 NeuroBot",
    description="Your AI Assistant | Powered by NeuroGen AI Academy"
)


# ==========================================
# 8. STARTUP: TEST PDF EMBEDDINGS
# ==========================================

if __name__ == "__main__":

    print("\n======================================")
    print("NEUROBOT STARTING")
    print("======================================")

    try:

        print("\nSTARTING EMBEDDING TEST...")

        from test_embeddings import create_embeddings

        create_embeddings()

        print("\nEMBEDDING TEST FINISHED.")

    except Exception as error:

        print("\nEMBEDDING TEST COULD NOT RUN:", str(error))


    # Start chatbot
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000))
    )
