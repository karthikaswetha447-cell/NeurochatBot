
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
# 2. EXTRACT TEXT FROM CONVERSATION HISTORY
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
# 3. CHECK COURSE-RELATED QUESTIONS
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
# 4. GET RECENT USER MESSAGES
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
# 5. DYNAMIC CONTEXT + CONTEXT FILTERING
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

        # FEE INFORMATION
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

        # CERTIFICATE INFORMATION
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

        # COURSE DURATION
        elif any(word in question for word in [
            "duration", "hours", "how long"
        ]):

            relevant_data = {
                "course_name": course_data.get("course_name"),
                "course_duration": course_data.get("course_duration")
            }

        # OTHER COURSE QUESTIONS
        else:

            relevant_data = course_data

        return {
            "status": "success",
            "data": relevant_data,
            "source": "Official Course Information (course_info.json)"
        }

    except Exception as e:

        print("Grounding Source Error:", str(e))

        return {
            "status": "error",
            "data": None,
            "source": "course_info.json"
        }


# ==========================================
# 6. MAIN NEUROBOT FUNCTION
# ==========================================

def neurobot(message, history):

    messages = []

    # SYSTEM CONTEXT
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
- For course-related questions, use only the official course
  information provided in the current request.
- Never invent course fees, dates, duration, certificate rules,
  eligibility or payment information.
- Treat the official course information as the authority.
- Previous conversation messages are not official evidence.
- If a requested detail is missing or its value is null,
  say that it is not specified in the official course information.
- If the official source cannot be loaded, do not guess.
- Do not use general knowledge to fill missing official details.
- For non-course questions, you may use general knowledge.

SOURCE CITATION RULES:
- Every answer based on official course information must
  mention the source at the end.
- Use this format:

  Source: Official Course Information (course_info.json)

- If the requested information is missing, still mention
  the source and explain that the detail is not specified.
- Never invent or claim a source that was not provided.

PRIVACY:
- Do not reveal private or confidential information.
"""

    messages.append({
        "role": "system",
        "content": system_prompt
    })


    # ======================================
    # LOAD OFFICIAL COURSE INFORMATION
    # ======================================

    grounding_context = build_relevant_context(message, history)

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
            + "\n\nSOURCE NAME: "
            + grounding_context["source"]
            + "\n\nUse this information as the only authority "
            "for course-related facts. If a required value is "
            "missing or null, say it is not specified. "
            "Do not guess or use previous conversation answers "
            "as official evidence. Include the source name "
            "at the end of your answer."
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
    # CURRENT USER MESSAGE
    # ======================================

    messages.append({
        "role": "user",
        "content": message
    })


    # ======================================
    # SEND REQUEST TO GROQ
    # ======================================

    try:

        response = client.chat.completions.create(
            model=model_name,
            messages=messages
        )

        return response.choices[0].message.content

    except Exception as e:

        print("NeuroBot Error:", str(e))

        return (
            "Sorry, something went wrong. "
            "Please try again."
        )


# ==========================================
# 7. GRADIO CHAT INTERFACE
# ==========================================

demo = gr.ChatInterface(
    fn=neurobot,
    title="🤖 NeuroBot",
    description="Your AI Assistant | Powered by NeuroGen AI Academy"
)


# ==========================================
# 8. LAUNCH APPLICATION + TEST PDF
# ==========================================

if __name__ == "__main__":

    print("\n====================================")
    print("NEUROBOT STARTING")
    print("====================================")

    # Run PDF test during Render startup
    try:

        print("\n========== STARTING PDF TEST ==========")

        from test_pdf import test_pdf

        test_pdf()

        print("\n========== PDF TEST COMPLETED ==========")

    except Exception as e:

        print("\nPDF TEST COULD NOT RUN:", str(e))


    # Start Gradio application
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000))
    )
