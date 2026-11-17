
import os
import json
import gradio as gr
from groq import Groq


# GROQ API CONNECTION
client = Groq(
    api_key=os.environ["GROQ_API_KEY"]
)

model_name = "openai/gpt-oss-20b"


# EXTRACT TEXT FROM CONVERSATION HISTORY
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


# CHECK WHETHER QUESTION IS COURSE-RELATED
def is_course_question(message):

    keywords = [
        "course", "fee", "fees", "price", "cost",
        "payment", "amount", "certificate",
        "certificates", "viva", "eligibility",
        "duration", "class", "classes",
        "completion", "issue date", "exam",
        "applied ai", "foundation"
    ]

    question = message.lower()

    return any(word in question for word in keywords)


# DYNAMIC CONTEXT + CONTEXT FILTERING + GROUNDING
def build_relevant_context(message):

    if not is_course_question(message):
        return None

    try:
        with open("course_info.json", "r", encoding="utf-8") as file:
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

        return json.dumps(
            relevant_data,
            ensure_ascii=False,
            indent=2
        )

    except Exception:
        return (
            "OFFICIAL COURSE SOURCE COULD NOT BE LOADED. "
            "Do not provide or guess course-related information."
        )


# MAIN NEUROBOT FUNCTION
def neurobot(message, history):

    messages = []

    # SYSTEM CONTEXT + GROUNDING RULES
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
- Use the official course information provided in the context
  for all course-related questions.
- Never invent course fees, dates, duration, certificate rules,
  eligibility or payment information.
- If the official information is missing or its value is null,
  clearly say:
  "Sorry, this information is not available in the official course details."
- Do not treat previous conversation messages as proof of
  official course information.
- If the official source cannot be loaded, do not guess.
- For questions unrelated to the academy's course details,
  you may answer using your general knowledge.
- Never claim that information is official unless it is
  supported by the provided official source.

PRIVACY:
- Do not reveal private or confidential information.
"""

    messages.append({
        "role": "system",
        "content": system_prompt
    })

    # ADD OFFICIAL COURSE INFORMATION
    relevant_context = build_relevant_context(message)

    if relevant_context is not None:

        messages.append({
            "role": "system",
            "content": (
                "OFFICIAL COURSE INFORMATION:\n"
                + relevant_context
                + "\n\nUse only this source for course-related facts. "
                "If a required value is null, missing, or the source "
                "could not be loaded, say the information is unavailable."
            )
        })

    # CONVERSATION HISTORY
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

    # CURRENT USER MESSAGE
    messages.append({
        "role": "user",
        "content": message
    })

    # SEND REQUEST TO GROQ
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


# GRADIO CHAT INTERFACE
demo = gr.ChatInterface(
    fn=neurobot,
    title="🤖 NeuroBot",
    description="Your AI Assistant | Powered by NeuroGen AI Academy"
)


# LAUNCH APPLICATION
if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000))
    )
