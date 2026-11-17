
import os
import json
import gradio as gr
from groq import Groq

# 1. Connect to Groq API
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# 2. Select AI model
model_name = "openai/gpt-oss-20b"


# 3. Extract text from Gradio history
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


# 4. Context Filtering
def build_relevant_context(message):

    try:
        with open("course_info.json", "r", encoding="utf-8") as file:
            course_data = json.load(file)

    except (FileNotFoundError, json.JSONDecodeError) as error:

        print("Course information error:", error)
        return ""

    question = message.lower()

    # Check whether the user asks about course fees
    fee_keywords = [
        "fee", "fees", "price", "cost",
        "how much", "amount", "payment"
    ]

    if any(word in question for word in fee_keywords):

        return f"""
RELEVANT INFORMATION FOR THIS QUESTION:

Course Name: {course_data['course_name']}
Current Course Fee: {course_data['currency']} {course_data['course_fee']}
Last Updated: {course_data['updated_on']}

Use this current information to answer the fee question.
Do not use an outdated fee.
"""

    # Check whether the user asks about the course name
    course_keywords = [
        "course name", "which course",
        "course title", "what course"
    ]

    if any(word in question for word in course_keywords):

        return f"""
RELEVANT INFORMATION:

Course Name: {course_data['course_name']}
"""

    # No relevant course information needed
    return ""


# 5. Main chatbot function
def neurobot(message, history):

    # System instructions
    messages = [
        {
            "role": "system",
            "content": """
You are NeuroBot, a friendly AI assistant created by NeuroGen AI Academy.

CONVERSATION RULES:
1. Always understand the current question using previous conversation.
2. If the user asks "What are its types?",
   identify the subject from previous messages.
3. Understand words like it, its, they, them,
   this and that using conversation history.
4. Continue the previous topic naturally.
5. Do not ask users to repeat information already provided.
6. If context is genuinely unclear, ask a clarification question.

RESPONSE STYLE:
- Give clear and beginner-friendly explanations.
- Use simple English.
- Provide practical examples.
"""
        }
    ]

    # 6. Add only relevant Dynamic Context
    relevant_context = build_relevant_context(message)

    if relevant_context:

        messages.append({
            "role": "system",
            "content": relevant_context
        })

    # 7. Read previous conversation history
    for item in history or []:

        # Modern Gradio message format
        if isinstance(item, dict):

            role = item.get("role")
            content = extract_text(item.get("content"))

            if (
                role in ["user", "assistant"]
                and content.strip()
            ):
                messages.append({
                    "role": role,
                    "content": content
                })

        # Older Gradio tuple/list format
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

    # 8. Add current user question
    messages.append({
        "role": "user",
        "content": message
    })

    # 9. Send information to Groq
    response = client.chat.completions.create(
        model=model_name,
        messages=messages
    )

    # 10. Return AI response
    return response.choices[0].message.content


# 11. Create Gradio interface
demo = gr.ChatInterface(
    fn=neurobot,
    title="🤖 NeuroBot",
    description="Your AI Assistant | Powered by NeuroGen AI Academy"
)


# 12. Launch application
if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
