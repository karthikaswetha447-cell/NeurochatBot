
import os
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


# 4. Main chatbot function
def neurobot(message, history):

    messages = [
        {
            "role": "system",
            "content": """
You are NeuroBot, a friendly AI assistant created by NeuroGen AI Academy.

Always use previous conversation history to understand
the user's current question.

Understand words like:
it, its, they, them, this, that and those.

If the user asks "What are its types?",
identify the subject from previous messages.

Continue the previous topic naturally.
Do not ask users to repeat information already provided.

Give clear, beginner-friendly answers in simple English.
"""
        }
    ]

    # 5. Read previous conversation
    for item in history or []:

        if isinstance(item, dict):

            role = item.get("role")
            content = extract_text(item.get("content"))

            if role in ["user", "assistant"] and content.strip():

                messages.append({
                    "role": role,
                    "content": content
                })

        # Compatibility with older tuple history
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

    # 6. Add current question
    messages.append({
        "role": "user",
        "content": message
    })

    # 7. Send conversation to Groq
    response = client.chat.completions.create(
        model=model_name,
        messages=messages
    )

    # 8. Return answer
    return response.choices[0].message.content


# 9. Create Gradio interface
demo = gr.ChatInterface(
    fn=neurobot,
    title="🤖 NeuroBot",
    description="Your AI Assistant | Powered by NeuroGen AI Academy"
)


# 10. Launch application
if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
