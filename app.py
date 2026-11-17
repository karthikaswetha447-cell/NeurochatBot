
import os
import gradio as gr
from groq import Groq

# 1. Connect to Groq API
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# 2. Select AI model
model_name = "openai/gpt-oss-20b"


# 3. Main chatbot function
def neurobot(message, history):

    # Create system instructions
    messages = [
        {
            "role": "system",
            "content": """
You are NeuroBot, a friendly AI assistant created by NeuroGen AI Academy.

CONVERSATION RULES:
1. Always understand the current question using previous conversation.
2. If the user asks "What are its types?", identify the subject from previous messages.
3. Understand words like it, its, they, this and that using conversation history.
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

    # 4. Read previous conversation history
    for item in history or []:

        # Modern Gradio dictionary format
        if isinstance(item, dict):

            role = item.get("role")
            content = item.get("content")

            if (
                role in ["user", "assistant"]
                and isinstance(content, str)
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

    # 5. Add current user question
    messages.append({
        "role": "user",
        "content": message
    })

    # 6. Send complete conversation to Groq
    response = client.chat.completions.create(
        model=model_name,
        messages=messages
    )

    # 7. Return AI response
    return response.choices[0].message.content


# 8. Create Gradio interface
demo = gr.ChatInterface(
    fn=neurobot,
    title="🤖 NeuroBot",
    description="Your AI Assistant | Powered by NeuroGen AI Academy"
)


# 9. Launch application
if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
