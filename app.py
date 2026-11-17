
import os
import gradio as gr
from groq import Groq

# 1. Connect to Groq
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# 2. Select AI model
model_name = "openai/gpt-oss-20b"


# 3. Main chatbot function
def neurobot(message, history):

    print("CURRENT MESSAGE:", message)
    print("CONVERSATION HISTORY:", repr(history))

    # Create messages list
    messages = [
        {
            "role": "system",
            "content": (
                "You are NeuroBot, a friendly AI assistant "
                "created by NeuroGen AI Academy. "
                "Use the previous conversation to understand "
                "follow-up questions. If the user says 'it', "
                "'its', 'they' or similar words, use the "
                "conversation context to identify the subject. "
                "Answer clearly and helpfully."
            )
        }
    ]

    # Read previous conversation
    for item in history or []:

        if isinstance(item, dict):

            role = item.get("role")
            content = item.get("content")

            if role in ["user", "assistant"] and isinstance(content, str):
                messages.append({
                    "role": role,
                    "content": content
                })

    # Add current question
    messages.append({
        "role": "user",
        "content": message
    })

    # Send conversation to Groq
    response = client.chat.completions.create(
        model=model_name,
        messages=messages
    )

    # Return AI response
    return response.choices[0].message.content


# 4. Create Gradio interface
demo = gr.ChatInterface(
    fn=neurobot,
    title="🤖 NeuroBot",
    description="Your AI Assistant | Powered by NeuroGen AI Academy"
)


# 5. Launch application
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
