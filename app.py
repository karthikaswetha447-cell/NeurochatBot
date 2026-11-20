import os
import re
import json
import math
import time
from datetime import date

import gradio as gr
from groq import Groq

# NEW: Imports for Memory and RAG
from memory_manager import MemoryStore, process_message_memory, approve_pending, CATEGORY_LABELS
from rag_engine import RAGEngine, format_rag_context

# ==========================================
# 1. INITIALIZATION
# ==========================================

client = Groq(api_key=os.environ["GROQ_API_KEY"])
model_name = "openai/gpt-oss-20b"

# Persistent Store & Engines
MEMORY_STORE = MemoryStore()
RAG_ENGINE = RAGEngine()

# Admin Settings
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
SHOW_SYSTEM_PROMPT = os.environ.get("SHOW_SYSTEM_PROMPT", "true").strip().lower() not in ("false", "0", "no")

# ==========================================
# 2. CORE CHAT LOGIC
# ==========================================

def assemble_prompt(message, history, user_id, rag_index, use_memory=True, use_rag=True):
    system_parts = [
        "You are NeuroBot, a helpful AI assistant for NeuroGen AI Academy.",
        "Grounding Rules: Use official context provided. Cite sources. Don't invent facts."
    ]

    # --- Memory Integration ---
    memory_text = ""
    if use_memory and user_id:
        mem_info = process_message_memory(MEMORY_STORE, user_id, message)
        memory_text = mem_info["memory_text"]
        if memory_text:
            system_parts.append(memory_text)

    # --- RAG Integration ---
    if use_rag and rag_index:
        hits = RAG_ENGINE.retrieve(rag_index, message)
        if hits:
            rag_context = format_rag_context(hits, rag_index["filename"])
            system_parts.append(f"UPLOADED DOCUMENT CONTEXT:\n{rag_context}")
        else:
            system_parts.append("A document is uploaded, but no relevant parts were found for this query.")

    messages = [{"role": "system", "content": "\n\n".join(system_parts)}]
    
    # History (Short-term memory)
    for turn in history[-10:]:
        messages.append(turn)
    
    messages.append({"role": "user", "content": message})
    return messages

def send_message(message, chat, user_state, rag_index, use_memory, use_rag):
    if not message.strip():
        return chat, "", user_state, gr.update(), "", []

    user_id = user_state.get("id") if user_state else None
    
    # 1. Process Memory Classification (Check for new facts)
    mem_info = None
    if use_memory and user_id:
        mem_info = process_message_memory(MEMORY_STORE, user_id, message)
    
    # 2. Build Prompt
    messages = assemble_prompt(message, chat, user_id, rag_index, use_memory, use_rag)
    
    # 3. Call LLM
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7,
            max_completion_tokens=1024
        )
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"Error calling AI: {str(e)}"

    new_chat = chat + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer}
    ]

    # Handle Consent for Memory
    consent_visible = False
    consent_html = ""
    pending_mems = []
    if mem_info and mem_info["pending"]:
        consent_visible = True
        pending_mems = mem_info["pending"]
        lines = [f"<li><b>{CATEGORY_LABELS[m['category']]}:</b> {m['value']}</li>" for m in pending_mems]
        consent_html = f"NeuroBot noticed new details. Save to memory?<ul>{''.join(lines)}</ul>"

    return (
        new_chat, 
        "", 
        user_state, 
        gr.update(visible=consent_visible), 
        consent_html, 
        pending_mems
    )

# ==========================================
# 3. AUTH & UTILS
# ==========================================

def handle_login(username, password):
    user, msg = MEMORY_STORE.login_user(username, password)
    if user:
        return user, f"✅ Logged in as {username}", gr.update(visible=False), gr.update(visible=True)
    return None, f"❌ {msg}", gr.update(visible=True), gr.update(visible=False)

def handle_register(username, password):
    user, msg = MEMORY_STORE.register_user(username, password)
    if user:
        return user, f"✅ Registered and logged in as {username}", gr.update(visible=False), gr.update(visible=True)
    return None, f"❌ {msg}", gr.update(visible=True), gr.update(visible=False)

def handle_logout():
    return None, "Logged out.", gr.update(visible=True), gr.update(visible=False)

def save_pending_memories(user_state, pending):
    if not user_state: return "Error: Not logged in.", gr.update()
    saved, errs = approve_pending(MEMORY_STORE, user_state["id"], pending)
    msg = f"Saved {len(saved)} items."
    if errs: msg += f" Errors: {len(errs)}"
    return msg, gr.update(visible=False)

def process_upload(file):
    if not file: return None, "No file."
    try:
        from pypdf import PdfReader
        import docx
        
        ext = file.name.split('.')[-1].lower()
        text = ""
        if ext == 'pdf':
            reader = PdfReader(file.name)
            text = "\n".join([p.extract_text() for p in reader.pages])
        elif ext == 'docx':
            doc = docx.Document(file.name)
            text = "\n".join([p.text for p in doc.paragraphs])
        else:
            return None, "Unsupported format."
        
        index = RAG_ENGINE.build_index(text, os.path.basename(file.name))
        return index, f"✅ Indexed {index['total_docs']} chunks from {index['filename']}"
    except Exception as e:
        return None, f"Error indexing: {str(e)}"

# ==========================================
# 4. GRADIO UI
# ==========================================

with gr.Blocks(title="NeuroBot | NeuroGen AI Academy") as demo:
    user_state = gr.State(None)
    rag_index = gr.State(None)
    pending_mems = gr.State([])

    gr.Markdown("# 🤖 NeuroBot Dashboard")
    
    with gr.Row():
        # --- Left: Chat Area ---
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(type="messages", height=500)
            
            with gr.Column(visible=False) as consent_area:
                consent_md = gr.HTML()
                with gr.Row():
                    save_btn = gr.Button("✅ Save Memories", variant="primary")
                    skip_btn = gr.Button("Skip")

            with gr.Row():
                msg = gr.Textbox(placeholder="Ask anything...", scale=4)
                send_btn = gr.Button("Send", variant="primary")

        # --- Right: Controls Area ---
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.Tab("🔑 Auth"):
                    with gr.Column() as login_view:
                        u_in = gr.Textbox(label="Username")
                        p_in = gr.Textbox(label="Password", type="password")
                        with gr.Row():
                            l_btn = gr.Button("Login")
                            r_btn = gr.Button("Register")
                    
                    with gr.Column(visible=False) as logout_view:
                        status_msg = gr.Markdown("Not logged in.")
                        lo_btn = gr.Button("Logout")

                with gr.Tab("🧠 Memory"):
                    use_mem = gr.Checkbox(label="Enable Long-term Memory", value=True)
                    mem_display = gr.Markdown("Login to see saved facts.")
                    refresh_mem = gr.Button("Refresh Memory List")

                with gr.Tab("📄 RAG"):
                    use_rag = gr.Checkbox(label="Enable RAG Context", value=True)
                    upload = gr.File(label="Upload PDF/DOCX", file_types=[".pdf", ".docx"])
                    rag_status = gr.Markdown("No document uploaded.")
                    clear_rag = gr.Button("Clear Document")

    # --- Events ---
    l_btn.click(handle_login, [u_in, p_in], [user_state, status_msg, login_view, logout_view])
    r_btn.click(handle_register, [u_in, p_in], [user_state, status_msg, login_view, logout_view])
    lo_btn.click(handle_logout, None, [user_state, status_msg, login_view, logout_view])
    
    send_btn.click(send_message, [msg, chatbot, user_state, rag_index, use_mem, use_rag], 
                  [chatbot, msg, user_state, consent_area, consent_md, pending_mems])
    
    save_btn.click(save_pending_memories, [user_state, pending_mems], [status_msg, consent_area])
    skip_btn.click(lambda: gr.update(visible=False), None, consent_area)
    
    upload.change(process_upload, upload, [rag_index, rag_status])
    clear_rag.click(lambda: (None, "Cleared."), None, [rag_index, rag_status])

    def show_mems(user):
        if not user: return "Please login."
        mems = MEMORY_STORE.get_memories(user["id"])
        if not mems: return "No facts saved yet."
        return "\n".join([f"- **{CATEGORY_LABELS[m['category']]}**: {m['value']}" for m in mems])
    
    refresh_mem.click(show_mems, user_state, mem_display)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 10000)))
