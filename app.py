import os
import json
import gradio as gr
import fitz

from groq import Groq
from sentence_transformers import SentenceTransformer, util


# ==========================================
# 1. GROQ API CONNECTION
# ==========================================

client = Groq(
    api_key=os.environ["GROQ_API_KEY"]
)

model_name = "openai/gpt-oss-20b"


# ==========================================
# 2. RAG MODEL AND PDF KNOWLEDGE BASE
# ==========================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

pdf_chunks = []
pdf_embeddings = None


# ==========================================
# 3. LOAD PDF AND EXTRACT TEXT
# ==========================================

def load_pdf():

    base_dir = os.path.dirname(os.path.abspath(__file__))

    pdf_path = os.path.join(
        base_dir,
        "documents",
        "syllabus.pdf"
    )

    pdf = fitz.open(pdf_path)

    all_pages = []

    for page_number, page in enumerate(pdf, start=1):

        page_text = page.get_text("text")

        if page_text.strip():
            all_pages.append({
                "page": page_number,
                "text": page_text
            })

    pdf.close()

    return all_pages


# ==========================================
# 4. SPLIT TEXT INTO CHUNKS
# ==========================================

def split_into_chunks(text, chunk_size=500, overlap=100):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ==========================================
# 5. CREATE PDF EMBEDDINGS
# ==========================================

def create_knowledge_base():

    global pdf_chunks
    global pdf_embeddings

    try:

        pages = load_pdf()

        for page_data in pages:

            page_number = page_data["page"]
            page_text = page_data["text"]

            chunks = split_into_chunks(page_text)

            for chunk in chunks:

                pdf_chunks.append({
                    "text": chunk,
                    "page": page_number
                })

        if not pdf_chunks:

            print("No readable text found in PDF.")
            return

        chunk_texts = [
            item["text"]
            for item in pdf_chunks
        ]

        print("Creating PDF embeddings...")

        pdf_embeddings = embedding_model.encode(
            chunk_texts,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        print("PDF knowledge base ready!")
        print("Total chunks:", len(pdf_chunks))

    except Exception as error:

        print("PDF Knowledge Base Error:", str(error))


# Load PDF when application starts
create_knowledge_base()


# ==========================================
# 6. RETRIEVE RELEVANT PDF INFORMATION
# ==========================================

def retrieve_relevant_chunks(message, top_k=3):

    if pdf_embeddings is None or not pdf_chunks:
        return None

    try:

        question_embedding = embedding_model.encode(
            message,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        similarity_scores = util.cos_sim(
            question_embedding,
            pdf_embeddings
        )[0]

        top_results = similarity_scores.topk(
            k=min(top_k, len(pdf_chunks))
        )

        retrieved_text = []

        for score, index in zip(
            top_results.values,
            top_results.indices
        ):

            similarity = score.item()

            if similarity < 0.25:
                continue

            chunk = pdf_chunks[index.item()]

            retrieved_text.append(
                f"Page {chunk['page']}:\n{chunk['text']}"
            )

        if not retrieved_text:
            return None

        return "\n\n".join(retrieved_text)

    except Exception as error:

        print("Retrieval Error:", str(error))
        return None


# ==========================================
# 7. EXTRACT TEXT FROM HISTORY
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
# 8. COURSE QUESTION DETECTION
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
# 9. RECENT USER MESSAGES
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
# 10. OFFICIAL COURSE INFORMATION
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
# 11. NEUROBOT CHAT FUNCTION
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
- For PDF-related questions, use the retrieved PDF
  information when it is relevant.
- Do not invent information from the PDF.
- If the answer is not available in the PDF,
  clearly say it is not found in the syllabus.
- For unrelated general questions, you may use
  general knowledge.

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
    # RAG: RETRIEVE PDF INFORMATION
    # ======================================

    retrieved_context = retrieve_relevant_chunks(message)

    if retrieved_context:

        rag_message = (
            "RETRIEVED INFORMATION FROM OFFICIAL SYLLABUS PDF:\n\n"
            + retrieved_context
            + "\n\nUse this retrieved information when relevant "
            "to answer the student's question. "
            "Do not invent facts that are not in this context. "
            "Mention the source as: Source: syllabus.pdf"
        )

        messages.append({
            "role": "system",
            "content": rag_message
        })


    # ======================================
    # CONVERSATION HISTORY
    # ======================================

    for item in history or []:

        if isinstance(item, dict):

            role = item.get("role")

            content = extract_text(
                item.get("content")
            )

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
# 12. GRADIO INTERFACE
# ==========================================

demo = gr.ChatInterface(
    fn=neurobot,
    title="🤖 NeuroBot",
    description="Your AI Assistant | Powered by NeuroGen AI Academy"
)


# ==========================================
# 13. START NEUROBOT ON RENDER
# ==========================================

if __name__ == "__main__":

    print("======================================")
    print("NEUROBOT STARTING")
    print("======================================")

    port = int(os.environ.get("PORT", 10000))

    print(f"Starting NeuroBot on port: {port}")

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
