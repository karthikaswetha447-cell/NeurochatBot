NEUROBOT UPGRADE — GRADIO + GROQ
================================

This package updates the original NeuroBot app.py you shared. It keeps the
Gradio interface and Groq model call, and adds uploaded-file RAG retrieval.

FILES
-----
app.py                              Updated chatbot application
requirements.txt                    Runtime dependencies (Gradio 6-compatible)
course_info.template.json           Safe template for official academy/course data
PROFILE_CONTEXT_EXAMPLE.json        Example student profile fields
RAG_CONTEXT_FILTERING_EXPLAINED.txt  Detailed explanation of generic document filtering
.env.example                        Environment-variable names (no real secrets)
tests/test_features.py              Offline unit tests for core features
app_original_backup.py              Backup of the original code you supplied

FEATURES IN THIS VERSION
------------------------
1. User profile context
   The Context tab collects student name, skill level, interests and learning goal.
   These profile values are added to the prompt when Use profile context is on.

2. Conversation history
   The chatbot sends the latest 10 chat messages to the model when Use conversation
   history is on, allowing follow-up questions such as "explain it again".

3. Dynamic context + context filtering
   There are TWO separate context filters. The course-data filter selects only the
   relevant fee/certificate/duration fields from course_info.json when the student
   asks an academy/course question. Separately, the uploaded-document RAG filter
   searches ANY subject/document against the student's current question. It does
   not require course keywords and is not limited to academy documents.

4. Grounding from course_info.json and uploaded references
   Course facts are provided as official context. Missing or null values are
   explicitly treated as not specified. The model is instructed not to guess.
   Retrieved uploaded-document excerpts are passed to the answer evaluator as
   reference material. User uploads are not independently verified; the app does
   not browse external websites to validate them. Maintain academy-approved source
   documents and `course_info.json` carefully.

5. Uploaded-file RAG and question-driven filtering
   The Context tab accepts TXT, MD, PDF, DOCX, CSV and JSON files. Each long file is
   extracted and split into overlapping chunks. For EVERY new question, the app
   ranks the chunks against the current question and sends only the top relevant
   passages to the LLM, together with source filenames. For short follow-ups such
   as "give me an example" or "explain it again", the latest user question is
   added to the retrieval query so the topic is retained. Broad summary questions
   select passages spread across the document. DOCX paragraph and table text are
   both extracted.

6. Prompt engineering and response constraints
   The Prompt tab includes role-based instructions, zero-shot, one-shot, few-shot,
   instruction-based, constraint-based and reasoning-summary techniques. Instruction
   style controls include Standard, Simple explanation, Step-by-step instructions and
   Manual/custom instructions. Users can choose Auto language, English, Tamil or
   Tanglish and set a maximum response word count (0 means no limit). The app reports
   when the model exceeds the requested word count; word limits are prompt constraints
   and cannot be guaranteed perfectly by an LLM.
   The Output tab supports normal text, JSON, Markdown tables, bullets, numbered
   lists and a custom format description.

7. Prompt preview, evaluation and learning dashboard
   Preview shows the prompt/context assembled for a question without making a model
   API call. Context panels display profile, history, selected context, grounding,
   response constraints and retrieved RAG excerpts. Optional classification/evaluation
   features remain. Evaluation compares the answer with academy course data and any
   retrieved uploaded-file excerpts. Uploaded files are user-supplied references and
   are not independently verified by the application.

IMPORTANT IMPLEMENTATION NOTES / LIMITATIONS
---------------------------------------------
- Uploaded-file retrieval uses a lightweight BM25-style lexical relevance ranker.
  It is generic document filtering: all chunks are ranked against the current question,
  then only the best matching passages are sent to the LLM. It is NOT neural embedding
  search and does NOT use a persistent vector database such as Chroma, FAISS, Pinecone
  or Weaviate. This avoids a large embedding model and external vector service on the
  existing Render deployment.
- Search tokenisation is Unicode-aware and can match English, Tanglish and Tamil text.
  As with keyword/BM25 retrieval, synonyms and semantically related wording that shares
  few words with the document may be missed. For that, a multilingual embedding model
  and vector index would be a separate next-stage enhancement.
- The app shows the effective RAG retrieval query, matched terms, source and selected
  excerpts in Context > RAG so the user can inspect what was filtered into the prompt.
- Scanned/image-only PDFs are not OCR-processed. Use text-based PDFs or DOCX/TXT files.
- Uploads are parsed into the current Gradio session state; the uploaded document
  chunks are not persisted in a database. Re-upload documents after starting a new
  session or when the state is cleared.
- Student profile fields and chat history are session-scoped. This starter does not
  implement student login or permanent per-student profile storage. The latest 10
  messages are sent with the next prompt; older messages are excluded.
- The current question, selected profile fields, recent chat messages and relevant
  document excerpts are sent to the configured Groq model provider. Do not enter
  passwords, payment credentials or other sensitive personal information.
- Prompt instructions cannot guarantee perfect compliance. The LLM may still make
  mistakes, misread a document or fail to follow a requested output format. Verify
  important course, fee, admission and certificate answers against academy records.
- The user-uploaded documents are treated as reference data, not trusted instructions.
  This reduces, but cannot completely eliminate, prompt-injection risks in documents.

SET UP THE JSON COURSE FILE
---------------------------
1. If your existing GitHub repository already has course_info.json, KEEP THAT FILE.
   Do not replace real academy data with the template.
2. If there is no course_info.json, copy course_info.template.json and rename the copy
   to exactly course_info.json.
3. Replace null values with verified current official details. Leave a value as null
   when it is unknown. Do not guess prices, dates, duration or certificate rules.
4. Commit course_info.json in the same folder as app.py.

LOCAL INSTALLATION
------------------
Python: use Python 3.10–3.13.

Build/install command:
    pip install -r requirements.txt

Environment variable:
    GROQ_API_KEY=your_real_groq_api_key

Start command:
    python app.py

The app reads the PORT environment variable supplied by Render and binds to
0.0.0.0. Do not hard-code the Groq API key in app.py, JSON, or GitHub.

DEPLOY TO YOUR EXISTING RENDER SERVICE
--------------------------------------
1. Back up/download your current GitHub app.py, requirements.txt and course_info.json.
2. Replace app.py with this package's updated app.py.
3. Replace requirements.txt with this package's requirements.txt.
4. Keep your existing real course_info.json in the repository. If you do not have
   one, create it from course_info.template.json and fill verified values.
5. In Render > your existing NeuroBot service > Environment, confirm GROQ_API_KEY
   exists. Add it there if missing. Never paste the key into the code.
6. Confirm the service Start Command is:
       python app.py
   Build Command:
       pip install -r requirements.txt
7. Commit and push the changed files to your connected GitHub repository. Render
   should deploy the new commit automatically (or use Manual Deploy).
8. Open the service, fill the student profile in Context, upload one text-based file
   in Context > RAG, wait for the upload status, and ask a specific question about ANY
   topic in that file. Open Context > RAG to see the retrieval query, matched terms,
   source excerpts and the passages selected for that exact question. Then try a
   follow-up such as "give me an example" to test history-aware retrieval.

MODEL
-----
The original model name is retained: openai/gpt-oss-20b. If the provider reports that
this model is unavailable for your Groq account, change model_name near the top of
app.py to a model currently available in your Groq account and supported by the API.

ROLLBACK
--------
If deployment fails, restore the backup app.py and requirements.txt in GitHub and
redeploy. Keep your existing course_info.json unchanged.
