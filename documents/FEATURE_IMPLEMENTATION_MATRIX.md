| Requested feature | Implemented behavior | Scope / limitation |
|---|---|---|
| User Profile Context | Name, learning level, interests and learning goal are added to the prompt when enabled. | Session-scoped UI fields; no student login or permanent profile database. |
| Conversation History | Sends the latest 10 messages (user + assistant) when enabled. | Older messages are excluded from the model request. |
| Dynamic Context | Loads academy course JSON when a course-related query or recent context indicates it is relevant; uploaded-file passages are retrieved on each question. | Course field selection currently uses academy topic terms. |
| Context Filtering | Ranks all uploaded-document chunks against the current question, and enriches short follow-ups with the last user question. | BM25-style lexical matching, not neural semantic search. |
| Grounding | Uses selected `course_info.json` values and retrieved document excerpts; includes source filename and chunk references in model context. | Uploaded files are user-supplied and not independently verified against the public web. |
| File Upload & RAG | Accepts TXT, MD, PDF, DOCX, CSV and JSON; extracts text, chunks it, retrieves the top 4 relevant passages. | Per-file size/text limits are in `README_DEPLOY.txt`; scanned PDFs are not OCR-processed; uploads are session-only. |
| Role-Based Prompting | Default NeuroBot, Friendly Teacher, Career Mentor, Technical Interviewer, Code Reviewer and Student Support Agent roles. | Roles guide style; they do not retrain the model or guarantee behavior. |
| Prompt Techniques | Zero-shot, One-shot, Few-shot, instruction-based, constraint-based, reasoning summary and step-by-step plan. | Output may vary between model calls. |
| Instruction-Based Prompting | Select Standard, Simple explanation, Step-by-step instructions or Manual/custom instructions. | Custom directions must remain consistent with privacy and grounding rules. |
| Constraint-Based Prompting | Select Auto, English, Tamil or Tanglish; select a maximum response length from 0–500 words. | Word count is a prompt constraint; the app detects and reports an over-limit answer but does not truncate structured output. |
| Structured Output | Normal text, JSON, Markdown table, bullet list, numbered list and custom format. JSON syntax is validated after generation. | Model formatting is not guaranteed; validation status is shown. |
| Evaluation | AI-estimated relevance, accuracy, completeness and faithfulness scores. Evaluation receives official academy data and retrieved passages where available. | Scores are estimates, not independent audits. User uploads are not treated as verified sources. |
