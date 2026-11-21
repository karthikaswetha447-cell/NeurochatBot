# NeuroBot — NeuroGen AI Academy

Existing Gradio + Groq chatbot project, upgraded additively.

## Existing functionality
The original dashboard, prompt engineering controls, LLM parameter experiments, user profile/context controls, PDF/DOCX and other text-file RAG workflow, trusted academy grounding, memory consent UI, evaluation tools, experiments and concepts dashboard remain in `app.py`.

## Added in this upgrade
- SQLite-only user accounts and long-term memory in `memory_manager.py`.
- Image understanding through Groq vision model.
- Audio transcription through Groq Whisper.
- Microphone upload support through Gradio Audio.
- English text-to-speech through Groq Orpheus.
- Video sampled-frame analysis and optional audio extraction/transcription.
- New Multimodal tab in the existing Gradio dashboard.
- Media analysis can be sent into the original chat so the existing memory, RAG and context flow can process the follow-up.

## Start / deploy
Render reads `requirements.txt` and runs the existing `app.py` web service. Set `GROQ_API_KEY` and configure the persistent SQLite disk as described in `RENDER_SETUP.md`.

## Important
The app uses Gradio 5 (`gradio>=5,<6`). The original text model remains `openai/gpt-oss-20b`. Vision and audio use separate modality-specific models; see `RENDER_SETUP.md`.
