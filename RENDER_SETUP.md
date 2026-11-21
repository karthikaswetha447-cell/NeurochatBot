# NeuroBot GitHub + Render deployment notes

## Files changed
- `app.py`: retains the existing dashboard, tabs, experiments, RAG and prompt workflows; adds a Multimodal tab.
- `memory_manager.py`: SQLite-only account and long-term memory store, with an adapter matching the memory API used by this app.
- `multimodal_engine.py`: Groq image understanding, audio transcription, TTS and sampled video-frame/audio processing.
- `requirements.txt`: Gradio 5 and the required packages.

## Render environment variables
Keep the existing `GROQ_API_KEY` secret. Add or verify:

| Key | Value |
|---|---|
| `MEMORY_SQLITE_PATH` | `/var/data/neurobot_memory.db` |
| `GROQ_VISION_MODEL` | `qwen/qwen3.8-27b` |
| `GROQ_TRANSCRIPTION_MODEL` | `whisper-large-v3-turbo` |
| `GROQ_TTS_MODEL` | `canopylabs/orpheus-v1-english` |
| `GROQ_TTS_VOICE` | `troy` |
| `MULTIMODAL_MAX_UPLOAD_MB` | `20` |
| `VIDEO_MAX_FRAMES` | `6` |

Do not set `DATABASE_URL`; the updated memory manager intentionally ignores PostgreSQL.

## Render persistent disk
1. Open the Render Web Service for NeuroBot.
2. Open Disks and add a persistent disk.
3. Mount path: `/var/data`.
4. Choose a size suitable for your account and expected memory records.
5. Set `MEMORY_SQLITE_PATH=/var/data/neurobot_memory.db`.
6. Save and redeploy.

Without a mounted persistent disk, a SQLite file on the service filesystem can be lost when Render replaces the instance. Do not delete or rename an existing SQLite database file during deployment.

## Multimodal behavior and limits
- Text chat continues to use the existing `openai/gpt-oss-20b` model.
- Image analysis uses Groq's documented vision model `qwen/qwen3.8-27b`.
- Audio transcription uses `whisper-large-v3-turbo`.
- TTS uses `canopylabs/orpheus-v1-english` with voice `troy` (English output).
- Video analysis samples up to six frames and attempts audio extraction/transcription using `imageio-ffmpeg`. It does not understand every frame continuously.
- Media is not written into long-term memory automatically. The user can send the analysis into the normal chat, where existing memory/RAG/context processing applies.
- API availability, account model access, rate limits and provider quotas still apply.

## GitHub-only replacement
Upload or replace the four files listed above in the repository using GitHub's Add file / Upload files or Edit pencil workflow. Commit the changes. Render should deploy from the connected branch automatically.
