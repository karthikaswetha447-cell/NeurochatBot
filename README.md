# NeuroBot name and memory fix

## Included files
- `app.py` — updated account identity context in the chat prompt and Memory trace.
- `memory_manager.py` — improved name phrase detection.

## What changed
1. Name capture now recognises:
   - `My name is Karthika`
   - `Hi, this is Karthika`
   - `I'm Karthika`
   - `I am Karthika`
   - `Call me Karthika`
2. Common descriptions such as `I'm a beginner`, `I am learning AI`, and `I am very happy` are not mistaken for names.
3. The logged-in username is passed to the model as an account identifier only. NeuroBot is instructed not to assume that it is the user's real or preferred name.
4. If a confirmed profile name or saved Name memory exists, that takes priority.
5. The Concepts text now correctly says long-term memory is stored in SQLite, not PostgreSQL.

## Apply using GitHub website (no terminal)
1. Open your NeuroBot GitHub repository.
2. Open `memory_manager.py` → Edit (pencil) → replace the full contents with the included `memory_manager.py` → Commit changes.
3. Open `app.py` → Edit (pencil) → replace the full contents with the included `app.py` → Commit changes.
4. Render should deploy automatically if GitHub auto-deploy is enabled.

## Test after deployment
1. Log in to NeuroBot.
2. In a new chat, send: `Hi, this is Karthika`.
3. Check the Memory panel. The name should appear as a suggestion if automatic saving is off; approve it.
4. Ask: `What's my name?`
5. NeuroBot should use the saved name. If no name has been approved yet, it should distinguish the account username from a confirmed preferred name.

## Important
These two files preserve the other code in the uploaded `app.py` and `memory_manager.py`. No changes were made to the multimodal engine, RAG implementation, prompt/parameter panels, experiments, or concepts layout. The current SQLite database and Render persistent-disk configuration are not changed.
