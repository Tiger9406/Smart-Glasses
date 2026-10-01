# Tests

From the repo root, run one module:

```
PYTHONPATH=. .venv/bin/python tests/<file>.py
```

## Offline

Safe without models or API keys. Do not load Silero, Redimnet, Parakeet, or InspireFace. Import helpers under `workers/*_utils`, not `workers.audio` or `workers.vision` (those pull in the models).

- `tests/test_worker_helpers.py` — speaker cosine match and merge; face-track update and expiry; best identity; voice and face association; intent and memory events; coordinator speech and registration; PCM, chunks, utterances, voice commands, sampled frames, tracked-face registration.
- `tests/test_database.py` — `DatabaseManager` users, face and voice embeddings, chat history. Opens a temp `identities.db` and deletes it (including `-shm` and `-wal`). Never use `./database/identities.db`.
- `tests/test_api_worker_queue.py` — `APIWorker` with fake clients: PARSE_INTENT and ANALYZE_MEMORY events, empty and unknown commands, `api_error` when the client raises.

## Live OpenAI

Do not run unless asked. Each needs `OPENAI_API_KEY`.

- `tests/test_openai_tool.py` — `OpenAIClient` memory, video frames from `api/simulator_resources/LongVideo.mp4`, and `parse_intent` (self-intro and third party).
- `tests/test_openai_memory.py` — `analyze_memory` extraction and dedup against known facts.
- `tests/test_api_worker_live.py` — real `APIWorker` process; PARSE_INTENT for "Hi, my name is John."; expects `intent` or `api_error`.
