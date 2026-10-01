# Smart glasses

UDP video and audio come in. Vision and audio workers process them. The coordinator turns those events into identity updates and LLM commands. The LLM is OpenAI. Identities live in sqlite.

`main.py` builds `SharedMem`, starts the workers, and runs UDP ingest on a thread. On shutdown it stops UDP, clears sample identities when `START_WITH_SAMPLE_DATA` is set, then joins the workers.

## Layout

- `main.py` — process entry and shutdown
- `core/` — config, queues, logging. Plumbing only.
- `api/` — UDP ingest, OpenAI client, simulator
- `database/` — sqlite identities
- `frontend/` — monitoring dashboard
- `workers/` — one process per worker; step sequence in the entry file, details in the helper package
- `tests/` — unit tests
- `firmware/` — device code. Separate from this Python server. Not run the same way. Treat it as its own repo.

Worker entry points: `workers/audio.py`, `workers/vision.py`, `workers/coordinator.py`, `workers/api_worker.py`. Base class: `workers/base.py`.

Helper packages: `workers/audio_utils/`, `workers/vision_utils/`, `workers/coordinator_utils/`, `workers/api_utils/`.

## Where to look

- STT (Parakeet): `workers/audio.py`, `workers/audio_utils/utterance.py`
- VAD (Silero): `workers/audio_utils/vad.py`
- Speaker ID: `workers/audio_utils/speaker.py`
- Face tracking: `workers/vision.py`, `workers/vision_utils/tracks.py`, `workers/vision_utils/inspireface_processor.py`, `workers/vision_utils/matching.py`
- Identity registration: `workers/coordinator_utils/handlers.py`, `database/database.py`
- LLM intent: `workers/api_worker.py`, `workers/api_utils/commands.py`, `api/openai_client.py`
- Memory: `api/openai_client.py` (`analyze_memory`), `workers/api_utils/commands.py`
- Simulator: `api/simulator.py`
- Dashboard: `frontend/server.py`, `frontend/index.html`
- Queues: `core/shared_mem.py`, [workers/event_definitions.md](workers/event_definitions.md)
- Config: `core/config.py`, `core/config_audio.py`, `core/config_vision.py`, `core/config_openai.py`
- Tests: `tests/`

## Flow

`api/udp_receiver.py` splits packets by `HEADER_VISION` / `HEADER_AUDIO` into `vision_queue` and `audio_queue`.

Audio loads Silero, Redimnet, and Parakeet in `setup()`, then emits `speech` on `results_queue`. Vision loads InspireFace in `setup()`, tracks faces, and emits `vision_result`. The coordinator (`workers/coordinator_utils/handlers.py`) registers identities and enqueues commands. `APIWorker` runs one OpenAI call at a time and puts `intent` or `memory_result` back on `results_queue`.

The dashboard (`frontend/server.py`, port `MONITORING_PORT`) drains `log_queue` and reads the identity db.

## Queues

Owned by `core/shared_mem.py`:

- `vision_queue`, `audio_queue` — raw UDP payloads
- `results_queue` — vision, speech, VLM, intent, memory, api errors
- `vision_command_queue`, `audio_command_queue`, `llm_command_queue` — coordinator to workers
- `log_queue` — required on every worker

Event shapes: [workers/event_definitions.md](workers/event_definitions.md).

## Folder docs

Read the folder doc before editing that area:

- [api/AGENTS.md](api/AGENTS.md)
- [core/AGENTS.md](core/AGENTS.md)
- [database/AGENTS.md](database/AGENTS.md)
- [frontend/AGENTS.md](frontend/AGENTS.md)
- [workers/AGENTS.md](workers/AGENTS.md)
- [workers/audio_utils/AGENTS.md](workers/audio_utils/AGENTS.md)
- [workers/vision_utils/AGENTS.md](workers/vision_utils/AGENTS.md)
- [workers/coordinator_utils/AGENTS.md](workers/coordinator_utils/AGENTS.md)
- [workers/api_utils/AGENTS.md](workers/api_utils/AGENTS.md)
- [tests/AGENTS.md](tests/AGENTS.md)

## Rules

Code is the source of truth. These docs lag. Read the code before acting on a doc. If you change behavior, update this file and the folder `AGENTS.md` that covers it.

Python 3.11 venv. Run with `PYTHONPATH=. .venv/bin/python`. Use that interpreter, not system python.

Workers subclass `multiprocessing.Process`. `__init__` runs in the parent. `setup()` and `run()` run in the child. Do not construct InspireFace, ONNX, Parakeet, asyncio loops, or aiohttp sessions in `__init__`.

`core/` is plumbing only (config, queues, logging). Domain rules stay in worker helpers.

`firmware/` is pretty separate from the rest. It is not run in the same format as this backend. Consider it a separate repo from the server.

A worker file reads as the sequence of steps. Details live in the helper package. Pass owned stateful objects into those functions.

Comments, when needed, are one nonchalant line above a branch, not docstrings.

`T | None` only when absence is a real state. `log_queue` is required. Type function params and returns.

LLM is OpenAI only. The queue name is `llm_command_queue`. Do not reintroduce Gemini. Env: `OPENAI_API_KEY` (`core/config_openai.py`).

Live identity db is `./database/identities.db` (`IDENTITY_DB_PATH`). `START_WITH_SAMPLE_DATA` loads sample embeddings at startup and calls `clear_db()` on shutdown. Tests must use a temp db. Do not wipe the live db.

Unit tests, and only these unless asked:

```
PYTHONPATH=. .venv/bin/python tests/test_worker_helpers.py
PYTHONPATH=. .venv/bin/python tests/test_database.py
PYTHONPATH=. .venv/bin/python tests/test_api_worker_queue.py
```

Do not run `tests/test_openai_tool.py`, `tests/test_openai_memory.py`, `main.py`, or `api/simulator.py` unless asked.
