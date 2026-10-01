# frontend

Monitoring UI for the running smart-glasses process. This folder is only `server.py` (aiohttp) and `index.html` (one page, CSS and JS inline). `main.py` calls `start_monitoring_server(log_queue, db_path)` on a daemon thread.

## Address

`MONITORING_HOST` and `MONITORING_PORT` live in `core/config.py`. Port is **8765**. Dashboard: `http://localhost:8765`. Read the config constants; do not hardcode the port.

## What the page shows

- **Coordinator log** — sources `[Coordinator]` and `[System]`.
- **Workers log** — `[AudioWorker]`, `[Vision]`, `[API Worker]`.
- **Database status** — identities (face/voice flags), last 50 chat rows, embedding and chat counts.
- **Command console** — rename or delete a user, clear chat, purge the DB, query history, upload an image for face lookup.

Logs arrive on WebSocket `/ws` (`type: "log"`). DB state arrives as `type: "db_update"` (initial push, every 5s, and when a log line matches `_DB_CHANGE_KEYWORDS`). The client does not send WebSocket messages.

## HTTP

| Method | Path | Effect |
|---|---|---|
| GET | `/` | `index.html` |
| GET | `/ws` | log and DB stream |
| GET | `/api/db` | snapshot JSON |
| PUT | `/api/users/{user_id}` | rename (`{"name": ...}`) |
| DELETE | `/api/users/{user_id}` | delete user |
| GET | `/api/users/{user_id}/history` | that user's chat |
| DELETE | `/api/chat` | clear `chat_history` |
| DELETE | `/api/db` | `DatabaseManager.clear_db()` |
| POST | `/api/face_lookup` | base64 image vs stored face embeddings |

Reads hit sqlite at `IDENTITY_DB_PATH`. Writes go through `database.database.DatabaseManager`, then broadcast a fresh snapshot. Face lookup lazy-starts InspireFace; if that fails, the handler returns an error.

## Editing

- Keep UI in `index.html`. No build step, framework, or package manifest in this folder.
- Keep `log` / `db_update` field names the page already reads: `source`, `text`, `ts`, `users`, `chat_history`, `face_count`, `voice_count`, `chat_count`.
- A new log `source` stays invisible until `handleLog` in `index.html` lists it.
- After a mutation, broadcast a snapshot so open dashboards update.
