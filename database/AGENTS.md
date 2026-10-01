# database

Sqlite identities. `DatabaseManager` in `database.py` is the only API. Live file is `./database/identities.db` (`IDENTITY_DB_PATH` in `core/config.py`).

## Tables

- `users` — `id` (text, usually a uuid), `name`
- `face_embeddings` — many per user, ndarray stored as `ARRAY`
- `voice_embeddings` — many per user, same storage
- `chat_history` — `transcript` plus `timestamp`, newest first on read

Foreign keys cascade on user delete. Embeddings use a numpy adapter (`sqlite3.Binary` in, `np.frombuffer` out).

## Calls that matter

- `create_user`, `update_user`, `delete_user`, `get_user_name`, `get_all_users`, `get_user_ids_by_name`
- `save_face_embedding`, `get_all_faces` — vision loads known faces from here
- `save_voice_embedding`, `get_all_voices`, `get_voice_embeddings_by_uid` — audio loads speakers; coordinator compares a self-intro
- `save_chat_history`, `get_chat_history`, `get_recent_chat_history`
- `clear_db` — drops every row

Create the user before saving an embedding. The face and voice tables reference `users.id`.

## Sample data and shutdown

`sample_data/` holds `.npy` voice embeddings (and face images used offline). `START_WITH_SAMPLE_DATA` in `core/config.py` loads `SAMPLE_VOICE_EMBEDDING_PATHS` at startup. On shutdown, `main.py` calls `clear_db()` and wipes the live db.

## Tests

`tests/test_database.py` opens a temp file and deletes it. Never point a test at `./database/identities.db`.
