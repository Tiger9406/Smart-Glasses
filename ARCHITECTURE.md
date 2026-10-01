# Architecture

Diagrams for this document are inline SVG in [docs/architecture.html](docs/architecture.html). Each section below includes a short prose version so this file stands alone.

This is a host-side pipeline for smart glasses. A device, or `python -m api.simulator`, sends UDP datagrams. Separate processes turn video into face tracks and audio into a transcript plus a speaker id. The coordinator decides who that was and asks OpenAI what the line meant. Identities sit in sqlite. A dashboard on port 8765 shows logs and that database. The firmware, when present, is the device. This Python process is the host.

## Runtime

`main.py` builds `SharedMem`, installs a log interceptor on the parent, and starts the monitoring thread. It opens `DatabaseManager` and, when `START_WITH_SAMPLE_DATA` is set, loads sample embeddings before any worker starts. It then starts four `multiprocessing.Process` workers and a UDP thread, and sleeps until Ctrl+C.

| Piece | Owns |
| --- | --- |
| UDP thread (`api/udp_receiver.py`) | The socket on `0.0.0.0:8000`. Splits datagrams into `vision_queue` and `audio_queue`. No models, no identities. |
| `VisionWorker` | InspireFace session, live face tracks, JPEG frame buffer, optional VLM client. Writes `vision_result`. |
| `AudioWorker` | Silero, Redimnet, Parakeet, known speaker profiles. Writes `speech`. |
| `Coordinator` | Identity decisions, vision and audio caches, conversation deque, pending voice registration. Writes command queues. |
| `APIWorker` | One OpenAI call at a time. Writes `intent`, `memory_result`, or `api_error` back to `results_queue`. |
| sqlite | `users`, `face_embeddings`, `voice_embeddings`, `chat_history` at `./database/identities.db`. |
| Dashboard thread | Drains `log_queue` and reads the same db file on its own connections. |

Workers subclass `multiprocessing.Process` (`workers/base.py`). `__init__` runs in the parent and only stores queues. `start()` launches the child. `setup()` and `run()` run in the child. The code never calls `set_start_method`. On macOS the default is `spawn`: the child is a new interpreter and receives a pickled copy of the process object. InspireFace, ONNX, and Parakeet are constructed in `setup()` so they exist in that child. `APIWorker` builds `OpenAIClient` in `__init__` (key, URLs, JSON only). The asyncio loop is created in `run()`, and the aiohttp session opens on the first request inside the child.

Shutdown stops the UDP thread, and if `START_WITH_SAMPLE_DATA` is set calls `clear_db()` while the workers are still alive. It then clears each worker's `running` event, closes the queues without draining them, `join`s each worker for 1 second, and `terminate`s any that are still alive. If both annotated video and recorded wav exist, it muxes them with ffmpeg.

**System context.** Glasses or the simulator send UDP to the host. The host is the only box that talks to OpenAI and the only box that writes identities. sqlite is a file beside the pipeline, not a service. The dashboard is a thread in the same OS process as `main`, reading logs off a queue and reading sqlite on its own. OpenAI is not on the path from the device to the workers; it is reached only after the coordinator has a transcript.

## Queues and events

`SharedMem` (`core/shared_mem.py`) creates seven `multiprocessing.Queue`s. Payload fields are listed in [workers/event_definitions.md](workers/event_definitions.md). Where that file and a worker disagree, the worker is the one on the wire: a `speech` event carries `user_id` and `time_start`, and a `vision_result` has `faces` with `user_id` and no timestamp of its own.

| Queue | Direction |
| --- | --- |
| `vision_queue` (max 100) | UDP thread writes a JPEG payload. `VisionWorker` reads it. If the queue is full, the oldest datagram is dropped. |
| `audio_queue` (max 100) | UDP thread writes PCM payload bytes. `AudioWorker` reads them. Same drop-oldest rule. |
| `results_queue` | `VisionWorker` writes `vision_result` (and `vlm_result` if a VLM call runs). `AudioWorker` writes `speech`. `APIWorker` writes `intent`, `memory_result`, or `api_error`. `Coordinator.run` is the only reader, and it calls `handle_event`. |
| `vision_command_queue` | Coordinator writes `REGISTER_FACE` and `GET_VIDEO_CONTEXT`. `VisionWorker` reads them at the start of each loop. |
| `audio_command_queue` | Coordinator writes `REGISTER_VOICE`. `AudioWorker` reads them at the start of each loop. |
| `llm_command_queue` | Coordinator writes `PARSE_INTENT`. `APIWorker` also accepts `ANALYZE_MEMORY`. Nothing in the coordinator enqueues `ANALYZE_MEMORY` on the live path. |
| `log_queue` (max 2000) | Each process's `install_log_interceptor` writes `{source, text, ts}` from `print`. The dashboard thread reads it. A full queue drops the line. |

**Process and queue map.** One UDP thread fans out by header byte: JPEG on `vision_queue`, PCM on `audio_queue`. Vision and audio never read each other's ingress queues. Both, plus the API worker, write `results_queue`, and only the coordinator reads it. Commands go back on dedicated queues: `PARSE_INTENT` on `llm_command_queue`, `REGISTER_FACE` and `GET_VIDEO_CONTEXT` on `vision_command_queue`, `REGISTER_VOICE` on `audio_command_queue`. The API worker's reply (`intent`, `api_error`, or `memory_result`) returns on `results_queue`. `log_queue` is a side channel from every process to the dashboard, not an input to recognition.

## Ingress

The host accepts datagrams. It does not capture the camera or the microphone.

`udp_receiver_thread` binds `config.HOST`:`config.PORT` (`0.0.0.0:8000`), receives up to 65507 bytes, and splits `data[0:1]` from `data[1:]`. `HEADER_VISION` (`b"\x01"`) puts the remainder on `vision_queue`. `HEADER_AUDIO` (`b"\x02"`) puts it on `audio_queue`. Any other first byte is ignored. The thread is a daemon started from `main.py`. `udp_shutdown_event` closes the socket.

The simulator is `python -m api.simulator`. It reads `api/simulator_resources/simulator_config.json` (`header_vision` 1, `header_audio` 2, port 8000) and sends JPEG frames and PCM chunks to `127.0.0.1` when the configured host is `0.0.0.0`. It is a stand-in for the glasses, not a worker.

This checkout does not contain `hardware/`. The device firmware imported on `task/import-hardware-repo` is ESP32-S3 PlatformIO (`hardware/platformio.ini`, `esp32-s3-devkitc-1`). `hardware/src/main.cpp` starts a camera task, an I2S audio task, and `udp_send_task`. The camera task pushes JPEG frames (`esp_camera`, `PIXFORMAT_JPEG`). The audio task reads 16 kHz I2S and queues `int16` samples. `hardware/src/udp.cpp` sends one header byte plus payload: `WS_FRAME_VIDEO` (`0x01`) or `WS_FRAME_AUDIO_TX` (`0x02`) from `hardware/include/config.h`. Those values match `HEADER_VISION` and `HEADER_AUDIO`. The Python host never reads the firmware headers. README also points at a separate hardware repo. The two sides share a one-byte convention and nothing else.

## Audio

`AudioWorker.run` turns a PCM byte stream into `speech` events. A sentence is an utterance: open on speech, extend while speech continues, close after enough silence, then transcribe and identify the speaker.

`setup()` loads Redimnet (`REDIMNET_PATH`), Silero (`VAD_PATH`, threshold 0.5, window 512, context 64), and Parakeet (`PARAKEET_MODEL`). It loads `known_speakers` as the mean embedding per user from `get_all_voices`. Chunk size is `AUDIO_CHUNK_SIZE_MS` (160) at `AUDIO_SAMPLE_RATE_HZ` (16000), so `chunk_samples` is 2560 and `chunk_bytes` is 5120. `padding_chunks` is `int(320 / chunk_ms)` (2). `SILENT_CHUNK_THRESHOLD` is 1. `SIMILARITY_THRESHOLD` is 0.30.

Each loop calls `apply_voice_commands` (only `REGISTER_VOICE`, which `save_voice_embedding`s and `merge_speaker`s into `known_speakers`), then `next_audio`. `record_chunk` writes a wav when `SAVE_ANNOTATED_VID` is set. `take_chunks` slices the byte buffer. `pcm_from_bytes` scales int16 to float.

`hears_speech` appends samples and runs every full 512-sample window. Any window probability above 0.5 makes the chunk count as speech. If speech arrives with no open utterance, `begin_utterance` copies `pre_speech`, clears it, and opens `transcribe_stream((CONTEXT_LEFT, CONTEXT_RIGHT))`. Further speech calls `extend_utterance`, which zeroes `silence_count`. Silence with no utterance appends to `pre_speech`. Silence inside an utterance calls `note_silence` (increment count, keep the samples). When `silence_count` reaches `silent_chunks`, the worker calls `reset_vad` and `finish_utterance`: concatenate samples, Parakeet `add_audio`, `voice_embedding` on Redimnet, `identify_speaker`. `speech_event` is queued only when the transcript is non-empty. `last_speaker` still updates. The event is `{type: speech, text, id, time_start, timestamp, final: True, embedding, user_id}`.

`identify_speaker` starts at `DEFAULT_ID` (`"unknown"`) with score equal to the threshold. If `last_speaker` is a known id and its cosine similarity beats the threshold, that score becomes the bar. Any other profile must score strictly higher. Cosine similarity is dot product over norms.

**Utterance states.** Silence with no utterance only fills `pre_speech`. The first `hears_speech` calls `begin_utterance` and enters the utterance, carrying that pad. More speech calls `extend_utterance`. A quiet chunk calls `note_silence`. One quiet chunk is enough (`SILENT_CHUNK_THRESHOLD` is 1), which calls `reset_vad` and `finish_utterance` and returns to silence. End of sentence is that finish, not a second model.

## Vision

`VisionWorker.run` turns JPEG bytes into face tracks and publishes them. Recognition is InspireFace. Tracking state is a dict of `ActiveIdentity` keyed by `track_id`, not a conversation.

`setup()` builds `InspireFaceProcessor` (launch Megatron or Pikachu, session in track-by-detection, recognition on), sets track-lost recovery, and creates `active_identities`, a frame deque of `FPS * BUFFER_DURATION` (10 × 5), an `OpenAIClient`, and a background asyncio loop for the VLM. `recheck_interval` is 2.0 seconds, `confidence_threshold` is 0.5, `lost_track_threshold` is 1.0 second. Known faces load from sqlite inside the processor.

Each loop drains `apply_vision_commands` before taking a frame. `GET_VIDEO_CONTEXT` runs only when `VLM_ACTIVE` is true (it is false). Inactive, that command prints and returns out of the drain. `REGISTER_FACE` calls `register_tracked_face`: `register_identity` appends the embedding in memory and `save_face_embedding`, then renames the live track. `next_frame` waits 0.01 seconds. The raw JPEG is appended to `frame_buffer` with `time.time()`. `decode_frame` is `cv2.imdecode`. `recognize_frame` calls `detect_faces`, and for each face that needs it `extract_embedding` plus `identify_embedding`. `identify_embedding` uses `best_identity`: the highest `feature_comparison` score above `CONFIDENCE_THRESHOLD_MATCHING` (0.5), else `DEFAULT_ID`. `update_tracks` applies that result. `publish_faces` puts `{type: vision_result, faces: [...]}` with `block=False`. A face dict is `track_id`, `bbox`, `user_id`, `name`, `score`, `emb`. `emb` is set only on a frame that recomputed it. `publish_faces` does not attach a timestamp.

`update_tracks`: a new `track_id` starts as `DEFAULT_ID` / `DEFAULT_NAME` (`"unknown"` / `"Unknown"`). The worker recognizes when the track is still unknown or when `now - checked_ts` exceeds `recheck_interval`. A score above `confidence_threshold` stores `user_id`, `name`, `score`, and `checked_ts`. Otherwise the track is set back to unknown. Between those checks the previous identity is copied onto the result and `emb` stays `None`. Tracks with `now - last_seen` greater than `lost_after` are deleted.

If `SAVE_ANNOTATED_VID` is set, frames are archived during the loop. After the loop, `render_annotated_video` writes `VIDEO_OUTPUT_PATH` with boxes and `name (ID: track_id)`. That file is produced after the stream stops. `main.py` muxes it with the wav when both exist.

**One face track.** A new id is unknown. The next recognition either keeps a name (score above 0.5) or leaves it unknown. Later frames reuse that decision until 2 seconds pass, then `update_tracks` recognizes again. If the id is missing for more than 1 second it is deleted. Nothing in this path reads the transcript.

## Coordinator

`Coordinator.run` blocks on `results_queue` and calls `handle_event`. `setup()` builds `CoordinatorState`: the db, three command queues, a vision cache (10 seconds), an audio cache (30 seconds), `conversation_history` (last 10 lines), `pending_voice_registration`, and `max_delay` of 20 seconds. `VOICE_TRAP_LIMIT` in `core/config.py` is not read.

`handle_vision_result` appends `(time.time(), faces)` and `trim_before`s the vision cache. The cache clock is when the coordinator received the event, not a capture time from vision.

`handle_speech` reads `user_id` (default `DEFAULT_ID`), `text`, `time_start`, and `embedding`. An unknown id with an embedding is cached (`remember_unknown_voice`), then `bind_pending_voice` or `associate_unknown_voice` may replace the id. `record_transcript` always runs: the line is appended to `conversation_history`, and `save_chat_history` runs only for a known user with non-empty text. `intent_prompt` adds `names_in_view` (known `user_id`s on the latest cached frame) and the history, and asks for the latest message from that speaker. The coordinator enqueues `PARSE_INTENT` with that text, the speech `time_start`, and the voice embedding. Known speakers skip the unknown-voice branch and still get history plus `PARSE_INTENT`.

`bind_pending_voice` calls `pending_registration_hit`. The speech time must fall in `[0, max_delay)` after the pending timestamp. A hit enqueues `REGISTER_VOICE` and clears the pending record. `associate_unknown_voice` calls `associate_voice` on the vision cache with a 1.0 second window around `time_start`. A known face with `is_speaking` on at least 30% of frames in that window wins. Live `vision_result` faces do not include `is_speaking`, so that count stays zero. If nobody is marked speaking and exactly one known person is present on at least 80% of those frames, that id is used. A hit also enqueues `REGISTER_VOICE`. Otherwise the speaker stays `unknown`.

`handle_intent` switches on `cmd`. `REGISTER_IDENTITY` calls `register_identity`. `SPEAK` prints the message (`[STEVE]`) and does not enqueue audio. `VISION_CONTEXT` enqueues `GET_VIDEO_CONTEXT` with `request_id` taken from `request_number` (it stays 0). `handle_vlm_result` and `handle_api_error` print. `memory_result` is not in the handler map, so it hits `handle_unknown_event`.

`register_identity` looks up `get_user_ids_by_name`. If the name exists, `bind_existing_identity` checks each id: `face_matches` on `closest_frame` (the cached frame nearest the intent timestamp; the uid is in that frame's faces) or `voice_matches` (self-intro only, cosine of the utterance embedding against the mean stored voice, threshold 0.30). The first id that matches is kept. A match registers the face when the face was not already that user (`register_unknown_face` → `REGISTER_FACE` for the unknown face with the highest `0.3 * area_scaled - 0.7 * distance_scaled` on that frame). A self-intro whose voice is not yet stored enqueues `REGISTER_VOICE`. When that does not run and the speaker is `USER_NAME` (`"Tiger"`), `attempt_voice_registration` runs: `resolve_unknown_voice` takes the first cached unknown embedding whose time is in `[0, max_delay]` after the intent, or sets `pending_voice_registration`. If no existing id matches and the speaker is `USER_NAME`, pending is set on `existing_user_ids[0]` and the function returns. Any other speaker only logs that the name was assumed.

If the name is new, `create_identity` makes a uuid, `create_user`, and `register_unknown_face`. Voice registration depends on who spoke. `speaker_name == DEFAULT_NAME` (`"Unknown"`) and `is_self_introduction` enqueues `REGISTER_VOICE` for the embedding on the intent. `speaker_name == USER_NAME` calls `attempt_voice_registration` (cached unknown voice, or the pending trap for the next unknown utterance).

Tool names in `core/config_tools_openai.json` are `register_identity`, `speak`, and `vision_context`. `OpenAIClient.parse_intent` uppercases the tool name, so the coordinator sees `REGISTER_IDENTITY`, `SPEAK`, and `VISION_CONTEXT`. Plain text comes back as `CHAT`, which `handle_intent` does not branch on.

**Speech decision.** Unknown embedding: try the pending trap, then visual association, otherwise leave the id unknown. Known or unknown, the next two steps always run: `record_transcript`, then `PARSE_INTENT`.

**Registration.** Name already in `users`: require a face on the closest frame or a self-intro voice above 0.30, then fill whichever of face or voice is missing. No match and the wearer is `USER_NAME`: set the pending trap on the first stored id. New name: create the user and try to register an unknown face. The voice is registered immediately only for an Unknown self-intro. A wearer naming someone uses a recent unknown voice or the same pending trap.

**One sentence, both streams.** A JPEG and a PCM datagram can arrive together. Vision decodes and publishes `vision_result` on that frame; the coordinator caches faces. Audio keeps chunks until `finish_utterance`, then emits `speech`. The coordinator prints the line and enqueues `PARSE_INTENT`. When the intent comes back as `REGISTER_IDENTITY`, the coordinator may enqueue `REGISTER_FACE` and `REGISTER_VOICE`. The frame does not wait for the sentence, and the sentence does not wait for a face.

## API worker

`APIWorker` is a serial OpenAI client. The LLM is OpenAI only (`api/openai_client.py`, `core/config_openai.py`).

`run()` creates the asyncio loop, then pulls one command at a time from `llm_command_queue`. `command_name` uppercases `cmd`. `PARSE_INTENT` requires `text`, calls `parse_intent`, and `intent_event` writes `{type: intent, cmd, args, timestamp, voice_embedding}` when the result is not `None`. A `None` result is dropped. `ANALYZE_MEMORY` requires `conversation_history`, calls `analyze_memory`, and `memory_event` writes `{type: memory_result, subject, facts, timestamp}`. Any exception in the loop writes `{type: api_error, error, timestamp}` and the worker continues. HTTP is one shared aiohttp session in `OpenAIClient`, posting to `OPENAI_API_LINK`. `parse_intent` sends the system text and tools from `config_tools_openai.json`. `analyze_memory` fills `memory_prompt_template` from `config_prompts.json` and parses a JSON list (invalid JSON becomes `[]`). The vision worker calls `analyze_video_frames` itself when VLM is on. That call does not go through `llm_command_queue`.

The client timeout is 20 seconds. `run_until_complete` does not watch `self.running`, so an in-flight call ignores shutdown until it returns. `main.py` then `terminate`s the process if `join(1.0)` expires, which skips `client.close()`.

## Database

`DatabaseManager` (`database/database.py`) is the identity store. Live path is `IDENTITY_DB_PATH`, `./database/identities.db`. Connections enable foreign keys and WAL. Embeddings are numpy arrays stored as sqlite `ARRAY` blobs.

Tables: `users` (`id`, `name`); `face_embeddings` and `voice_embeddings` (many per user); `chat_history` (`transcript`, `timestamp`). Deleting a user cascades.

The parent loads samples only when `START_WITH_SAMPLE_DATA` is true (it is `True` in `core/config.py`). `SAMPLE_FACE_EMBEDDING_PATHS` is empty. `SAMPLE_VOICE_EMBEDDING_PATHS` loads Tiger's `.npy` files via `create_user` and `save_voice_embedding`. Workers open their own connections in the child and read those rows in `setup()`. On Ctrl+C, `main.py` calls `clear_db()` before stopping the workers. `clear_db` is `DELETE FROM users`, which cascades to embeddings and chat history. That wipes the live file, including anyone registered during the run.

The dashboard and the workers do not share a connection or a transaction. Each `save_*` commits on its own.

## Dashboard

`start_monitoring_server` runs an aiohttp app on a daemon thread at `MONITORING_HOST`:`MONITORING_PORT` (`0.0.0.0:8765`). The page shows coordinator and system logs, worker logs, identity rows, and recent chat, and it can rename, delete, and look up faces. Routes are listed in [frontend/AGENTS.md](frontend/AGENTS.md).

`install_log_interceptor` puts `{source, text, ts}` on `log_queue`. `_log_drain_loop` broadcasts `{type: log, ...}` on `/ws`. `_db_poll_loop` reads sqlite every 5 seconds and broadcasts a snapshot. A log line that matches the server's db-change keywords triggers an extra snapshot. The page does not send websocket messages. Writes from the page use a separate `DatabaseManager` on the same file.

**Logs.** `log_queue` is the only pipe from worker `print`s to the browser. The websocket is a fan-out of that queue plus occasional db snapshots. It is not in the recognition path.

## Where the logic lives

Worker files are the sequence of steps: `workers/audio.py`, `workers/vision.py`, `workers/coordinator.py`, `workers/api_worker.py`. Decisions live in the helper packages and take the owned objects as arguments: the db, caches, queues, and models. `workers/audio_utils/` (VAD, utterance, speaker), `workers/vision_utils/` (decode, tracks, matching, InspireFace), `workers/coordinator_utils/` (`handle_event`, association), `workers/api_utils/commands.py` (event shaping). `core/` is config, queues, and the log interceptor.

## Not on the hot path

`workers/audio_utils/voice_embedding_creator.py` is an offline wav-to-`.npy` script. The worker calls `voice_embedding` in `vad.py` instead. `workers/vision_utils/get_embedding.py` extracts one face embedding from a sample image. `facial_processing_demos` is not in this checkout, and `VisionWorker` does not import it. `tests/test_openai_tool.py`, `tests/test_openai_memory.py`, and `tests/test_api_worker_live.py` call OpenAI for real. The running pipeline does not.

## Limits and directions

Limits are what the current code does. Directions are the next change that follows from that limit.

- **Limit.** The start method is the interpreter default, `spawn` on macOS, and the code never sets it. The child only receives what `__init__` stored. InspireFace, ONNX, and Parakeet are created in `setup()` because a session built in the parent would have to be pickled into that new interpreter.
- **Limit.** Shutdown `join`s each worker for 1 second and `terminate`s whoever is left (`main.py`). `APIWorker` is inside `run_until_complete` for the HTTP call and does not see `running` until that call returns, so the kill skips `client.close()`.
- **Direction.** Join the API worker through the end of `run()` without `terminate`, so the in-flight call finishes and `client.close()` runs.
- **Limit.** `START_WITH_SAMPLE_DATA` is true. After Ctrl+C, `main.py` calls `clear_db()` on `./database/identities.db` before the workers stop. That deletes every user, and foreign keys take the embeddings and chat history with them, including people registered in that run.
- **Direction.** Stop calling `clear_db()` on the live database when sample data was loaded.
- **Limit.** An existing name is accepted only from `closest_frame` plus `face_matches`, or from one cosine comparison in `voice_matches` (0.30) on a self-intro. `identify_speaker` is the same shape: one embedding, a threshold, and a last-speaker bias. `associate_voice` returns a user id or `None`. It looks at a 1.0 second window, a 0.3 speaking ratio, and a 0.8 presence ratio when only one known person is there. Live faces never set `is_speaking`, so the speaking ratio does not fire. None of this walks a conversation.
- **Direction.** Return an association score from `associate_voice` with the id, instead of a bare id or nothing.
- **Limit.** End of sentence is `utterance.silence_count >= SILENT_CHUNK_THRESHOLD`, and that threshold is 1. One quiet 160 ms chunk calls `finish_utterance`.
- **Direction.** Decide the sentence boundary with more than `SILENT_CHUNK_THRESHOLD`.
- **Limit.** `ANALYZE_MEMORY` and `memory_event` already shape a `memory_result` (`subject`, `facts`). The coordinator never enqueues that command, and `handle_event` has no `memory_result` branch. Facts are not written to sqlite.
- **Direction.** Apply that `memory_result` as a small per-person fact record. The worker and `memory_event` already produce the event.
- **Limit.** This checkout has no `hardware/` tree. The imported firmware (`hardware/src/udp.cpp`, `WS_FRAME_VIDEO` `0x01`, `WS_FRAME_AUDIO_TX` `0x02`) and `core/config.py` (`HEADER_VISION`, `HEADER_AUDIO`) are the same byte values under different names. Nothing compares them. The host and the device meet only at that byte.
- **Direction.** One contract test that the header bytes in `hardware/src/udp.cpp` match `core/config.py`.
