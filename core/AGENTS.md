# core

Process plumbing only: config, queues, and logging. Domain rules do not belong here.

## config.py

Host, UDP headers, names, paths, and flags. `HOST` / `PORT`, `HEADER_VISION` (`b"\x01"`), `HEADER_AUDIO` (`b"\x02"`), `USER_NAME`, `DEFAULT_NAME`, `DEFAULT_ID`, `RESOLUTION`, `IDENTITY_DB_PATH`, simulator output paths, `VOICE_TRAP_LIMIT`, `MONITORING_HOST` / `MONITORING_PORT`.

`START_WITH_SAMPLE_DATA` loads `SAMPLE_FACE_EMBEDDING_PATHS` and `SAMPLE_VOICE_EMBEDDING_PATHS` at startup. On shutdown, `main.py` calls `DatabaseManager.clear_db()` and wipes the identity db.

## config_vision.py

InspireFace settings: `FPS`, `DEFAULT_ISF_MODEL`, Megatron and Pikachu paths from the environment, `get_model_path`, detection and matching thresholds, `BUFFER_DURATION`, `VLM_ACTIVE`, annotated video path.

## config_audio.py

`PARAKEET_MODEL`, `REDIMNET_PATH`, `VAD_PATH`, `AUDIO_CHUNK_SIZE_MS`, `AUDIO_SAMPLE_RATE_HZ`, left/right context, `SILENT_CHUNK_THRESHOLD`, `SIMILARITY_THRESHOLD`, `DEBUG_AUDIO`.

## config_openai.py

OpenAI is the LLM. Gemini config is gone. `OPENAI_API_KEY`, `OPENAI_API_LINK`, `TEMPERATURE`, `MAXOUTPUTTOKENS`, `MODEL`, `VISION_MODEL`, `USER_NAME`, `VLM_ACTIVE`. Live schema paths: `TOOLS_JSON_PATH` (`config_tools_openai.json`) and `PROMPT_JSON_PATH` (`config_prompts.json`).

## shared_mem.py

`SharedMem` creates the process queues. `shutdown` cancels join threads and closes every queue.

- `vision_queue`, `audio_queue`: frame ingress, maxsize 100.
- `results_queue`: worker results.
- `vision_command_queue`, `audio_command_queue`: worker commands.
- `llm_command_queue`: the LLM command queue.
- `log_queue`: maxsize 2000.

## log_interceptor.py

`install_log_interceptor(log_queue, source_tag)` runs inside the target process. It keeps the real `print` and `put_nowait`s `{source, text, ts}` onto `log_queue`. A full queue is dropped.

## JSON

`config_tools_openai.json` declares `register_identity`, `speak`, and `vision_context`. `config_prompts.json` holds `memory_prompt_template`. `config_tools.json` is unused Gemini `functionDeclarations` schema.
