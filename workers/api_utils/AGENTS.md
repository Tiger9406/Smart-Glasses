# api_utils

Pure helpers for the API worker. No network, no event loop, no queues.

`workers/api_worker.py` is a serial OpenAI request loop. `APIWorker.run()` creates the loop with `asyncio.new_event_loop()` and handles `llm_command_queue` one command at a time. `__init__` only stores `OpenAIClient()` and sets `loop` to `None`. Do not create the loop in `__init__` or in this package.

The HTTP client lives in `api/openai_client.py`. `LLMClient` on the worker is the protocol tests implement. This package must not import that client or call the network.

## commands.py

`command_name` uppercases `cmd`. A missing cmd is `""`.

`intent_event` shapes a `PARSE_INTENT` result into an `intent` event. Return `None` when `text` is empty or `result` is `None`. Keep the command `timestamp` (fallback `now`) and pass through `voice_embedding`. Default `cmd` is `CHAT` and default `args` is `{}`.

`memory_event` shapes an `ANALYZE_MEMORY` result into a `memory_result` event. Return `None` when `conversation_history` is empty. `subject` falls back to `default_name` (`DEFAULT_NAME` from the worker). `facts` is the raw client result. Timestamp is `now`.

Unknown commands stay in the worker and emit nothing. Run-loop exceptions become `api_error` events there, not here.

## Tests

`tests/test_api_worker_queue.py` uses a fake client plus an attached loop (`attach_loop`). Those tests do not hit OpenAI. Call `_process_command` only after a loop is attached; `run_async` raises `RuntimeError` if `loop` is `None`.

`tests/test_worker_helpers.py` (`test_command_events`) covers these helpers with plain dicts. Prefer that file for helper changes.

Event field names match `workers/event_definitions.md` (`intent`, `memory_result`).
