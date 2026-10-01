# Workers

`BaseWorker` and `IngestionWorker` in `base.py` subclass `multiprocessing.Process`. The parent process calls `__init__`. After `start()`, the child calls `setup()` then `run()`.

## Parent vs child

`__init__` stores queues and other plain values the child needs. Do not construct InspireFace, ONNX, Parakeet, asyncio, or aiohttp there. The instance is sent into the child; those objects belong in `setup()` or `run()`.

`log_queue: mp.Queue` is required on every worker. Pass it through to `BaseWorker`. `run()` calls `install_log_interceptor(self.log_queue, prefix)` before `setup()`.

`shutdown()` clears `self.running`. Loops stop when `self.running.is_set()` is false.

`IngestionWorker` adds `input_queue` and `output_queue` (`audio.py`, `vision.py`, `api_worker.py`). `Coordinator` subclasses `BaseWorker` and reads `results_queue`.

## Sequence vs helpers

The worker file is the high-level sequence. Step details live in the helper package.

- `audio.py` — chunk PCM, detect speech, open and close an utterance, emit `speech`. [audio_utils/AGENTS.md](audio_utils/AGENTS.md)
- `vision.py` — apply commands, decode a frame, recognize faces, publish them. [vision_utils/AGENTS.md](vision_utils/AGENTS.md)
- `coordinator.py` — take one `results_queue` event and dispatch it. [coordinator_utils/AGENTS.md](coordinator_utils/AGENTS.md)
- `api_worker.py` — one LLM command at a time (`PARSE_INTENT`, `ANALYZE_MEMORY`). [api_utils/AGENTS.md](api_utils/AGENTS.md)

Payloads: [event_definitions.md](event_definitions.md).

## Types

Use `T | None` only when `None` is a real absence (no current utterance, queue timeout, nothing to publish). Do not mark a field optional because `setup()` has not assigned it yet.
