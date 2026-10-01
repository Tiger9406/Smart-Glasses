# api

Device ingress and the LLM HTTP client. Package (`api/__init__.py`). `main.py` starts the UDP thread; `workers/` calls the client.

## openai_client.py

`OpenAIClient` is the only LLM client (Gemini is gone). It posts to the OpenAI-compatible chat URL in `core/config_openai.py`, using that module's models, temperature, and token cap. System text and tools come from `core/config_tools_openai.json`; the memory template comes from `core/config_prompts.json`.

- `parse_intent`: tool call becomes `{"cmd": NAME, "args": {...}}`; plain text becomes `CHAT`. `APIWorker` uses this for `PARSE_INTENT`.
- `analyze_memory`: JSON from conversation history and known facts. `APIWorker` uses this for `ANALYZE_MEMORY`. Invalid JSON returns `[]`.
- `analyze_video_frames`: JPEG bytes as data URLs on the vision model. The vision pipeline calls this.

One shared `aiohttp` session (`ssl=False`). Call `close()`. A missing API key raises `ValueError`. Non-200 raises `RuntimeError`.

## udp_receiver.py

`udp_receiver_thread(shared_mem)` binds `core.config.HOST`:`PORT`, which is `0.0.0.0:8000`. Each datagram is one header byte plus payload (`HEADER_VISION` `b"\x01"`, `HEADER_AUDIO` `b"\x02"`). Payloads go to `SharedMem.vision_queue` and `audio_queue`. A full queue drops the oldest item. Set `udp_shutdown_event` to stop. Started from `main.py`.

## simulator.py

Local stand-in for the glasses. From the repo root:

```
python -m api.simulator
```

Config is `api/simulator_resources/simulator_config.json` (`host` `0.0.0.0`, `port` `8000`, vision header `1`, audio header `2`). Host `0.0.0.0` is sent to `127.0.0.1`. It loops the configured video as JPEG frames and the WAV as PCM. The WAV must match the configured channel count and sample width. JPEG quality drops if a datagram would exceed 65400 bytes.

## Other files

`video_converter.py` and `simulator_resources/video_slower.py` are one-shot OpenCV scripts with hardcoded paths. Importing `video_converter.py` runs the conversion. `simulator_resources/` holds the simulator config plus local video and audio used by the simulator and debug output paths. `README.md` still says WebSockets; the transport is UDP.
