# audio_utils

Silero VAD state, the utterance/transcript session, speaker cosine match and profile merge, and voice registration commands. `AudioWorker` (`workers/audio.py`) owns the ONNX sessions, the Parakeet model, the database, and the input, output, and command queues. Functions here receive those objects.

## Worker loop

`setup` builds Silero (`VAD_PATH`, threshold 0.5, window 512, context 64, state `(2, 1, 128)` float32), Redimnet (`REDIMNET_PATH`), and Parakeet (`PARAKEET_MODEL`). `known_speakers` is `{user_id: {embedding: mean, count}}` from `get_all_voices`.

Each tick drains voice commands, then `next_audio` (0.5s; other queue errors become `RuntimeError`). `record_chunk` opens `open_wav` (mono s16, `AUDIO_SAMPLE_RATE_HZ`) when `SAVE_ANNOTATED_VID` is set. `take_chunks` splits on `chunk_bytes` (160 ms × 16 kHz × 2). `pcm_from_bytes` is int16 / 32767.0 float32. One chunk is 2560 samples, five VAD windows.

`hears_speech` starts (`begin_utterance`) or extends an utterance. Silence with no utterance fills `pre_speech` (deque, `int(320 / chunk_ms)` frames). Silence inside an utterance calls `note_silence` and keeps those samples. At `SILENT_CHUNK_THRESHOLD` the worker `reset_vad`s, `finish_utterance`s, and queues a non-null event. `last_speaker` updates even when the transcript is empty. Shutdown `close_stream`s (safe if `ctx` is already `None`) and closes the WAV.

## vad.py

`hears_speech` appends samples and runs each full window: `input` is context‖chunk, plus `sr` and `state`. Any probability above the threshold counts as speech. It writes back `state` and the last `context_size` samples. A short tail stays in `buffer`. `reset_vad` zeros state, context, and buffer. `voice_embedding(session, audio)` runs `{"audio": audio}` and returns output `[0][0]`. `as_ndarray` raises if an ONNX output is not an ndarray.

## utterance.py

`Utterance`: 8-char id, `started_at`, samples, `silence_count`, transcriber, stream context. `begin_utterance` opens `transcribe_stream((CONTEXT_LEFT, CONTEXT_RIGHT))`, copies `pre_speech`, then clears it. `extend_utterance` zeroes the silence count. `finish_utterance` concatenates, optionally writes `api/simulator_resources/debug_audios/` (`DEBUG_AUDIO`), `add_audio`s an MLX array, embeds `sentence.reshape(1, -1)`, and calls `identify_speaker`. `speech_event` is `{type: speech, text, id, time_start, timestamp, final: True, embedding, user_id}`; blank text is `None`.

`apply_voice_commands` drains `CommandQueue` and accepts only `{cmd: REGISTER_VOICE, user_id, embedding}`. `register_voice` calls `VoiceStore.save_voice_embedding` and `merge_speaker` into `known_speakers`. Other commands and exceptions are logged and skipped.

## speaker.py

`cosine_sim` is dot / norms. `identify_speaker` starts at `default_id` with score `threshold` (`SIMILARITY_THRESHOLD`, 0.30). If `last_user_id` is already known and beats the threshold, that score is the bar; another profile must be strictly higher. `last_user_id` must be the default or a key in `known_speakers`. `merge_speaker(None, emb)` starts at count 1; otherwise a running mean. Count 0 leaves the profile unchanged.

## voice_embedding_creator.py

Offline CLI: mono 16 kHz 16-bit WAV to a Redimnet `.npy`. It loads its own session. `__main__` paths are local. The worker never calls it.

## Tests

Unit-test the pure pieces in `tests/test_worker_helpers.py`: `pcm_from_bytes`, `take_chunks`, `speech_event`, `cosine_sim`, `identify_speaker`, `merge_speaker`, `register_voice`, and `apply_voice_commands` with a fake queue and `VoiceStore`. Do not load ONNX, Parakeet, or MLX in unit tests. `hears_speech`, `voice_embedding`, and `finish_utterance` need a process that already holds the models.
