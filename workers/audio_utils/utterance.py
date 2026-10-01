import os
import time
import uuid
import wave
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from core import config
from workers.audio_utils.speaker import SpeakerProfile, identify_speaker, merge_speaker

class VoiceStore(Protocol):
    def save_voice_embedding(self, user_id: str, embedding: np.ndarray) -> None: ...


class CommandQueue(Protocol):
    def empty(self) -> bool: ...
    def get_nowait(self) -> Any: ...


# Consider an utterance a sentence—one piece of information we'll throw at stt
@dataclass
class Utterance:
    session_id: str
    started_at: float
    samples: list[np.ndarray]
    silence_count: int
    transcriber: Any
    ctx: Any


def pcm_from_bytes(chunk_bytes: bytes) -> np.ndarray:
    return np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32767.0


def take_chunks(buffer: bytes, chunk_bytes: int) -> tuple[bytes, list[bytes]]:
    chunks: list[bytes] = []
    while len(buffer) >= chunk_bytes:
        chunks.append(buffer[:chunk_bytes])
        buffer = buffer[chunk_bytes:]
    return buffer, chunks


def open_wav(path: str, sample_rate: int) -> wave.Wave_write:
    output_dir = os.path.dirname(path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    writer = wave.open(path, "wb")
    writer.setnchannels(1)
    writer.setsampwidth(2)
    writer.setframerate(sample_rate)
    return writer


def begin_utterance(
    model: Any,
    pre_speech: deque[np.ndarray],
    samples: np.ndarray,
    context_left: int,
    context_right: int,
) -> Utterance:
    ctx = model.transcribe_stream(context_size=(context_left, context_right))
    utterance = Utterance(
        session_id=str(uuid.uuid4())[:8],
        started_at=time.time(),
        samples=list(pre_speech),
        silence_count=0,
        transcriber=ctx.__enter__(),
        ctx=ctx,
    )
    pre_speech.clear()
    utterance.samples.append(samples)
    return utterance


def extend_utterance(utterance: Utterance, samples: np.ndarray) -> None:
    utterance.silence_count = 0
    utterance.samples.append(samples)


def note_silence(utterance: Utterance, samples: np.ndarray) -> None:
    utterance.silence_count += 1
    utterance.samples.append(samples)


def close_stream(utterance: Utterance | None) -> None:
    if utterance is None or utterance.ctx is None:
        return
    utterance.ctx.__exit__(None, None, None)
    utterance.ctx = None


def finish_utterance(
    utterance: Utterance,
    *,
    embed: Callable[[np.ndarray], np.ndarray],
    known_speakers: dict[str, SpeakerProfile],
    last_speaker: str,
    threshold: float,
    debug: bool,
) -> tuple[str, dict[str, Any] | None]:
    import mlx.core as mx

    sentence = np.concatenate(utterance.samples)
    if debug:
        save_debug_wav(sentence)
    utterance.transcriber.add_audio(mx.array(sentence))
    text = utterance.transcriber.result.text.strip()
    embedding = embed(sentence.reshape(1, -1))
    speaker = identify_speaker(
        embedding, known_speakers, last_speaker, threshold, config.DEFAULT_ID
    )
    close_stream(utterance)
    return speaker, speech_event(text, utterance, embedding, speaker)


def speech_event(
    text: str,
    utterance: Utterance,
    embedding: np.ndarray,
    speaker: str,
) -> dict[str, Any] | None:
    if not text:
        return None
    return {
        "type": "speech",
        "text": text,
        "id": utterance.session_id,
        "time_start": utterance.started_at,
        "timestamp": time.time(),
        "final": True,
        "embedding": embedding,
        "user_id": speaker,
    }


def save_debug_wav(sentence_audio: np.ndarray) -> None:
    from core import config_audio

    os.makedirs("api/simulator_resources/debug_audios", exist_ok=True)
    path = f"api/simulator_resources/debug_audios/debug_{int(time.time())}.wav"
    audio_int16 = (sentence_audio * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(config_audio.AUDIO_SAMPLE_RATE_HZ)
        writer.writeframes(audio_int16.tobytes())
    print("[Debug] Saved audio chunk")


def register_voice(
    db: VoiceStore,
    known_speakers: dict[str, SpeakerProfile],
    user_id: str,
    embedding: np.ndarray,
) -> None:
    db.save_voice_embedding(user_id, embedding)
    known_speakers[user_id] = merge_speaker(known_speakers.get(user_id), embedding)


def apply_voice_commands(
    command_queue: CommandQueue,
    db: VoiceStore,
    known_speakers: dict[str, SpeakerProfile],
) -> None:
    while not command_queue.empty():
        try:
            command = command_queue.get_nowait()
            if command.get("cmd") != "REGISTER_VOICE":
                continue
            user_id = command.get("user_id")
            embedding = command.get("embedding")
            if user_id and embedding is not None:
                register_voice(db, known_speakers, user_id, embedding)
                print(f"[Audio] Registered voice for id '{user_id}'")
        except Exception as error:
            print(f"[Audio] Command error: {error}")
