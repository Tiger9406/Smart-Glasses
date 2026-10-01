import collections
import multiprocessing as mp
import queue
import wave

import numpy as np
import onnxruntime as ort
from parakeet_mlx import from_pretrained

from core import config, config_audio
from core.log_interceptor import install_log_interceptor
from database.database import DatabaseManager
from workers.audio_utils.speaker import SpeakerProfile
from workers.audio_utils.utterance import (
    Utterance,
    apply_voice_commands,
    begin_utterance,
    close_stream,
    extend_utterance,
    finish_utterance,
    note_silence,
    open_wav,
    pcm_from_bytes,
    take_chunks,
)
from workers.audio_utils.vad import VadState, hears_speech, reset_vad, voice_embedding
from workers.base import IngestionWorker


class AudioWorker(IngestionWorker):
    command_queue: mp.Queue
    silent_chunks: int
    chunk_samples: int
    chunk_bytes: int
    similarity_threshold: float
    padding_chunks: int
    chunk_duration_sec: float
    audio_writer: wave.Wave_write | None
    redimnet_session: ort.InferenceSession
    vad: VadState
    model: object
    db: DatabaseManager
    known_speakers: dict[str, SpeakerProfile]

    def __init__(
        self,
        input_queue: mp.Queue,
        output_queue: mp.Queue,
        audio_command_queue: mp.Queue,
        log_queue: mp.Queue,
    ) -> None:
        super().__init__(input_queue, output_queue, log_queue=log_queue)
        self.command_queue = audio_command_queue

    def setup(self) -> None:
        chunk_ms = config_audio.AUDIO_CHUNK_SIZE_MS
        sample_rate = config_audio.AUDIO_SAMPLE_RATE_HZ
        self.silent_chunks = config_audio.SILENT_CHUNK_THRESHOLD
        self.chunk_samples = int(sample_rate * chunk_ms / 1000)
        self.chunk_bytes = self.chunk_samples * 2
        self.similarity_threshold = config_audio.SIMILARITY_THRESHOLD
        self.padding_chunks = int(320 / chunk_ms)
        self.chunk_duration_sec = chunk_ms / 1000.0
        self.audio_writer = None

        print("[AudioWorker] Loading Redimnet model...")
        self.redimnet_session = ort.InferenceSession(config_audio.REDIMNET_PATH)

        print("[AudioWorker] Loading Silero VAD...")
        context_size = 64
        self.vad = VadState(
            session=ort.InferenceSession(config_audio.VAD_PATH),
            threshold=0.5,
            sample_rate=np.array(sample_rate, dtype=np.int64),
            window=512,
            context_size=context_size,
            state=np.zeros((2, 1, 128), dtype=np.float32),
            context=np.zeros((1, context_size), dtype=np.float32),
            buffer=np.array([], dtype=np.float32),
        )

        print(f"[AudioWorker] Loading model: {config_audio.PARAKEET_MODEL}")
        self.model = from_pretrained(config_audio.PARAKEET_MODEL)

        self.db = DatabaseManager()
        speaker_embeddings = self.db.get_all_voices()
        self.known_speakers = {
            user_id: {"embedding": np.mean(embeddings, axis=0), "count": len(embeddings)}
            for user_id, embeddings in speaker_embeddings.items()
        }
        print(f"[AudioWorker] Loaded {len(self.known_speakers)} voice identities")
        print(f"[AudioWorker] Ready. Chunk: {chunk_ms}ms ({self.chunk_bytes} bytes)")

    def run(self) -> None:
        install_log_interceptor(self.log_queue, "[AudioWorker]")
        self.setup()

        audio_buffer = b""
        last_speaker = config.DEFAULT_ID
        utterance: Utterance | None = None
        pre_speech: collections.deque[np.ndarray] = collections.deque(
            maxlen=self.padding_chunks
        )

        try:
            while self.running.is_set():
                apply_voice_commands(self.command_queue, self.db, self.known_speakers)

                raw_bytes = next_audio(self.input_queue)
                if raw_bytes is None:
                    continue

                self.audio_writer = record_chunk(self.audio_writer, raw_bytes)

                # converts the audio to chunks that we will use to process
                audio_buffer, chunks = take_chunks(
                    audio_buffer + raw_bytes, self.chunk_bytes
                )

                for chunk in chunks:
                    samples = pcm_from_bytes(chunk)
                    if hears_speech(self.vad, samples):
                        # utterance, consider it "sentence"
                        if utterance is None:
                            utterance = begin_utterance(
                                self.model,
                                pre_speech,
                                samples,
                                config_audio.CONTEXT_LEFT,
                                config_audio.CONTEXT_RIGHT,
                            )
                        else:
                            extend_utterance(utterance, samples)
                        continue

                    if utterance is None:
                        pre_speech.append(samples)
                        continue

                    # increments the number of blocks we've been silent for
                    note_silence(utterance, samples)
                    if utterance.silence_count < self.silent_chunks:
                        continue

                    # Reached end of sentence
                    reset_vad(self.vad)

                    # return the sentence in text & associated outputs
                    last_speaker, event = finish_utterance(
                        utterance,
                        embed=lambda audio: voice_embedding(self.redimnet_session, audio),
                        known_speakers=self.known_speakers,
                        last_speaker=last_speaker,
                        threshold=self.similarity_threshold,
                        debug=config_audio.DEBUG_AUDIO,
                    )
                    utterance = None
                    if event is not None:
                        self.output_queue.put(event)
        finally:
            close_stream(utterance)
            if self.audio_writer is not None:
                self.audio_writer.close()
                print("[AudioWorker] AudioWriter released")


def next_audio(input_queue: mp.Queue) -> bytes | None:
    try:
        return input_queue.get(timeout=0.5)
    except queue.Empty:
        return None
    except Exception as error:
        print("[AudioWorker] Error: ", error)
        raise RuntimeError


def record_chunk(writer: wave.Wave_write | None, raw_bytes: bytes) -> wave.Wave_write | None:
    if config.SAVE_ANNOTATED_VID and writer is None:
        writer = open_wav(config.AUDIO_OUTPUT_PATH, config_audio.AUDIO_SAMPLE_RATE_HZ)
        print(f"[AudioWorker] AudioWriter initialized: {config.AUDIO_OUTPUT_PATH}")
    if writer is not None:
        writer.writeframes(raw_bytes)
    return writer
