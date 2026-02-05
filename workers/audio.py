from workers.base import IngestionWorker
import queue
import time
import uuid

import mlx.core as mx
import numpy as np
from parakeet_mlx import from_pretrained

from core import config


class AudioWorker(IngestionWorker):
    def setup(self):
        # so it doesn't read the config every single loop

        print(f"[AudioWorker] Loading model: {config.PARAKEET_MODEL}")
        self.model = from_pretrained(config.PARAKEET_MODEL)
        self.chunk_ms = config.AUDIO_CHUNK_SIZE_MS
        self.sample_rate = config.AUDIO_SAMPLE_RATE_HZ
        self.context_left = config.CONTEXT_LEFT
        self.context_right = config.CONTEXT_RIGHT
        self.silent_chunks = config.SPEECH_CHUNK_SIZE
        self.loudness_threshold = config.LOUDNESS_THRESHOLD
        self.chunk_samples = int(self.sample_rate * self.chunk_ms / 1000)
        self.chunk_bytes = self.chunk_samples * 2

        print(
            f"[AudioWorker] Ready. Chunk: {self.chunk_ms}ms ({self.chunk_bytes} bytes)"
        )

    def speech_checker(self, speech) -> bool:
        loudness = np.sqrt(
            np.mean(speech**2)
        )  # since its a wave between 0 and 1 its squared so its positive, find mean and sqrt it to make it normalize
        return (
            loudness > self.loudness_threshold
        )  # if the mean is very low, the user likely isnt talking

    def run(self):
        self.setup()

        audio_buffer = b""
        context_size = (self.context_left, self.context_right)

        # session state
        session_id = None
        transcriber = None
        ctx = None
        last_text = ""
        silence_count = 0  # track consecutive silent chunks

        try:
            while self.running.is_set():
                try:
                    raw_bytes = self.input_queue.get(
                        timeout=1.0
                    )  # so it doesnt block forevers

                except queue.Empty:
                    continue
                except Exception as E:
                    print("[Error] ", E)
                    raise RuntimeError

                audio_buffer += raw_bytes

                while len(audio_buffer) >= self.chunk_bytes:
                    chunk_bytes = audio_buffer[: self.chunk_bytes]
                    audio_buffer = audio_buffer[self.chunk_bytes :]

                    samples = (
                        np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32)
                        / 32767.0
                    )
                    is_speech = self.speech_checker(samples)  # check if speech

                    if is_speech:
                        silence_count = 0  # reset silence counter cus speech

                        # start new session if needed could start with no speech
                        if transcriber is None:
                            session_id = str(uuid.uuid4())[:8]
                            ctx = self.model.transcribe_stream(
                                context_size=context_size
                            )
                            transcriber = ctx.__enter__()
                            last_text = ""

                        transcriber.add_audio(mx.array(samples))
                        text = transcriber.result.text.strip()

                        if text and text != last_text:
                            last_text = text

                    else:
                        # silence
                        if transcriber is not None:
                            silence_count += 1

                            if silence_count >= self.silent_chunks:
                                # sentence break
                                if last_text:
                                    self.output_queue.put(
                                        {
                                            "type": "speech",
                                            "text": last_text,
                                            "id": session_id,
                                            "timestamp": time.time(),
                                            "final": True,
                                            # name and embedding added here soon
                                        }
                                    )
                                # close session
                                ctx.__exit__(None, None, None)
                                transcriber = None
                                ctx = None
                                session_id = None
                                last_text = ""
                                silence_count = 0
        finally:
            if ctx:
                ctx.__exit__(None, None, None)
