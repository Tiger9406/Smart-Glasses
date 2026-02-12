import json
import queue
import time
import uuid
from typing import Optional

import mlx.core as mx
import numpy as np
import onnxruntime as ort
from parakeet_mlx import from_pretrained

from core import config
from workers.base import IngestionWorker


class AudioWorker(IngestionWorker):
    def setup(self):
        # so it doesn't read the config every single loop
        self.session = ort.InferenceSession("workers/audio_utils/redimnet_b2.onnx")

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
        self.similarity_threshold = config.SIMILARITY_THRESHOLD

        # load embeddings in from json so we can manually add em n stuff
        with open("workers/audio_utils/EmbeddingDict.json") as f:
            speaker_paths = json.load(f)

        # average their embeddings in indentify_speaker
        self.known_speakers = {
            name: [np.load(p) for p in paths] for name, paths in speaker_paths.items()
        }
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

    def get_embedding(self, audio):
        return self.session.run(None, {"audio": audio})[0][0]

    def cosine_sim(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def identify_speaker(self, embedding) -> Optional[str]:
        # optional so i can convert it to last speaker if it throws None
        best_name = None
        best_score = self.similarity_threshold

        for person in self.known_speakers:
            avg_emb = np.mean(self.known_speakers[person], axis=0)

            similarity = self.cosine_sim(embedding, avg_emb)
            if similarity > best_score:
                best_score = similarity
                best_name = person
        return best_name

    def run(self):
        self.setup()

        audio_buffer = b""
        context_size = (self.context_left, self.context_right)

        audio_chunk_holder = []  # this is for storing all chunks to voice recognize at end of sentence
        last_speaker = None

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
                        audio_chunk_holder.append(samples)

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

                                sentence_audio = np.concatenate(audio_chunk_holder)
                                audio_reshaped = sentence_audio.reshape(
                                    1, -1
                                )  # to make it 2d arr
                                embedding = self.get_embedding(audio_reshaped)
                                identity = self.identify_speaker(embedding)
                                if identity:
                                    speaker = identity
                                    last_speaker = speaker
                                else:
                                    speaker = last_speaker or "Unkown"
                                if last_text:
                                    self.output_queue.put(
                                        {
                                            "type": "speech",
                                            "text": last_text,
                                            "id": session_id,
                                            "timestamp": time.time(),
                                            "final": True,
                                            "embedding": embedding,
                                            "name": speaker,
                                        }
                                    )
                                # close session
                                ctx.__exit__(None, None, None)
                                transcriber = None
                                ctx = None
                                session_id = None
                                last_text = ""
                                silence_count = 0
                                audio_chunk_holder = []
        finally:
            if ctx:
                ctx.__exit__(None, None, None)
