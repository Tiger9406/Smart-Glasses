import collections
import multiprocessing as mp
import os
import queue
import time
import uuid
import wave

import mlx.core as mx
import numpy as np
import onnxruntime as ort
from parakeet_mlx import from_pretrained

from core import config, config_audio
from database.database import DatabaseManager
from workers.base import IngestionWorker


class AudioWorker(IngestionWorker):
    def __init__(
        self,
        input_queue: mp.Queue,
        output_queue: mp.Queue,
        audio_command_queue: mp.Queue,
    ):
        super().__init__(input_queue, output_queue)
        self.command_queue = audio_command_queue

    def setup(self):
        chunk_ms = config_audio.AUDIO_CHUNK_SIZE_MS
        sample_rate = config_audio.AUDIO_SAMPLE_RATE_HZ
        self.silent_chunks = config_audio.SILENT_CHUNK_THRESHOLD

        self.chunk_samples = int(sample_rate * chunk_ms / 1000)  # 16khz * ms of chunks
        self.chunk_bytes = self.chunk_samples * 2
        self.similarity_threshold = config_audio.SIMILARITY_THRESHOLD

        self.padding_chunks = int(320 / chunk_ms)  # how many chunks make up 320ms
        self.chunk_duration_sec = chunk_ms / 1000.0

        self.audio_writer = None

        print("[AudioWorker] Loading Redimnet model...")
        self.redimnet_session = ort.InferenceSession(config_audio.REDIMNET_PATH)

        print("[AudioWorker] Loading Silero VAD...")
        self.vad_session = ort.InferenceSession(config_audio.VAD_PATH)
        self.vad_threshold = 0.5  # anything above 0.5 probability is considered speech

        self.vad_sample_rate = np.array(sample_rate, dtype=np.int64)  # sticky note
        self.vad_window = 512  # 32ms at 16kHz; new model requirement
        self.vad_context_size = 64  # a model innate parameter; 64 samples of context

        self.reset_vad_states()

        print(f"[AudioWorker] Loading model: {config_audio.PARAKEET_MODEL}")
        self.model = from_pretrained(config_audio.PARAKEET_MODEL)

        self.db = DatabaseManager()
        speaker_embeddings = self.db.get_all_voices()

        # average their embeddings in indentify_speaker
        self.known_speakers = {
            user_id: {"embedding": np.mean(embeddings, axis=0), "count": len(embeddings)}
            for user_id, embeddings in speaker_embeddings.items()
        }

        print(f"[AudioWorker] Loaded {len(self.known_speakers)} voice identities")

        print(f"[AudioWorker] Ready. Chunk: {chunk_ms}ms ({self.chunk_bytes} bytes)")

    def reset_vad_states(self):
        """
        Called at end of sentence; wipe previous state
        """
        # Silero v5 expects state as (2, batch_size, 128)
        self.vad_state = np.zeros((2, 1, 128), dtype=np.float32)

        # 64 contexts from prev chunk; context for new info
        self.vad_context = np.zeros((1, self.vad_context_size), dtype=np.float32)

        # buffer to handle input chunk sample not perfectly divisible by 512
        self.vad_buffer = np.array([], dtype=np.float32)

    def speech_checker(self, speech) -> bool:
        # takes new speech chunk into vad_buffer
        self.vad_buffer = np.concatenate((self.vad_buffer, speech))

        speech_detected = False

        while len(self.vad_buffer) >= self.vad_window:
            # Pop 512 samples from the front
            chunk = self.vad_buffer[: self.vad_window].reshape(1, -1)
            self.vad_buffer = self.vad_buffer[self.vad_window :]

            # Concatenate the previous 64-sample context to the front of the 512 chunk
            # model actually expects 576 samples (64 + 512)
            x = np.concatenate((self.vad_context, chunk), axis=1).astype(np.float32)

            ort_inputs = {
                "input": x,  # audio file
                "sr": self.vad_sample_rate,  # sample rate: 16khz
                "state": self.vad_state,  # current state/chain of thought
            }

            # run inference
            out, state = self.vad_session.run(None, ort_inputs)

            # update state and context for the NEXT 512-sample chunk
            self.vad_state = state
            self.vad_context = chunk[:, -self.vad_context_size :]  # last 64 bytes

            # Check if this specific 32ms micro-chunk contains speech
            if out[0][0] > self.vad_threshold:
                speech_detected = True

        # Returns True if any part of the 300ms window had speech
        return speech_detected

    def get_embedding(self, audio):
        return self.redimnet_session.run(None, {"audio": audio})[0][0]

    def cosine_sim(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def identify_speaker(self, embedding, last_user_id) -> str:
        # optional so i can convert it to last speaker if it throws None
        best_uid = config.DEFAULT_ID
        best_score = self.similarity_threshold

        if last_user_id != config.DEFAULT_ID:
            avg_emb_last_speaker = self.known_speakers[last_user_id]["embedding"]
            similarity_last_speaker = self.cosine_sim(embedding, avg_emb_last_speaker)
            if similarity_last_speaker > self.similarity_threshold:
                best_uid = last_user_id
                best_score = similarity_last_speaker

        for uid, data in self.known_speakers.items():
            similarity = self.cosine_sim(embedding, data["embedding"])
            if similarity > best_score:
                best_score = similarity
                best_uid = uid

        return best_uid

    def run(self):
        self.setup()

        audio_buffer = b""
        audio_chunk_holder = []  # this is for storing all chunks to voice recognize at end of sentence
        last_speaker = config.DEFAULT_ID 

        # session state
        session_id = None
        transcriber = None
        ctx = None
        silence_count = 0  # track consecutive silent chunks

        self.pre_speech_buffer = collections.deque(maxlen=self.padding_chunks)

        try:
            while self.running.is_set():
                self._handle_commands()
                try:
                    # so it doesnt block forevers
                    raw_bytes = self.input_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                except Exception as E:
                    print("[AudioWorker] Error: ", E)
                    raise RuntimeError
                
                if getattr(config_audio, "SAVE_AUDIO_STREAM", False) and self.audio_writer is None:
                    self._init_audio_writer()

                if self.audio_writer:
                    self.audio_writer.writeframes(raw_bytes)

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

                            speech_start_time = time.time()

                            ctx = self.model.transcribe_stream(
                                context_size=(
                                    config_audio.CONTEXT_LEFT,
                                    config_audio.CONTEXT_RIGHT,
                                )
                            )
                            transcriber = ctx.__enter__()

                            for buffered_samples in self.pre_speech_buffer:
                                audio_chunk_holder.append(buffered_samples)
                            self.pre_speech_buffer.clear()

                        audio_chunk_holder.append(samples)

                    else:  # silence
                        if transcriber is not None:
                            silence_count += 1

                            audio_chunk_holder.append(samples)

                            # if past certain number chunks & no speech
                            if silence_count >= self.silent_chunks:
                                # sentence break
                                self.reset_vad_states()

                                sentence_audio = np.concatenate(audio_chunk_holder)

                                if config_audio.DEBUG_AUDIO:
                                    self._debug_audio(session_id, sentence_audio)

                                transcriber.add_audio(mx.array(sentence_audio))
                                final_text = transcriber.result.text.strip()

                                audio_reshaped = sentence_audio.reshape(
                                    1, -1
                                )  # to make it 2d arr
                                embedding = self.get_embedding(audio_reshaped)
                                speaker = self.identify_speaker(embedding, last_speaker)
                                last_speaker = speaker

                                if final_text:
                                    self.output_queue.put(
                                        {
                                            "type": "speech",
                                            "text": final_text,
                                            "id": session_id,
                                            "time_start": speech_start_time,
                                            "timestamp": time.time(),
                                            "final": True,
                                            "embedding": embedding,
                                            "user_id": speaker,
                                        }
                                    )
                                # close session
                                ctx.__exit__(None, None, None)
                                transcriber = None
                                ctx = None
                                session_id = None
                                silence_count = 0
                                audio_chunk_holder = []

                        else:
                            self.pre_speech_buffer.append(samples)
        finally:
            if ctx:
                ctx.__exit__(None, None, None)
            if hasattr(self, "audio_writer") and self.audio_writer:
                self.audio_writer.close()
                print("[AudioWorker] AudioWriter released")

    def register_identity(self, user_id, embedding):
        curr_data = self.known_speakers.get(user_id, None)
        self.db.save_voice_embedding(user_id, embedding)
        if curr_data is None:
            self.known_speakers[user_id] = {"embedding": embedding, "count": 1}
            return
        curr_embedding = curr_data.get("embedding", None)
        curr_count = curr_data.get("count", 0)

        if curr_embedding is not None and curr_count != 0:
            new_embedding = (curr_embedding * curr_count + embedding) / (curr_count + 1)
            self.known_speakers[user_id] = {
                "embedding": new_embedding,
                "count": curr_count + 1,
            }

    def _handle_commands(self):
        while not self.command_queue.empty():
            try:
                command = self.command_queue.get_nowait()
                if command.get("cmd") == "REGISTER_VOICE":
                    # Expected payload: {"cmd": "REGISTER_VOICE", "user_id": "whatever userid", "embedding": np.ndarray}
                    user_id = command.get("user_id")
                    emb = command.get("embedding")
                    if user_id and emb is not None:
                        self.register_identity(user_id, emb)
                        print(f"[Audio] Registered voice for id '{user_id}'")
                else:
                    pass
            except Exception as e:
                print(f"[Audio] Command error: {e}")

    def _debug_audio(self, session_id, sentence_audio):
        os.makedirs("api/simulator_resources/debug_audios", exist_ok=True)
        debug_filename = (
            f"api/simulator_resources/debug_audios/debug_{int(time.time())}.wav"
        )

        # Convert the float32 array (-1.0 to 1.0) back to int16 PCM
        audio_int16 = (sentence_audio * 32767.0).astype(np.int16)

        with wave.open(debug_filename, "wb") as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 2 bytes = 16-bit
            wf.setframerate(config_audio.AUDIO_SAMPLE_RATE_HZ)
            wf.writeframes(audio_int16.tobytes())

        print("[Debug] Saved audio chunk")

    def _init_audio_writer(self):
        output_path = getattr(config_audio, 'RECORDED_AUDIO_PATH', 'api/simulator_resources/recorded_audio.wav')
        output_dir = os.path.dirname(output_path)
        
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        self.audio_writer = wave.open(output_path, "wb")
        self.audio_writer.setnchannels(1)  # Mono
        self.audio_writer.setsampwidth(2)  # 2 bytes = 16-bit
        self.audio_writer.setframerate(config_audio.AUDIO_SAMPLE_RATE_HZ)
        print(f"[AudioWorker] AudioWriter initialized: {output_path}")