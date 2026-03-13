# coordinator: looks at global queue and processes output from sub workers vision and audio
# kinda like the decision making part

import multiprocessing as mp
import queue
import time
from collections import deque

import inspireface as isf
import numpy as np

from api.gemini_client import GeminiClient
from core import config
from workers.base import BaseWorker


class Coordinator(BaseWorker):
    def __init__(
        self,
        results_queue: mp.Queue,
        gemini_commands_queue: mp.Queue,
        vision_commands_queue: mp.Queue,
        audio_commands_queue: mp.Queue,
    ):
        super().__init__()
        self.events_queue = results_queue
        self.gemini_commands_queue = gemini_commands_queue
        self.vision_commands_queue = vision_commands_queue
        self.audio_commands_queue = audio_commands_queue

    def setup(self):
        self.start_time = time.time()
        self.request_number = 0

        self.CACHE_DURATION = 10.0
        self.vision_cache = deque()  # stores tuples of timestamp, face_list
        self.pending_voice_registration = None

        self.sample_embeddings = {}
        self.registered_tracks = set()
        if config.TEST_REGISTER_IDENTITY and config.SAMPLE_FACE_EMBEDDING_PATHS:
            self.sample_embeddings = {
                name: np.load(path)
                for name, path in config.SAMPLE_FACE_EMBEDDING_PATHS.items()
            }

        self.gemini_client = GeminiClient()
        self.vlm_tested = False

    def run(self):
        print("[Coordinator] Started")
        self.setup()

        try:
            while self.running.is_set():
                try:
                    event = self.events_queue.get(timeout=0.1)
                    self._handle_event(event)
                    self._test_VLM()
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    break
        finally:
            print("[Coordinator] Shutting down")

    def _handle_event(self, event):
        # handling events; gotta coordinate event data format
        # for instance if event type is a face in view, we throw it on the picture or sum

        event_type = event.get("type", "unknown")

        if event_type == "vision_result":
            faces = event.get("faces", [])
            self._update_vision_cache(faces)
            self._test_register_identity(event)
            pass

        elif event_type == "speech":
            speaker_name = event.get("name", config.DEFAULT_NAME)
            text = event.get("text", "")
            timestamp = event.get("speech_start_time", time.time())
            voice_embedding = event.get("embedding")

            if speaker_name == config.DEFAULT_NAME and self.pending_voice_registration:
                time_since_reg = (
                    timestamp - self.pending_voice_registration["timestamp"]
                )
                if time_since_reg < 15.0:
                    target_name = self.pending_voice_registration["name"]
                    print(
                        f"[Coordinator] Binding unknown voice to pending identity: {target_name}"
                    )
                    self.audio_commands_queue.put_nowait(
                        {
                            "cmd": "REGISTER_VOICE",
                            "name": target_name,
                            "embedding": voice_embedding,
                        }
                    )
                    self.pending_voice_registration = None

            prompt = f"Speaker: {speaker_name}. Speech: '{text}'"
            self.gemini_commands_queue.put_nowait(
                {
                    "cmd": "PARSE_INTENT",
                    "text": prompt,
                    "timestamp": timestamp,
                    "voice_embedding": voice_embedding,
                }
            )

            print(f"[Coordinator] {event['name']}: {event['text']}")

        elif event_type == "vlm_result":
            print(f"[Coordinator] Received VLM output: {event['text']}")

        elif event_type == "intent":
            command = event.get("cmd", "CHAT")
            args = event.get("args", [])

            timestamp = event.get("timestamp", time.time())
            voice_embedding = event.get("voice_embedding")

            if command == "REGISTER_IDENTITY":
                name = args.get("name", config.DEFAULT_NAME)
                speaker_name = args.get("speaker_name", "Unknown")
                is_self_intro = args.get("is_self_introduction", False)

                target_face = self._resolve_target_face(timestamp)
                if target_face:
                    self.vision_commands_queue.put_nowait(
                        {
                            "cmd": "REGISTER_FACE",
                            "track_id": target_face.get("track_id"),
                            "name": name,
                            "emb": target_face.get("emb"),
                        }
                    )

                if speaker_name != config.USER_NAME:
                    if speaker_name == config.DEFAULT_NAME and is_self_intro:
                        self.audio_commands_queue.put_nowait(
                            {
                                "cmd": "REGISTER_VOICE",
                                "name": name,
                                "embedding": voice_embedding,
                            }
                        )
                    else:
                        # TODO: someone else introducint someone else; no clue bruh
                        pass
                elif speaker_name == config.USER_NAME:
                    # user introducing an individual; set a trap for next speaker
                    self.pending_voice_registration = {
                        "name": name,
                        "timestamp": timestamp,
                    }
            elif command == "SPEAK":
                message = args.get("message", "")
                if message:
                    print(f"[Coordinator]: [STEVE]: {message}")
            elif command == "VISION_CONTEXT":
                prompt = args.get("prompt", "Summarize the video in a sentence")
                self.vision_commands_queue.put_nowait(
                    {
                        "cmd": "GET_VIDEO_CONTEXT",
                        "prompt": prompt,
                        "request_id": self.request_number,
                    }
                )

        elif event_type == "api_error":
            print(f"Error: {event.get('error')}\nTime: {event.get('timestamp')}")

        else:
            print("\n[Coordinator] got other event")

        return

    def _update_vision_cache(self, faces):
        """Add and remove old face cache"""
        current_time = time.time()
        self.vision_cache.append((current_time, faces))

        while (
            self.vision_cache
            and (current_time - self.vision_cache[0][0]) > self.CACHE_DURATION
        ):
            self.vision_cache.popleft()

    def _resolve_target_face(self, target_timestamp):
        """Finds frame closest to target_timestamp and return largest unknown face"""
        if not self.vision_cache:
            return None

        # finds closest metadata to said timestamp
        closest_frame = min(
            self.vision_cache, key=lambda x: abs(x[0] - target_timestamp)
        )
        _, faces = closest_frame

        if not faces:
            return None

        max_area = 0
        best_face = None
        for face in faces:
            if face.get("name", config.DEFAULT_NAME) != config.DEFAULT_NAME:
                continue
            x1, y1, x2, y2 = face.get("bbox", (0, 0, 0, 0))
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                best_face = face

        return best_face

    def _test_VLM(self):
        # comment return statement to test VLM funcitonality
        if self.vlm_tested:
            return
        if time.time() - self.start_time > 5:
            self.vlm_tested = True
            command = {
                "cmd": "GET_VIDEO_CONTEXT",
                "prompt": "Summarize the video in a sentence",
                "request_id": self.request_number,
            }
            self.request_number += 1
            print("[Coordinator] Put command in vision queue")
            self.vision_commands_queue.put_nowait(command)

    def _test_register_identity(self, event: list[dict]):
        if not config.TEST_REGISTER_IDENTITY:
            return
        faces = event.get("faces", [])

        for face in faces:
            new_emb = face.get("emb", None)
            track_id = face.get("track_id")

            # skip if no embedding or we already registered
            if new_emb is None or track_id in self.registered_tracks:
                continue

            for name, emb in self.sample_embeddings.items():
                score = isf.feature_comparison(emb, new_emb)
                if score > 0.5:
                    command = {
                        "cmd": "REGISTER_FACE",
                        "track_id": track_id,
                        "name": name,
                        "emb": new_emb,
                    }
                    self.request_number += 1
                    self.vision_commands_queue.put_nowait(command)

                    # Mark as registered so we don't spam the queue
                    self.registered_tracks.add(track_id)
                    print(
                        f"[Coordinator] Put command to register {name} (Track {track_id}) in vision queue"
                    )
                    break
