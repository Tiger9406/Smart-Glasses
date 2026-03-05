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
    def __init__(self, results_queue: mp.Queue, commands_queue: mp.Queue):
        super().__init__()
        self.results_queue = results_queue
        self.commands_queue = commands_queue

    def setup(self):
        self.start_time = time.time()
        self.request_number = 0

        self.CACHE_DURATION = 10.0
        self.vision_cache = deque()  # stores tuples of timestamp, face_list

        self.vlm_tested = False
        self.sample_embeddings = {}
        self.registered_tracks = set()

        if config.TEST_REGISTER_IDENTITY and config.SAMPLE_FACE_EMBEDDING_PATHS:
            self.sample_embeddings = {
                name: np.load(path)
                for name, path in config.SAMPLE_FACE_EMBEDDING_PATHS.items()
            }
            self.registered_tracks = set()

        self.gemini_client = GeminiClient()

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

    def run(self):
        print("[Coordinator] Started")
        self.setup()

        try:
            while self.running.is_set():
                try:
                    event = self.results_queue.get(timeout=0.1)
                    self._handle_event(event)
                    self._test_VLM()
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    break
        finally:
            print("[Coordinator] Shutting down")

    def _test_VLM(self):
        # comment return statement to test VLM funcitonality
        if not config.VLM_ACTIVE:
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
            self.commands_queue.put_nowait(command)

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
                    self.commands_queue.put_nowait(command)

                    # Mark as registered so we don't spam the queue
                    self.registered_tracks.add(track_id)
                    print(
                        f"[Coordinator] Put command to register {name} (Track {track_id}) in vision queue"
                    )
                    break

    def _handle_event(self, event):
        # handling events; gotta coordinate event data format
        # for instance if event type is a face in view, we throw it on the picture or sum

        event_type = event.get("type", "unknown")

        if event_type == "vision_result":
            # given list of the following:
            """{
                "track_id": track_id,
                "bbox": (x1, y1, x2, y2),
                "name": self.active_identities[track_id]["name"],
                "score": self.active_identities[track_id]["score"],
                "emb": emb # could be none if embedding not extracted on this frame
            }"""

            # potential code to work with input faces; nothing for now, too many frames
            """
            faces = event.get("faces", [])
            if faces:
                print(f"\n [Coordinator] Vision Event: detected {len(faces)} faces")
                for face in faces:
                    _name = face.get("name", config.DEFAULT_NAME)
                    _score = face.get("score", 0.0)
                    _bbox = face.get("bbox")
                    # print(f" - ID: {face['track_id']} | Name: {name} ({score:.2f}) | Loc: {bbox}")

            """
            faces = event.get("faces", [])
            self._update_vision_cache(faces)
            self._test_register_identity(event)
            pass

        elif event_type == "speech":
            # given:
            """
            "type": "speech",
            "text": text,    (would be the audio transcription)
            "id": session_id,
            "speech_start_time" : time.time()
            "timestamp": time.time(),
            "final": False,
            "name": Unkown,
            "embedding":
            """

            # TODO: parse for intent and if it's register face, get timestamp and have logic there

            print(f"[Coordinator] {event['name']}: {event['text']}")

        elif event_type == "vlm_result":
            """"
            "type": "vlm_result",
            "request_id": request_id,
            "text": response_text,
            "timestamp": time.time(),
            """
            print(f"[Coordinator] Received VLM output: {event['text']}")

        else:
            print("\n[Coordinator] got other event")

        return
