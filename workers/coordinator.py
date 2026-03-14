# coordinator: looks at global queue and processes output from sub workers vision and audio
# kinda like the decision making part

import multiprocessing as mp
import queue
import time
import uuid
from collections import deque

import numpy as np

from api.gemini_client import GeminiClient
from core import config
from database.database import DatabaseManager
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
        self.db = DatabaseManager()

    def setup(self):
        self.start_time = time.time()
        self.request_number = 0

        self.CACHE_DURATION = 10.0
        self.vision_cache = deque()  # stores tuples of timestamp, face_list
        self.pending_voice_registration = None

        self.frame_area = config.RESOLUTION[0] * config.RESOLUTION[1]
        self.frame_center_x = config.RESOLUTION[0] / 2
        self.frame_center_y = config.RESOLUTION[1] / 2
        self.max_frame_distance = np.sqrt(
            self.frame_center_x**2 + self.frame_center_y**2
        )

        self.sample_embeddings = {}
        self.registered_tracks = set()
        if config.TEST_REGISTER_IDENTITY and config.SAMPLE_FACE_EMBEDDING_PATHS:
            self.sample_embeddings = {
                name: np.load(path)
                for name, path in config.SAMPLE_FACE_EMBEDDING_PATHS.items()
            }

        self.gemini_client = GeminiClient()

    def run(self):
        print("[Coordinator] Started")
        self.setup()

        try:
            while self.running.is_set():
                try:
                    event = self.events_queue.get(timeout=0.1)
                    self._handle_event(event)
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
            pass

        elif event_type == "speech":
            user_id = event.get(
                "user_id", config.DEFAULT_ID
            )  # if known, uuid; otherwise unknown

            speaker_name = (
                self.db.get_user_name(user_id)
                if user_id != config.DEFAULT_ID
                else config.DEFAULT_NAME
            )  # if known uuid we get a name otherwise unknown

            text = event.get("text", "")
            timestamp = event.get("speech_start_time", time.time())
            voice_embedding = event.get("embedding")

            if user_id == config.DEFAULT_ID and self.pending_voice_registration:
                time_since_reg = (
                    timestamp - self.pending_voice_registration["timestamp"]
                )
                if time_since_reg < 8.0:
                    target_id = self.pending_voice_registration["user_id"]
                    target_name = self.db.get_user_name(target_id)
                    print(
                        f"[Coordinator] Binding unknown voice to pending identity: {target_name} ({target_id})"
                    )
                    self.audio_commands_queue.put_nowait(
                        {
                            "cmd": "REGISTER_VOICE",
                            "user_id": target_id,
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

            print(f"[Coordinator] {speaker_name}: {event['text']}")

        elif event_type == "vlm_result":
            print(f"[Coordinator] Received VLM output: {event['text']}")

        elif event_type == "intent":
            command = event.get("cmd", "CHAT")
            args = event.get("args", [])

            timestamp = event.get("timestamp", time.time())
            voice_embedding = event.get("voice_embedding")

            if command == "REGISTER_IDENTITY":
                # person to be registered
                name = args.get("name", config.DEFAULT_NAME)

                # whosever is speaking
                speaker_name = args.get("speaker_name", config.DEFAULT_NAME)
                is_self_intro = args.get("is_self_introduction", False)

                # uuids associated with name
                existing_user_ids = self.db.get_user_ids_by_name(name)

                if existing_user_ids:
                    matched_user_id = None
                    is_face_already_known = False
                    is_voice_already_known = False

                    closest_frame = None
                    if self.vision_cache:
                        closest_frame = min(
                            self.vision_cache, key=lambda x: abs(x[0] - timestamp)
                        )

                    for uid in existing_user_ids:
                        face_match = False
                        voice_match = False

                        # check vision cache; should be quick, how many faces can there
                        # be in a frame at once
                        if closest_frame:
                            _, faces = closest_frame
                            for face in faces:
                                # if frame already has uid, we have uuid and face alr stored
                                if face.get("user_id") == uid:
                                    face_match = True
                                    break

                        # check voice match; only applicable when self intro
                        if is_self_intro and voice_embedding is not None:
                            stored_voices = self.db.get_voice_embeddings_by_uid(uid)
                            if stored_voices:
                                avg_stored_voice = np.mean(stored_voices, axis=0)
                                sim = self._cosine_sim(
                                    voice_embedding, avg_stored_voice
                                )
                                if (
                                    sim > 0.30
                                ):  # if similar enough, we have a voice match
                                    voice_match = True

                        # if either match, we don't gotta create new user
                        if face_match or voice_match:
                            matched_user_id = uid
                            is_face_already_known = face_match
                            is_voice_already_known = voice_match
                            break
                    if matched_user_id:
                        # positive it's an existing person
                        print(
                            f"[Coordinator] Verified existing identity: {name} (ID: {matched_user_id})."
                        )
                        target_user_id = matched_user_id

                        if not is_face_already_known:
                            target_face = self._resolve_unknown_face(timestamp)
                            if target_face:
                                self.vision_commands_queue.put_nowait(
                                    {
                                        "cmd": "REGISTER_FACE",
                                        "track_id": target_face.get("track_id"),
                                        "user_id": target_user_id,
                                        "emb": target_face.get("emb"),
                                    }
                                )

                        if (
                            not is_voice_already_known
                            and is_self_intro
                            and voice_embedding is not None
                        ):
                            self.audio_commands_queue.put_nowait(
                                {
                                    "cmd": "REGISTER_VOICE",
                                    "user_id": target_user_id,
                                    "embedding": voice_embedding,
                                }
                            )

                        elif speaker_name == config.USER_NAME:
                            self.pending_voice_registration = {
                                "user_id": target_user_id,
                                "timestamp": timestamp,
                            }
                    else:
                        # name exists but couldn't verify face nor voice
                        # don't create new one for now
                        #TODO refine logic
                        print(
                            f"[Coordinator] Name '{name}' exists, but couldn't verify face/voice. Assuming existing identity."
                        )
                        target_user_id = existing_user_ids[0]

                        # Set trap in case they respond, so we can capture their voice later
                        if speaker_name == config.USER_NAME:
                            self.pending_voice_registration = {
                                "user_id": target_user_id,
                                "timestamp": timestamp,
                            }

                else:
                    new_user_id = str(uuid.uuid4())
                    self.db.create_user(new_user_id, name)
                    print(f"[Coordinator] Created new identity: {name} ({new_user_id})")
                    target_user_id = new_user_id

                    target_face = self._resolve_unknown_face(timestamp)
                    if target_face:
                        self.vision_commands_queue.put_nowait(
                            {
                                "cmd": "REGISTER_FACE",
                                "track_id": target_face.get("track_id"),
                                "user_id": target_user_id,
                                "emb": target_face.get("emb"),
                            }
                        )

                    # voice reg
                    if speaker_name != config.USER_NAME:
                        # unknown introducing themselves
                        if speaker_name == config.DEFAULT_NAME and is_self_intro:
                            self.audio_commands_queue.put_nowait(
                                {
                                    "cmd": "REGISTER_VOICE",
                                    "user_id": target_user_id,
                                    "embedding": voice_embedding,
                                }
                            )
                        else:
                            # known introducing someone else
                            # or unknown introducing self
                            # or unknown introducing other people
                            # don't know how to deterministically parse this
                            pass
                    elif speaker_name == config.USER_NAME:
                        self.pending_voice_registration = {
                            "user_id": target_user_id,
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

    def _resolve_unknown_face(self, target_timestamp):
        """
        Finds frame closest to target_timestamp and
        Return unknown face best matching wearer's attention
        """
        if not self.vision_cache:
            return None

        # finds closest metadata to said timestamp
        closest_frame = min(
            self.vision_cache, key=lambda x: abs(x[0] - target_timestamp)
        )
        _, faces = closest_frame

        if not faces:
            return None

        best_face = None
        highest_score = -float("inf")

        for face in faces:
            if face.get("user_id", config.DEFAULT_ID) != config.DEFAULT_ID:
                continue
            x1, y1, x2, y2 = face.get("bbox", (0, 0, 0, 0))
            area = (x2 - x1) * (y2 - y1)
            area_scaled = area / self.frame_area

            face_center_x = x1 + ((x2 - x1) / 2)
            face_center_y = y1 + ((y2 - y1) / 2)

            distance_to_center = np.sqrt(
                (face_center_x - self.frame_center_x) ** 2
                + (face_center_y - self.frame_center_y) ** 2
            )
            distance_scaled = distance_to_center / self.max_frame_distance

            weight_center = 0.7
            weight_size = 0.3

            score = (weight_size * area_scaled) - (weight_center * distance_scaled)

            if score > highest_score:
                highest_score = score
                best_face = face

        return best_face

    def _cosine_sim(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
