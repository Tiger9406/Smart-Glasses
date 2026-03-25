# coordinator: looks at global queue and processes output from sub workers vision and audio
# kinda like the decision making part

import multiprocessing as mp
import queue
import time
import uuid
from collections import deque

import numpy as np

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
        self.llm_commands_queue = gemini_commands_queue
        self.vision_commands_queue = vision_commands_queue
        self.audio_commands_queue = audio_commands_queue
        self.db = DatabaseManager()

    def setup(self):
        self.start_time = time.time()
        self.request_number = 0

        self.CACHE_DURATION = 10.0
        self.vision_cache = deque()  # stores tuples of timestamp, face_list
        self.AUDIO_CACHE_DURATION = 30.0
        self.audio_cache = deque()
        self.max_delay = 20.0

        self.conversation_history = deque(maxlen=10)

        self.pending_voice_registration = None

        self.frame_area = config.RESOLUTION[0] * config.RESOLUTION[1]
        self.frame_center_x = config.RESOLUTION[0] / 2
        self.frame_center_y = config.RESOLUTION[1] / 2
        self.max_frame_distance = np.sqrt(
            self.frame_center_x**2 + self.frame_center_y**2
        )

        self.sample_embeddings = {}
        if config.TEST_REGISTER_IDENTITY and config.SAMPLE_FACE_EMBEDDING_PATHS:
            self.sample_embeddings = {
                name: np.load(path)
                for name, path in config.SAMPLE_FACE_EMBEDDING_PATHS.items()
            }

        self.event_handlers = {
            "vision_result": self._handle_vision_result,
            "speech": self._handle_speech,
            "vlm_result": self._handle_vlm_result,
            "intent": self._handle_intent,
            "api_error": self._handle_api_error,
        }

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
        event_type = event.get("type", "unknown")
        handler = self.event_handlers.get(event_type, self._handle_unknown_event)
        handler(event)

    def _handle_vision_result(self, event):
        faces = event.get("faces", [])
        self._update_vision_cache(faces)

    def _handle_speech(self, event):
        user_id = event.get("user_id", config.DEFAULT_ID)
        speaker_name = (
            self.db.get_user_name(user_id)
            if user_id != config.DEFAULT_ID
            else config.DEFAULT_NAME
        )
        text = event.get("text", "")
        timestamp = event.get("time_start", time.time())
        voice_embedding = event.get("embedding")

        if user_id == config.DEFAULT_ID and voice_embedding is not None:
            self._update_audio_cache(timestamp, voice_embedding)
            resolved = self._check_pending_voice_registration(timestamp, voice_embedding)

            if not resolved:
                self._try_visual_voice_association(timestamp, voice_embedding)

        active_names = set()
        if self.vision_cache:
            _, faces = self.vision_cache[-1] # Most recent frame
            for face in faces:
                uid = face.get("user_id", config.DEFAULT_ID)
                if uid != config.DEFAULT_ID:
                    active_names.add(self.db.get_user_name(uid))

        current_message = f"{speaker_name}: {text}"
        self.conversation_history.append(current_message)
        
        # NEW: Compile the history into a single string
        history_text = "\n".join(self.conversation_history)

        # NEW: Build a comprehensive prompt
        vision_context = f"Known people in view: {list(active_names)}\n" if active_names else ""
        prompt = (
            f"{vision_context}"
            f"Transcript:\n{history_text}\n\n"
            f"Task: Parse the intent for the latest message from {speaker_name}."
        )

        self.llm_commands_queue.put_nowait(
            {
                "cmd": "PARSE_INTENT",
                "text": prompt,
                "timestamp": timestamp,
                "voice_embedding": voice_embedding,
            }
        )
        print(f"[Coordinator] {speaker_name}: {event['text']}")

    def _handle_vlm_result(self, event):
        print(f"[Coordinator] Received VLM output: {event['text']}")

    def _handle_intent(self, event):
        command = event.get("cmd", "CHAT")
        args = event.get("args", {})
        timestamp = event.get("timestamp", time.time())
        voice_embedding = event.get("voice_embedding")

        if command == "REGISTER_IDENTITY":
            self._register_identity(args, timestamp, voice_embedding)
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

    def _handle_api_error(self, event):
        print(f"Error: {event.get('error')}\nTime: {event.get('timestamp')}")

    def _handle_unknown_event(self, event):
        print("\n[Coordinator] got other event")

    def _check_pending_voice_registration(self, timestamp, voice_embedding):
        """
        Handles when trap set to register next unknown speaker
        """
        if not self.pending_voice_registration:
            return False

        time_since_reg = timestamp - self.pending_voice_registration["timestamp"]
        if 0 <= time_since_reg < self.max_delay:
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
            return True
        return False

    def _register_identity(self, args, timestamp, voice_embedding):
        """
        Handles llm says to register person
        """
        name = args.get("name", config.DEFAULT_NAME)
        speaker_name = args.get("speaker_name", config.DEFAULT_NAME)
        is_self_intro = args.get("is_self_introduction", False)
        existing_user_ids = self.db.get_user_ids_by_name(name)

        if existing_user_ids:
            # user already exists in db
            self._handle_existing_identity(
                existing_user_ids,
                name,
                speaker_name,
                is_self_intro,
                timestamp,
                voice_embedding,
            )
        else:
            # user doesn't exist
            self._handle_new_identity(
                name, speaker_name, is_self_intro, timestamp, voice_embedding
            )

    def _handle_existing_identity(
        self,
        existing_user_ids,
        name,
        speaker_name,
        is_self_intro,
        timestamp,
        voice_embedding,
    ):
        matched_user_id = None
        is_face_already_known = False
        is_voice_already_known = False

        closest_frame = (
            min(self.vision_cache, key=lambda x: abs(x[0] - timestamp))
            if self.vision_cache
            else None
        )

        for uid in existing_user_ids:
            face_match = self._check_face_match(uid, closest_frame)
            voice_match = self._check_voice_match(uid, is_self_intro, voice_embedding)

            if face_match or voice_match:
                matched_user_id = uid
                is_face_already_known = face_match
                is_voice_already_known = voice_match
                break

        if matched_user_id:
            print(
                f"[Coordinator] Verified existing identity: {name} (ID: {matched_user_id})."
            )
            target_user_id = matched_user_id

            if not is_face_already_known:
                self._register_unknown_face(timestamp, target_user_id)

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
                self._attempt_voice_registration(timestamp, target_user_id)
        else:
            print(
                f"[Coordinator] Name '{name}' exists, but couldn't verify face/voice. Assuming existing identity."
            )
            target_user_id = existing_user_ids[0]
            if speaker_name == config.USER_NAME:
                self.pending_voice_registration = {
                    "user_id": target_user_id,
                    "timestamp": timestamp,
                }

    def _handle_new_identity(
        self, name, speaker_name, is_self_intro, timestamp, voice_embedding
    ):
        new_user_id = str(uuid.uuid4())
        self.db.create_user(new_user_id, name)
        print(f"[Coordinator] Created new identity: {name} ({new_user_id})")
        target_user_id = new_user_id

        self._register_unknown_face(timestamp, target_user_id)

        if speaker_name != config.USER_NAME:
            if (
                speaker_name == config.DEFAULT_NAME
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
            self._attempt_voice_registration(timestamp, target_user_id)

    def _check_face_match(self, uid, closest_frame):
        if not closest_frame:
            return False
        _, faces = closest_frame
        for face in faces:
            if face.get("user_id") == uid:
                return True
        return False

    def _check_voice_match(self, uid, is_self_intro, voice_embedding):
        if not (is_self_intro and voice_embedding is not None):
            return False
        stored_voices = self.db.get_voice_embeddings_by_uid(uid)
        if stored_voices:
            avg_stored_voice = np.mean(stored_voices, axis=0)
            sim = self._cosine_sim(voice_embedding, avg_stored_voice)
            return sim > 0.30
        return False

    def _register_unknown_face(self, timestamp, target_user_id):
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

    def _attempt_voice_registration(self, timestamp, target_user_id):
        target_voice = self._resolve_unknown_voice(timestamp)
        if target_voice is not None:
            self.audio_commands_queue.put_nowait(
                {
                    "cmd": "REGISTER_VOICE",
                    "user_id": target_user_id,
                    "embedding": target_voice,
                }
            )
        else:
            self.pending_voice_registration = {
                "user_id": target_user_id,
                "timestamp": timestamp,
            }

    def _update_vision_cache(self, faces):
        """Add and remove old face cache"""
        current_time = time.time()
        self.vision_cache.append((current_time, faces))

        while (
            self.vision_cache
            and (current_time - self.vision_cache[0][0]) > self.CACHE_DURATION
        ):
            self.vision_cache.popleft()

    def _update_audio_cache(self, timestamp, audio):
        current_time = time.time()
        self.audio_cache.append((timestamp, audio))
        while (
            self.audio_cache
            and (current_time - self.audio_cache[0][0]) > self.AUDIO_CACHE_DURATION
        ):
            self.audio_cache.popleft()

    def _resolve_unknown_voice(self, target_timestamp):
        if not self.audio_cache:
            return None

        for cached_timestamp, voice_embedding in self.audio_cache:
            time_diff = cached_timestamp - target_timestamp

            if 0 <= time_diff <= self.max_delay:
                return voice_embedding

        return None

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

    def _try_visual_voice_association(self, timestamp, voice_embedding):
        if not self.vision_cache:
            return False

        # Define a time window around the speech (e.g., +/- 1.0 second)
        window_size = 1.0
        frames_in_window = [
            faces for frame_time, faces in self.vision_cache
            if abs(frame_time - timestamp) <= window_size
        ]

        if not frames_in_window:
            return False

        speaker_counts = {}
        presence_counts = {}
        total_frames = len(frames_in_window)

        # Tally up presence and speaking flags across all frames in the window
        for faces in frames_in_window:
            for face in faces:
                uid = face.get("user_id", config.DEFAULT_ID)
                if uid == config.DEFAULT_ID:
                    continue
                
                presence_counts[uid] = presence_counts.get(uid, 0) + 1
                
                if face.get("is_speaking", False):
                    speaker_counts[uid] = speaker_counts.get(uid, 0) + 1

        target_id = None

        # Logic A: Find the person who is actively speaking in the most frames
        # Threshold: Must be flagged as speaking in at least 30% of the frames
        if speaker_counts:
            most_active_speaker = max(speaker_counts, key=speaker_counts.get)
            if (speaker_counts[most_active_speaker] / total_frames) >= 0.3:
                target_id = most_active_speaker

        # Logic B (Fallback): No clear speaker detected, but is there only ONE person consistently present?
        # Threshold: One person is present in > 80% of the frames, and no other known faces are detected.
        elif len(presence_counts) == 1:
            only_person = list(presence_counts.keys())[0]
            if (presence_counts[only_person] / total_frames) >= 0.8:
                target_id = only_person

        if target_id:
            target_name = self.db.get_user_name(target_id)
            print(f"[Coordinator] Visually associated voice with {target_name} across {total_frames} frames.")
            
            self.audio_commands_queue.put_nowait({
                "cmd": "REGISTER_VOICE",
                "user_id": target_id,
                "embedding": voice_embedding,
            })
            return True

        return False