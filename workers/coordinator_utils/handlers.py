import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from core import config
from workers.coordinator_utils.association import (
    PendingVoiceRegistration,
    associate_voice,
    closest_frame,
    face_matches,
    pending_registration_hit,
    resolve_unknown_face,
    resolve_unknown_voice,
    trim_before,
    voice_matches,
)


class QueueWriter(Protocol):
    def put_nowait(self, obj: Any) -> None: ...


class IdentityDirectory(Protocol):
    def get_user_name(self, user_id: str) -> str: ...
    def save_chat_history(self, user_id: str, transcript: str) -> None: ...
    def get_user_ids_by_name(self, name: str) -> list[str]: ...
    def get_voice_embeddings_by_uid(self, user_id: str) -> list[np.ndarray]: ...
    def create_user(self, user_id: str, name: str) -> None: ...


@dataclass
class CoordinatorState:
    db: IdentityDirectory
    llm_commands: QueueWriter
    vision_commands: QueueWriter
    audio_commands: QueueWriter
    vision_cache: deque[tuple[float, list[dict[str, Any]]]]
    audio_cache: deque[tuple[float, np.ndarray]]
    conversation_history: deque[str]
    pending_voice_registration: PendingVoiceRegistration | None
    cache_duration: float
    audio_cache_duration: float
    max_delay: float
    frame_area: float
    frame_center_x: float
    frame_center_y: float
    max_frame_distance: float
    request_number: int


def handle_event(event: dict[str, Any], state: CoordinatorState) -> None:
    handlers: dict[str, Callable[[dict[str, Any], CoordinatorState], None]] = {
        "vision_result": handle_vision_result,
        "speech": handle_speech,
        "vlm_result": handle_vlm_result,
        "intent": handle_intent,
        "api_error": handle_api_error,
    }
    handler = handlers.get(event.get("type", "unknown"), handle_unknown_event)
    handler(event, state)


def handle_vision_result(event: dict[str, Any], state: CoordinatorState) -> None:
    # keep a short history of who was in frame
    now = time.time()
    state.vision_cache.append((now, event.get("faces", [])))
    trim_before(state.vision_cache, now, state.cache_duration)


def handle_speech(event: dict[str, Any], state: CoordinatorState) -> None:
    user_id = event.get("user_id", config.DEFAULT_ID)
    text = event.get("text", "")
    timestamp = event.get("time_start", time.time())
    voice_embedding = event.get("embedding")

    # unknown voice: stash it, then see if we were waiting on someone or a face lines up
    if user_id == config.DEFAULT_ID and voice_embedding is not None:
        remember_unknown_voice(state, timestamp, voice_embedding)
        user_id = (
            bind_pending_voice(state, timestamp, voice_embedding)
            or associate_unknown_voice(state, timestamp, voice_embedding)
            or user_id
        )

    # save the line and ask the llm what they just meant
    speaker_name = display_name(user_id, state.db.get_user_name)
    record_transcript(state, user_id, speaker_name, text)
    state.llm_commands.put_nowait(
        {
            "cmd": "PARSE_INTENT",
            "text": intent_prompt(state, speaker_name),
            "timestamp": timestamp,
            "voice_embedding": voice_embedding,
        }
    )
    print(f"[Coordinator] {speaker_name}: {event['text']}")


def handle_vlm_result(event: dict[str, Any], state: CoordinatorState) -> None:
    print(f"[Coordinator] Received VLM output: {event['text']}")


def handle_intent(event: dict[str, Any], state: CoordinatorState) -> None:
    command = event.get("cmd", "CHAT")
    args = event.get("args", {})
    timestamp = event.get("timestamp", time.time())
    voice_embedding = event.get("voice_embedding")

    # llm told us to add a person
    if command == "REGISTER_IDENTITY":
        register_identity(state, args, timestamp, voice_embedding)
    elif command == "SPEAK":
        message = args.get("message", "")
        if message:
            print(f"[Coordinator]: [STEVE]: {message}")
    # ask vision to describe what's in frame
    elif command == "VISION_CONTEXT":
        state.vision_commands.put_nowait(
            {
                "cmd": "GET_VIDEO_CONTEXT",
                "prompt": args.get("prompt", "Summarize the video in a sentence"),
                "request_id": state.request_number,
            }
        )


def handle_api_error(event: dict[str, Any], state: CoordinatorState) -> None:
    print(f"Error: {event.get('error')}\nTime: {event.get('timestamp')}")


def handle_unknown_event(event: dict[str, Any], state: CoordinatorState) -> None:
    print("\n[Coordinator] got other event")


def display_name(user_id: str, name_for: Callable[[str], str]) -> str:
    if user_id == config.DEFAULT_ID:
        return config.DEFAULT_NAME
    return name_for(user_id)


def remember_unknown_voice(
    state: CoordinatorState, timestamp: float, voice_embedding: np.ndarray
) -> None:
    now = time.time()
    state.audio_cache.append((timestamp, voice_embedding))
    trim_before(state.audio_cache, now, state.audio_cache_duration)


def bind_pending_voice(
    state: CoordinatorState, timestamp: float, voice_embedding: np.ndarray
) -> str | None:
    target_id = pending_registration_hit(
        state.pending_voice_registration, timestamp, state.max_delay
    )
    if target_id is None:
        return None
    print(
        "[Coordinator] Binding unknown voice to pending identity: "
        f"{state.db.get_user_name(target_id)} ({target_id})"
    )
    enqueue_register_voice(state.audio_commands, target_id, voice_embedding)
    state.pending_voice_registration = None
    return target_id


def associate_unknown_voice(
    state: CoordinatorState, timestamp: float, voice_embedding: np.ndarray
) -> str | None:
    target_id = associate_voice(state.vision_cache, timestamp)
    if target_id is None:
        return None
    print(
        "[Coordinator] Visually associated voice with "
        f"{state.db.get_user_name(target_id)}."
    )
    enqueue_register_voice(state.audio_commands, target_id, voice_embedding)
    return target_id


def record_transcript(
    state: CoordinatorState, user_id: str, speaker_name: str, text: str
) -> None:
    transcript = f"{speaker_name}: {text}"
    if text and user_id != config.DEFAULT_ID:
        state.db.save_chat_history(user_id, transcript)
    state.conversation_history.append(transcript)


def intent_prompt(state: CoordinatorState, speaker_name: str) -> str:
    active_names = names_in_view(state)
    vision_context = (
        f"Known people in view: {list(active_names)}\n" if active_names else ""
    )
    history_text = "\n".join(state.conversation_history)
    return (
        f"{vision_context}"
        f"Transcript:\n{history_text}\n\n"
        f"Task: Parse the intent for the latest message from {speaker_name}."
    )


def names_in_view(state: CoordinatorState) -> set[str]:
    if not state.vision_cache:
        return set()
    _, faces = state.vision_cache[-1]
    return {
        state.db.get_user_name(uid)
        for face in faces
        if (uid := face.get("user_id", config.DEFAULT_ID)) != config.DEFAULT_ID
    }


def register_identity(
    state: CoordinatorState,
    args: dict[str, Any],
    timestamp: float,
    voice_embedding: np.ndarray | None,
) -> None:
    name = args.get("name", config.DEFAULT_NAME)
    speaker_name = args.get("speaker_name", config.DEFAULT_NAME)
    is_self_intro = args.get("is_self_introduction", False)
    existing_user_ids = state.db.get_user_ids_by_name(name)
    # same name already stored, otherwise make a new person
    if existing_user_ids:
        bind_existing_identity(
            state,
            existing_user_ids,
            name,
            speaker_name,
            is_self_intro,
            timestamp,
            voice_embedding,
        )
        return
    create_identity(state, name, speaker_name, is_self_intro, timestamp, voice_embedding)


def bind_existing_identity(
    state: CoordinatorState,
    existing_user_ids: list[str],
    name: str,
    speaker_name: str,
    is_self_intro: bool,
    timestamp: float,
    voice_embedding: np.ndarray | None,
) -> None:
    # check the face in frame, or the voice, actually belongs to that name
    frame = closest_frame(state.vision_cache, timestamp)
    matched_user_id = None
    face_known = False
    voice_known = False
    for uid in existing_user_ids:
        face_known = face_matches(uid, frame)
        voice_known = voice_is_known(state, uid, is_self_intro, voice_embedding)
        if face_known or voice_known:
            matched_user_id = uid
            break

    if matched_user_id is None:
        print(
            f"[Coordinator] Name '{name}' exists, but couldn't verify face/voice. Assuming existing identity."
        )
        # couldn't prove it; if the wearer said the name, wait for that person's voice
        if speaker_name == config.USER_NAME:
            state.pending_voice_registration = {
                "user_id": existing_user_ids[0],
                "timestamp": timestamp,
            }
        return

    # matched, so fill in whichever of face or voice we still don't have
    print(f"[Coordinator] Verified existing identity: {name} (ID: {matched_user_id}).")
    if not face_known:
        register_unknown_face(state, timestamp, matched_user_id)
    if not voice_known and is_self_intro and voice_embedding is not None:
        enqueue_register_voice(state.audio_commands, matched_user_id, voice_embedding)
    elif speaker_name == config.USER_NAME:
        attempt_voice_registration(state, timestamp, matched_user_id)


def create_identity(
    state: CoordinatorState,
    name: str,
    speaker_name: str,
    is_self_intro: bool,
    timestamp: float,
    voice_embedding: np.ndarray | None,
) -> None:
    user_id = str(uuid.uuid4())
    state.db.create_user(user_id, name)
    print(f"[Coordinator] Created new identity: {name} ({user_id})")
    register_unknown_face(state, timestamp, user_id)
    # they introduced themselves, so keep the voice that just spoke
    if (
        speaker_name == config.DEFAULT_NAME
        and is_self_intro
        and voice_embedding is not None
    ):
        enqueue_register_voice(state.audio_commands, user_id, voice_embedding)
    # wearer named someone else, so grab a recent unknown voice or wait for the next one
    elif speaker_name == config.USER_NAME:
        attempt_voice_registration(state, timestamp, user_id)


def voice_is_known(
    state: CoordinatorState,
    uid: str,
    is_self_intro: bool,
    voice_embedding: np.ndarray | None,
) -> bool:
    if not (is_self_intro and voice_embedding is not None):
        return False
    return voice_matches(
        voice_embedding,
        state.db.get_voice_embeddings_by_uid(uid),
        is_self_intro=True,
    )


def register_unknown_face(
    state: CoordinatorState, timestamp: float, user_id: str
) -> None:
    # unknown face nearest the center of the frame, around when the name was said
    face = resolve_unknown_face(
        state.vision_cache,
        timestamp,
        frame_area=state.frame_area,
        frame_center_x=state.frame_center_x,
        frame_center_y=state.frame_center_y,
        max_frame_distance=state.max_frame_distance,
    )
    if not face:
        return
    state.vision_commands.put_nowait(
        {
            "cmd": "REGISTER_FACE",
            "track_id": face.get("track_id"),
            "user_id": user_id,
            "emb": face.get("emb"),
        }
    )


def attempt_voice_registration(
    state: CoordinatorState, timestamp: float, user_id: str
) -> None:
    # use an unknown voice from just after the intro, or set a trap for the next one
    voice = resolve_unknown_voice(state.audio_cache, timestamp, state.max_delay)
    if voice is not None:
        enqueue_register_voice(state.audio_commands, user_id, voice)
        return
    state.pending_voice_registration = {"user_id": user_id, "timestamp": timestamp}


def enqueue_register_voice(queue: QueueWriter, user_id: str, embedding: np.ndarray) -> None:
    queue.put_nowait(
        {"cmd": "REGISTER_VOICE", "user_id": user_id, "embedding": embedding}
    )
