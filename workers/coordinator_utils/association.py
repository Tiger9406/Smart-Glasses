from collections import deque
from collections.abc import Sequence
from typing import Any, TypedDict

import numpy as np

from core import config


class PendingVoiceRegistration(TypedDict):
    user_id: str
    timestamp: float


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def trim_before(
    cache: deque[tuple[float, Any]], now: float, duration: float
) -> None:
    while cache and (now - cache[0][0]) > duration:
        cache.popleft()


def pending_registration_hit(
    pending: PendingVoiceRegistration | None, timestamp: float, max_delay: float
) -> str | None:
    if not pending:
        return None
    time_since_reg = timestamp - pending["timestamp"]
    if 0 <= time_since_reg < max_delay:
        return pending["user_id"]
    return None


def closest_frame(
    vision_cache: Sequence[tuple[float, list[dict[str, Any]]]], timestamp: float
) -> tuple[float, list[dict[str, Any]]] | None:
    if not vision_cache:
        return None
    return min(vision_cache, key=lambda frame: abs(frame[0] - timestamp))


def face_matches(
    uid: str, closest: tuple[float, list[dict[str, Any]]] | None
) -> bool:
    if not closest:
        return False
    _, faces = closest
    return any(face.get("user_id") == uid for face in faces)


def voice_matches(
    embedding: np.ndarray | None,
    stored_voices: list[np.ndarray],
    *,
    is_self_intro: bool,
    threshold: float = 0.30,
) -> bool:
    if not (is_self_intro and embedding is not None and stored_voices):
        return False
    avg_stored_voice = np.mean(stored_voices, axis=0)
    return cosine_sim(embedding, avg_stored_voice) > threshold


def resolve_unknown_voice(
    audio_cache: Sequence[tuple[float, np.ndarray]],
    target_timestamp: float,
    max_delay: float,
) -> np.ndarray | None:
    for cached_timestamp, voice_embedding in audio_cache:
        time_diff = cached_timestamp - target_timestamp
        if 0 <= time_diff <= max_delay:
            return voice_embedding
    return None


def resolve_unknown_face(
    vision_cache: Sequence[tuple[float, list[dict[str, Any]]]],
    target_timestamp: float,
    *,
    frame_area: float,
    frame_center_x: float,
    frame_center_y: float,
    max_frame_distance: float,
    default_id: str = config.DEFAULT_ID,
) -> dict[str, Any] | None:
    if not vision_cache:
        return None

    _, faces = min(vision_cache, key=lambda frame: abs(frame[0] - target_timestamp))
    if not faces:
        return None

    best_face = None
    highest_score = -float("inf")
    for face in faces:
        if face.get("user_id", default_id) != default_id:
            continue
        x1, y1, x2, y2 = face.get("bbox", (0, 0, 0, 0))
        area = (x2 - x1) * (y2 - y1)
        area_scaled = area / frame_area

        face_center_x = x1 + ((x2 - x1) / 2)
        face_center_y = y1 + ((y2 - y1) / 2)
        distance_to_center = np.sqrt(
            (face_center_x - frame_center_x) ** 2
            + (face_center_y - frame_center_y) ** 2
        )
        distance_scaled = distance_to_center / max_frame_distance
        score = (0.3 * area_scaled) - (0.7 * distance_scaled)
        if score > highest_score:
            highest_score = score
            best_face = face

    return best_face


def associate_voice(
    vision_cache: Sequence[tuple[float, list[dict[str, Any]]]],
    timestamp: float,
    *,
    default_id: str = config.DEFAULT_ID,
    window_size: float = 1.0,
) -> str | None:
    frames_in_window = [
        faces
        for frame_time, faces in vision_cache
        if abs(frame_time - timestamp) <= window_size
    ]
    if not frames_in_window:
        return None

    speaker_counts: dict[str, int] = {}
    presence_counts: dict[str, int] = {}
    total_frames = len(frames_in_window)

    for faces in frames_in_window:
        for face in faces:
            uid = face.get("user_id", default_id)
            if uid == default_id:
                continue
            presence_counts[uid] = presence_counts.get(uid, 0) + 1
            if face.get("is_speaking", False):
                speaker_counts[uid] = speaker_counts.get(uid, 0) + 1

    target_id = None
    if speaker_counts:
        most_active_speaker = max(speaker_counts, key=lambda uid: speaker_counts[uid])
        if (speaker_counts[most_active_speaker] / total_frames) >= 0.3:
            target_id = most_active_speaker
    elif len(presence_counts) == 1:
        only_person = next(iter(presence_counts))
        if (presence_counts[only_person] / total_frames) >= 0.8:
            target_id = only_person

    return target_id
