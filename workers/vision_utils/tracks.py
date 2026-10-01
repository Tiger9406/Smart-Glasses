from collections.abc import Callable, Sequence
from typing import Any, Protocol, TypedDict

import numpy as np


class FaceTrack(TypedDict):
    track_id: int
    bbox: tuple[int, int, int, int]
    user_id: str
    name: str
    score: float
    emb: np.ndarray | None


class ActiveIdentity(TypedDict):
    user_id: str
    name: str
    score: float
    checked_ts: float
    last_seen: float


class ObservedFace(Protocol):
    track_id: int

    @property
    def location(self) -> Sequence[Any]: ...


def update_tracks(
    active: dict[int, ActiveIdentity],
    faces: Sequence[ObservedFace],
    now: float,
    *,
    recheck_interval: float,
    lost_after: float,
    confidence_threshold: float,
    default_id: str,
    default_name: str,
    recognize: Callable[[ObservedFace], tuple[np.ndarray, str, float]],
    name_for: Callable[[str], str],
) -> list[FaceTrack]:
    result: list[FaceTrack] = []

    for face in faces:
        track_id = face.track_id
        x1, y1, x2, y2 = map(int, face.location)

        if track_id not in active:
            active[track_id] = {
                "user_id": default_id,
                "name": default_name,
                "score": 0.0,
                "checked_ts": 0.0,
                "last_seen": now,
            }
        else:
            active[track_id]["last_seen"] = now

        identity_data = active[track_id]
        emb: np.ndarray | None = None
        should_recognize = identity_data["user_id"] == default_id or (
            now - identity_data["checked_ts"]
        ) > recheck_interval
        if should_recognize:
            emb, user_id, score = recognize(face)
            if score > confidence_threshold:
                identity_data.update(
                    {
                        "user_id": user_id,
                        "name": name_for(user_id),
                        "score": score,
                        "checked_ts": now,
                    }
                )
            else:
                identity_data.update(
                    {
                        "user_id": default_id,
                        "name": default_name,
                        "score": score,
                        "checked_ts": now,
                    }
                )

        result.append(
            {
                "track_id": track_id,
                "bbox": (x1, y1, x2, y2),
                "user_id": identity_data["user_id"],
                "name": identity_data["name"],
                "score": identity_data["score"],
                "emb": emb,
            }
        )

    expired_ids = [
        track_id
        for track_id, data in active.items()
        if (now - data["last_seen"]) > lost_after
    ]
    for track_id in expired_ids:
        del active[track_id]

    return result
