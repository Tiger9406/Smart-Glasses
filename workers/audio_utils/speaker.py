from typing import TypedDict

import numpy as np


class SpeakerProfile(TypedDict):
    embedding: np.ndarray
    count: int


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def identify_speaker(
    embedding: np.ndarray,
    known_speakers: dict[str, SpeakerProfile],
    last_user_id: str,
    threshold: float,
    default_id: str,
) -> str:
    best_uid = default_id
    best_score = threshold

    if last_user_id != default_id:
        avg_emb_last_speaker = known_speakers[last_user_id]["embedding"]
        similarity_last_speaker = cosine_sim(embedding, avg_emb_last_speaker)
        if similarity_last_speaker > threshold:
            best_uid = last_user_id
            best_score = similarity_last_speaker

    for uid, data in known_speakers.items():
        similarity = cosine_sim(embedding, data["embedding"])
        if similarity > best_score:
            best_score = similarity
            best_uid = uid

    return best_uid


def merge_speaker(
    profile: SpeakerProfile | None, embedding: np.ndarray
) -> SpeakerProfile:
    if profile is None:
        return {"embedding": embedding, "count": 1}

    curr_embedding = profile.get("embedding")
    curr_count = profile.get("count", 0)
    if curr_embedding is not None and curr_count != 0:
        new_embedding = (curr_embedding * curr_count + embedding) / (curr_count + 1)
        return {"embedding": np.asarray(new_embedding), "count": curr_count + 1}
    return profile
