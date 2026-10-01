from collections.abc import Callable


def best_identity(
    user_ids: list[str],
    score_person: Callable[[str], float],
    threshold: float,
    default_id: str,
) -> tuple[str, float]:
    best_score = 0.0
    best_match = default_id

    for user_id in user_ids:
        score = score_person(user_id)
        if score > threshold and score > best_score:
            best_score = score
            best_match = user_id

    return best_match, best_score
