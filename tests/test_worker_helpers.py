from collections import deque

import numpy as np

from core.config import DEFAULT_ID, DEFAULT_NAME, USER_NAME
from workers.api_utils.commands import command_name, intent_event, memory_event
from workers.audio_utils.speaker import cosine_sim, identify_speaker, merge_speaker
from workers.audio_utils.utterance import (
    Utterance,
    apply_voice_commands,
    pcm_from_bytes,
    register_voice,
    speech_event,
    take_chunks,
)
from workers.coordinator_utils.association import (
    associate_voice,
    face_matches,
    pending_registration_hit,
    resolve_unknown_face,
    resolve_unknown_voice,
    trim_before,
    voice_matches,
)
from workers.coordinator_utils.handlers import CoordinatorState, handle_event
from workers.vision_utils.matching import best_identity
from workers.vision_utils.pipeline import register_tracked_face, sampled_frames
from workers.vision_utils.tracks import update_tracks


class Face:
    def __init__(self, track_id: int, location: tuple[float, float, float, float]) -> None:
        self.track_id = track_id
        self.location = location


def test_speaker_matching() -> None:
    tiger = np.array([1.0, 0.0])
    other = np.array([0.0, 1.0])
    speakers = {
        "tiger": {"embedding": tiger, "count": 1},
        "other": {"embedding": other, "count": 1},
    }
    assert cosine_sim(tiger, tiger) == 1.0
    assert identify_speaker(tiger, speakers, DEFAULT_ID, 0.5, DEFAULT_ID) == "tiger"
    assert identify_speaker(np.array([0.1, 0.1]), speakers, DEFAULT_ID, 0.99, DEFAULT_ID) == DEFAULT_ID

    merged = merge_speaker(speakers["tiger"], tiger)
    assert merged["count"] == 2
    assert np.allclose(merged["embedding"], tiger)
    first = merge_speaker(None, other)
    assert first["count"] == 1


def test_track_updates() -> None:
    active: dict = {}
    face = Face(7, (1, 2, 11, 22))
    calls = {"n": 0}

    def recognize(_face: Face):
        calls["n"] += 1
        return np.array([1.0]), "user-1", 0.9

    result = update_tracks(
        active,
        [face],
        now=10.0,
        recheck_interval=2.0,
        lost_after=1.0,
        confidence_threshold=0.5,
        default_id=DEFAULT_ID,
        default_name=DEFAULT_NAME,
        recognize=recognize,
        name_for=lambda user_id: "Ada",
    )
    assert calls["n"] == 1
    assert result[0]["user_id"] == "user-1"
    assert result[0]["name"] == "Ada"
    assert result[0]["emb"] is not None

    again = update_tracks(
        active,
        [face],
        now=11.0,
        recheck_interval=2.0,
        lost_after=1.0,
        confidence_threshold=0.5,
        default_id=DEFAULT_ID,
        default_name=DEFAULT_NAME,
        recognize=recognize,
        name_for=lambda user_id: "Ada",
    )
    assert calls["n"] == 1
    assert again[0]["emb"] is None

    update_tracks(
        active,
        [],
        now=13.0,
        recheck_interval=2.0,
        lost_after=1.0,
        confidence_threshold=0.5,
        default_id=DEFAULT_ID,
        default_name=DEFAULT_NAME,
        recognize=recognize,
        name_for=lambda user_id: "Ada",
    )
    assert 7 not in active


def test_best_identity() -> None:
    scores = {"a": 0.2, "b": 0.8}
    match, score = best_identity(list(scores), lambda uid: scores[uid], 0.5, DEFAULT_ID)
    assert match == "b"
    assert score == 0.8
    match, score = best_identity([], lambda uid: 1.0, 0.5, DEFAULT_ID)
    assert match == DEFAULT_ID
    assert score == 0.0


def test_association() -> None:
    assert pending_registration_hit(None, 10.0, 20.0) is None
    pending = {"user_id": "u1", "timestamp": 10.0}
    assert pending_registration_hit(pending, 15.0, 20.0) == "u1"
    assert pending_registration_hit(pending, 40.0, 20.0) is None

    frame = (10.0, [{"user_id": "u1"}, {"user_id": DEFAULT_ID}])
    assert face_matches("u1", frame)
    assert not face_matches("missing", frame)
    assert not face_matches("u1", None)

    voice = np.array([1.0, 0.0])
    assert voice_matches(voice, [voice], is_self_intro=True, threshold=0.5)
    assert not voice_matches(voice, [np.array([0.0, 1.0])], is_self_intro=True, threshold=0.5)
    assert not voice_matches(voice, [voice], is_self_intro=False)

    audio_cache = [(12.0, voice), (50.0, np.array([0.0, 1.0]))]
    assert resolve_unknown_voice(audio_cache, 10.0, 5.0) is voice
    assert resolve_unknown_voice(audio_cache, 10.0, 1.0) is None

    known = {"user_id": "known", "bbox": (0, 0, 10, 10)}
    unknown_center = {"user_id": DEFAULT_ID, "bbox": (300, 220, 340, 260)}
    unknown_edge = {"user_id": DEFAULT_ID, "bbox": (0, 0, 20, 20)}
    cache = [(5.0, [known, unknown_edge, unknown_center])]
    chosen = resolve_unknown_face(
        cache,
        5.0,
        frame_area=640 * 480,
        frame_center_x=320,
        frame_center_y=240,
        max_frame_distance=(320**2 + 240**2) ** 0.5,
    )
    assert chosen == unknown_center

    speaking = [{"user_id": "ada", "is_speaking": True}]
    quiet = [{"user_id": "ada", "is_speaking": False}]
    vision = [(0.0, speaking), (0.5, speaking), (1.0, quiet)]
    assert associate_voice(vision, 0.5) == "ada"
    assert associate_voice([], 0.5) is None

    timed: deque = deque([(0.0, "old"), (9.0, "new")])
    trim_before(timed, 10.0, 5.0)
    assert list(timed) == [(9.0, "new")]


def test_command_events() -> None:
    assert command_name({"cmd": "parse_intent"}) == "PARSE_INTENT"
    assert command_name({}) == ""

    event = intent_event(
        {"text": "hi", "timestamp": 3.0, "voice_embedding": [1]},
        {"cmd": "CHAT", "args": {"message": "hello"}},
        9.0,
    )
    assert event is not None
    assert event["type"] == "intent"
    assert event["timestamp"] == 3.0
    assert intent_event({"text": ""}, {"cmd": "CHAT"}, 1.0) is None
    assert intent_event({"text": "hi"}, None, 1.0) is None

    memory = memory_event(
        {"conversation_history": "talk", "subject": "Ada"},
        [{"fact": "likes tea"}],
        4.0,
        "Unknown",
    )
    assert memory is not None
    assert memory["subject"] == "Ada"
    assert memory_event({"conversation_history": ""}, [], 1.0, "Unknown") is None


class Sink:
    def __init__(self) -> None:
        self.items: list[object] = []

    def put_nowait(self, obj: object) -> None:
        self.items.append(obj)


class Directory:
    def __init__(self) -> None:
        self.names: dict[str, str] = {}
        self.history: list[tuple[str, str]] = []
        self.by_name: dict[str, list[str]] = {}
        self.voices: dict[str, list[np.ndarray]] = {}
        self.created: list[tuple[str, str]] = []

    def get_user_name(self, user_id: str) -> str:
        return self.names.get(user_id, DEFAULT_NAME)

    def save_chat_history(self, user_id: str, transcript: str) -> None:
        self.history.append((user_id, transcript))

    def get_user_ids_by_name(self, name: str) -> list[str]:
        return self.by_name.get(name, [])

    def get_voice_embeddings_by_uid(self, user_id: str) -> list[np.ndarray]:
        return self.voices.get(user_id, [])

    def create_user(self, user_id: str, name: str) -> None:
        self.created.append((user_id, name))
        self.names[user_id] = name
        self.by_name.setdefault(name, []).append(user_id)

    def save_voice_embedding(self, user_id: str, embedding: np.ndarray) -> None:
        self.voices.setdefault(user_id, []).append(embedding)


def blank_state() -> tuple[Directory, Sink, Sink, Sink, CoordinatorState]:
    directory = Directory()
    llm = Sink()
    vision = Sink()
    audio = Sink()
    state = CoordinatorState(
        db=directory,
        llm_commands=llm,
        vision_commands=vision,
        audio_commands=audio,
        vision_cache=deque(),
        audio_cache=deque(),
        conversation_history=deque(maxlen=10),
        pending_voice_registration=None,
        cache_duration=10.0,
        audio_cache_duration=30.0,
        max_delay=20.0,
        frame_area=640 * 480,
        frame_center_x=320,
        frame_center_y=240,
        max_frame_distance=(320**2 + 240**2) ** 0.5,
        request_number=0,
    )
    return directory, llm, vision, audio, state


def test_speech_handler() -> None:
    directory, llm, _, audio, state = blank_state()
    directory.names["ada"] = "Ada"
    handle_event(
        {"type": "speech", "user_id": "ada", "text": "hello", "time_start": 1.0},
        state,
    )
    assert directory.history == [("ada", "Ada: hello")]
    assert llm.items[0]["cmd"] == "PARSE_INTENT"
    assert "Ada" in llm.items[0]["text"]
    assert audio.items == []

    voice = np.array([1.0, 0.0])
    state.pending_voice_registration = {"user_id": "ada", "timestamp": 10.0}
    handle_event(
        {
            "type": "speech",
            "user_id": DEFAULT_ID,
            "text": "it's me",
            "time_start": 12.0,
            "embedding": voice,
        },
        state,
    )
    assert state.pending_voice_registration is None
    assert audio.items[-1]["cmd"] == "REGISTER_VOICE"
    assert audio.items[-1]["user_id"] == "ada"
    assert directory.history[-1] == ("ada", "Ada: it's me")

    state.pending_voice_registration = {"user_id": "ada", "timestamp": 10.0}
    handle_event(
        {
            "type": "speech",
            "user_id": DEFAULT_ID,
            "text": "later",
            "time_start": 40.0,
            "embedding": voice,
        },
        state,
    )
    assert state.pending_voice_registration == {"user_id": "ada", "timestamp": 10.0}


def test_identity_registration() -> None:
    directory, _, vision, audio, state = blank_state()
    face = {
        "user_id": DEFAULT_ID,
        "track_id": 4,
        "emb": np.array([1.0]),
        "bbox": (300, 220, 340, 260),
    }
    state.vision_cache.append((5.0, [face]))
    voice = np.array([1.0, 0.0])
    handle_event(
        {
            "type": "intent",
            "cmd": "REGISTER_IDENTITY",
            "timestamp": 5.0,
            "voice_embedding": voice,
            "args": {
                "name": "Ada",
                "speaker_name": DEFAULT_NAME,
                "is_self_introduction": True,
            },
        },
        state,
    )
    assert directory.created[0][1] == "Ada"
    assert vision.items[0]["cmd"] == "REGISTER_FACE"
    assert vision.items[0]["track_id"] == 4
    assert audio.items[0]["cmd"] == "REGISTER_VOICE"

    directory.by_name["Bob"] = ["bob-1"]
    directory.names["bob-1"] = "Bob"
    handle_event(
        {
            "type": "intent",
            "cmd": "REGISTER_IDENTITY",
            "timestamp": 6.0,
            "voice_embedding": voice,
            "args": {
                "name": "Bob",
                "speaker_name": USER_NAME,
                "is_self_introduction": False,
            },
        },
        state,
    )
    assert state.pending_voice_registration == {"user_id": "bob-1", "timestamp": 6.0}


def test_audio_and_vision_helpers() -> None:
    pcm = np.array([1, -1], dtype=np.int16).tobytes()
    samples = pcm_from_bytes(pcm)
    assert samples.dtype == np.float32
    assert np.allclose(samples, np.array([1 / 32767.0, -1 / 32767.0]))

    leftover, chunks = take_chunks(b"abcdef", 2)
    assert chunks == [b"ab", b"cd", b"ef"]
    assert leftover == b""
    leftover, chunks = take_chunks(b"abcde", 2)
    assert chunks == [b"ab", b"cd"]
    assert leftover == b"e"

    utterance = Utterance("abc", 1.5, [], 0, None, None)
    assert speech_event("", utterance, np.array([1.0]), "ada") is None
    event = speech_event("hi", utterance, np.array([1.0]), "ada")
    assert event is not None
    assert event["user_id"] == "ada"
    assert event["time_start"] == 1.5

    class Commands:
        def __init__(self, items: list[dict]) -> None:
            self.items = items

        def empty(self) -> bool:
            return not self.items

        def get_nowait(self) -> dict:
            return self.items.pop(0)

    directory = Directory()
    speakers: dict = {}
    voice = np.array([1.0, 0.0])
    apply_voice_commands(
        Commands([{"cmd": "REGISTER_VOICE", "user_id": "ada", "embedding": voice}]),
        directory,
        speakers,
    )
    assert speakers["ada"]["count"] == 1
    assert len(directory.voices["ada"]) == 1

    frames = deque([(0.0, b"a"), (1.0, b"b"), (2.0, b"c"), (3.0, b"d")])
    assert sampled_frames(frames) == [b"a", b"d"]

    class Processor:
        def __init__(self) -> None:
            self.db = directory
            self.registered = None

        def register_identity(self, user_id: str, embedding: np.ndarray) -> None:
            self.registered = (user_id, embedding)

    directory.names["ada"] = "Ada"
    active = {
        3: {
            "user_id": DEFAULT_ID,
            "name": DEFAULT_NAME,
            "score": 0.0,
            "checked_ts": 0.0,
            "last_seen": 0.0,
        }
    }
    embedding = np.array([2.0])
    register_tracked_face(Processor(), active, 3, "ada", embedding, 9.0)
    assert active[3]["name"] == "Ada"
    assert active[3]["score"] == 1.0


def run_tests() -> None:
    test_speaker_matching()
    test_track_updates()
    test_best_identity()
    test_association()
    test_command_events()
    test_speech_handler()
    test_identity_registration()
    test_audio_and_vision_helpers()
    print("Worker helper tests passed")


if __name__ == "__main__":
    run_tests()
