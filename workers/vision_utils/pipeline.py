import asyncio
import os
import queue
import time
from collections import deque
from typing import Any

import cv2
import numpy as np

from core import config
from workers.vision_utils.tracks import ActiveIdentity, FaceTrack, update_tracks


def decode_frame(raw_bytes: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)


def recognize_frame(
    processor: Any,
    active: dict[int, ActiveIdentity],
    frame: np.ndarray,
    *,
    recheck_interval: float,
    lost_after: float,
    confidence_threshold: float,
) -> list[FaceTrack]:
    detected = processor.detect_faces(frame)

    def recognize(face: Any) -> tuple[np.ndarray, str, float]:
        embedding = processor.extract_embedding(frame, face)
        user_id, score = processor.identify_embedding(embedding)
        return embedding, user_id, score

    return update_tracks(
        active,
        detected,
        time.time(),
        recheck_interval=recheck_interval,
        lost_after=lost_after,
        confidence_threshold=confidence_threshold,
        default_id=config.DEFAULT_ID,
        default_name=config.DEFAULT_NAME,
        recognize=recognize,
        name_for=processor.db.get_user_name,
    )


def publish_faces(output_queue: Any, faces: list[FaceTrack]) -> None:
    try:
        output_queue.put({"type": "vision_result", "faces": faces}, block=False)
    except queue.Full:
        print("Queue Full; passing")


def sampled_frames(
    frame_buffer: deque[tuple[float, bytes]], step: int = 3
) -> list[bytes]:
    snapshot = list(frame_buffer)
    return [data for _, data in snapshot[::step]]


def register_tracked_face(
    processor: Any,
    active: dict[int, ActiveIdentity],
    track_id: int,
    user_id: str,
    embedding: np.ndarray,
    now: float,
) -> None:
    processor.register_identity(user_id, embedding)
    name = processor.db.get_user_name(user_id)
    print(f"[Vision] Registered '{name}' using provided embedding.")
    if track_id in active:
        identity = active[track_id]
        identity["user_id"] = user_id
        identity["name"] = name
        identity["score"] = 1.0
        identity["checked_ts"] = now


async def ask_vlm(
    client: Any, output_queue: Any, frames: list[bytes], prompt: str, request_id: int
) -> None:
    try:
        response_text = await client.analyze_video_frames(frames, prompt)
        output_queue.put(
            {
                "type": "vlm_result",
                "request_id": request_id,
                "text": response_text,
                "timestamp": time.time(),
            }
        )
    except Exception as error:
        print(f"[Vision] VLM task error: {error}")


def apply_vision_commands(
    command_queue: Any,
    *,
    vlm_active: bool,
    frame_buffer: deque[tuple[float, bytes]],
    loop: asyncio.AbstractEventLoop,
    client: Any,
    output_queue: Any,
    processor: Any,
    active: dict[int, ActiveIdentity],
) -> None:
    while not command_queue.empty():
        try:
            command = command_queue.get_nowait()
            cmd = command.get("cmd")
            # subsample recent frames and ask the vlm
            if cmd == "GET_VIDEO_CONTEXT":
                if not vlm_active:
                    print("[Vision] VLM configed to be inactive")
                    return
                request_video_context(command, frame_buffer, loop, client, output_queue)
            # store the embedding and rename the live track
            elif cmd == "REGISTER_FACE":
                track_id = command.get("track_id")
                user_id = command.get("user_id")
                embedding = command.get("emb")
                if track_id is not None and user_id and embedding is not None:
                    register_tracked_face(
                        processor, active, track_id, user_id, embedding, time.time()
                    )
        except Exception as error:
            print(f"[Vision] Command error: {error}")


def request_video_context(
    command: dict[str, Any],
    frame_buffer: deque[tuple[float, bytes]],
    loop: asyncio.AbstractEventLoop,
    client: Any,
    output_queue: Any,
) -> None:
    if not frame_buffer:
        print("[Vision] Can't analyze context because buffer empty")
        return
    asyncio.run_coroutine_threadsafe(
        ask_vlm(
            client,
            output_queue,
            sampled_frames(frame_buffer),
            command["prompt"],
            command["request_id"],
        ),
        loop,
    )


def draw_face_label(
    frame: np.ndarray, bbox: tuple[int, int, int, int], text: str
) -> None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    text_y_start = max(y1 - 20, 0)
    cv2.rectangle(
        frame, (x1, text_y_start), (x1 + text_w, text_y_start + 20), (0, 255, 0), -1
    )
    cv2.putText(
        frame,
        text,
        (x1, max(y1 - 5, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def render_annotated_video(
    archive: list[tuple[float, bytes, list[FaceTrack]]],
    output_path: str,
    fps: float,
) -> None:
    print(f"[Vision] Rendering {len(archive)} frames to disk. This may take a moment...")
    if not archive:
        return

    first_time, first_raw, _ = archive[0]
    first_frame = decode_frame(first_raw)
    if first_frame is None:
        return

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height)
    )
    print(f"[Vision] VideoWriter initialized: {output_path} ({width}x{height} @ {fps}fps)")

    frame_duration = 1.0 / fps
    expected_time = first_time
    for timestamp, raw_bytes, faces in archive:
        frame = decode_frame(raw_bytes)
        if frame is None:
            continue
        for face in faces:
            draw_face_label(
                frame, face["bbox"], f"{face['name']} (ID: {face['track_id']})"
            )
        # write the frame once, and repeat it if we skipped a slot
        while expected_time <= timestamp:
            writer.write(frame)
            expected_time += frame_duration

    writer.release()
    print("[Vision] Offline video rendering complete.")


def shutdown_vlm(
    loop: asyncio.AbstractEventLoop, client: Any, thread: Any
) -> None:
    if loop.is_running():
        future = asyncio.run_coroutine_threadsafe(client.close(), loop)
        try:
            future.result(timeout=5)
        except Exception as error:
            print(f"[Vision] Error closing VLM client: {error}")
        loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=1)
