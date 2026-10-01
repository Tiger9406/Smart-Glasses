import asyncio
import multiprocessing as mp
import queue
import threading
import time
from collections import deque

from api.openai_client import OpenAIClient
from core import config, config_vision
from core.log_interceptor import install_log_interceptor
from workers.base import IngestionWorker
from workers.vision_utils.inspireface_processor import InspireFaceProcessor
from workers.vision_utils.pipeline import (
    apply_vision_commands,
    decode_frame,
    publish_faces,
    recognize_frame,
    render_annotated_video,
    shutdown_vlm,
)
from workers.vision_utils.tracks import ActiveIdentity, FaceTrack


class VisionWorker(IngestionWorker):
    command_queue: mp.Queue
    processor: InspireFaceProcessor
    active_identities: dict[int, ActiveIdentity]
    recheck_interval: float
    confidence_threshold: float
    lost_track_threshold: float
    buffer_len: int
    frame_buffer: deque[tuple[float, bytes]]
    vlm_client: OpenAIClient
    loop: asyncio.AbstractEventLoop
    async_thread: threading.Thread
    frame_archive: list[tuple[float, bytes, list[FaceTrack]]]

    def __init__(
        self,
        input_queue: mp.Queue,
        output_queue: mp.Queue,
        vision_command_queue: mp.Queue,
        log_queue: mp.Queue,
    ) -> None:
        super().__init__(input_queue, output_queue, log_queue=log_queue)
        self.command_queue = vision_command_queue

    def setup(self) -> None:
        print("[Vision] Worker setting up")
        self.processor = InspireFaceProcessor()
        self.processor.session.set_track_lost_recovery_mode(True)
        self.active_identities = {}
        self.recheck_interval = 2.0
        self.confidence_threshold = 0.5
        self.lost_track_threshold = 1.0

        self.buffer_len = int(config_vision.FPS * config_vision.BUFFER_DURATION)
        self.frame_buffer = deque(maxlen=self.buffer_len)
        self.vlm_client = OpenAIClient()
        self.loop = asyncio.new_event_loop()
        self.async_thread = threading.Thread(
            target=self._start_background_loop, daemon=True
        )
        self.async_thread.start()
        self.frame_archive = []
        print("[Vision] Ready")

    def _start_background_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self) -> None:
        install_log_interceptor(self.log_queue, "[Vision]")
        self.setup()

        try:
            while self.running.is_set():
                # commands first: register a face, or ask the vlm about recent frames
                apply_vision_commands(
                    self.command_queue,
                    vlm_active=config_vision.VLM_ACTIVE,
                    frame_buffer=self.frame_buffer,
                    loop=self.loop,
                    client=self.vlm_client,
                    output_queue=self.output_queue,
                    processor=self.processor,
                    active=self.active_identities,
                )
                raw_bytes = next_frame(self.input_queue)
                if raw_bytes is None:
                    continue

                now = time.time()
                self.frame_buffer.append((now, raw_bytes))
                frame = decode_frame(raw_bytes)
                if frame is None:
                    continue

                # figure out who is in this frame and how long we've been seeing them
                faces = recognize_frame(
                    self.processor,
                    self.active_identities,
                    frame,
                    recheck_interval=self.recheck_interval,
                    lost_after=self.lost_track_threshold,
                    confidence_threshold=self.confidence_threshold,
                )
                # send the faces onward, and keep the frame if we're recording
                publish_faces(self.output_queue, faces)
                if config.SAVE_ANNOTATED_VID:
                    self.frame_archive.append((now, raw_bytes, faces))
        finally:
            print("[Vision] Releasing resources")
            self.processor.session.release()
            # burn the boxes into a video once the stream is done
            if config.SAVE_ANNOTATED_VID and self.frame_archive:
                render_annotated_video(
                    self.frame_archive, config.VIDEO_OUTPUT_PATH, config_vision.FPS
                )
                print("[Vision] VideoWriter released")
            shutdown_vlm(self.loop, self.vlm_client, self.async_thread)


def next_frame(input_queue: mp.Queue) -> bytes | None:
    try:
        return input_queue.get(timeout=0.01)
    except queue.Empty:
        return None
