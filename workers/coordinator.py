# coordinator: looks at global queue and processes output from sub workers vision and audio
# kinda like the decision making part

import multiprocessing as mp
import queue
from collections import deque

import numpy as np

from core import config
from core.log_interceptor import install_log_interceptor
from database.database import DatabaseManager
from workers.base import BaseWorker
from workers.coordinator_utils.handlers import CoordinatorState, handle_event


class Coordinator(BaseWorker):
    events_queue: mp.Queue
    state: CoordinatorState

    def __init__(
        self,
        results_queue: mp.Queue,
        llm_commands_queue: mp.Queue,
        vision_commands_queue: mp.Queue,
        audio_commands_queue: mp.Queue,
        log_queue: mp.Queue,
    ) -> None:
        super().__init__(log_queue=log_queue)
        self.events_queue = results_queue
        self.llm_commands_queue = llm_commands_queue
        self.vision_commands_queue = vision_commands_queue
        self.audio_commands_queue = audio_commands_queue
        self.db = DatabaseManager()

    def setup(self) -> None:
        frame_center_x = config.RESOLUTION[0] / 2
        frame_center_y = config.RESOLUTION[1] / 2
        self.state = CoordinatorState(
            db=self.db,
            llm_commands=self.llm_commands_queue,
            vision_commands=self.vision_commands_queue,
            audio_commands=self.audio_commands_queue,
            vision_cache=deque(),
            audio_cache=deque(),
            conversation_history=deque(maxlen=10),
            pending_voice_registration=None,
            cache_duration=10.0,
            audio_cache_duration=30.0,
            max_delay=20.0,
            frame_area=config.RESOLUTION[0] * config.RESOLUTION[1],
            frame_center_x=frame_center_x,
            frame_center_y=frame_center_y,
            max_frame_distance=float(
                np.sqrt(frame_center_x**2 + frame_center_y**2)
            ),
            request_number=0,
        )

    def run(self) -> None:
        install_log_interceptor(self.log_queue, "[Coordinator]")
        print("[Coordinator] Started")
        self.setup()

        try:
            while self.running.is_set():
                try:
                    event = self.events_queue.get(timeout=0.1)
                    # vision, speech, and llm results all land here
                    handle_event(event, self.state)
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    break
        finally:
            print("[Coordinator] Shutting down")
