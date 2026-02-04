import multiprocessing as mp

import cv2
import numpy as np

from workers.base import BaseWorker


class VisionWorker(BaseWorker):
    def setup(self, vision_command_queue: mp.Queue):
        # load whatever initial vision model u need; keep in mind single core so not gonna be performant
        print("Vision Worker setting up")
        self.command_queue = vision_command_queue
        self.objects_tracking = []
        self.faces_tracking = []

    def _decode_image(self, raw_bytes):
        # turn byte sinto numpy array
        np_arr = np.frombuffer(raw_bytes, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def _get_active_commands(self) -> list:
        if self.command_queue.empty():
            return []

        commands = []
        while not self.command_queue.empty():
            commands.append(self.command_queue.get_nowait())
        return commands

    def process(self, raw_bytes):
        # given raw bytes, we could do several things
        # better if coordinator has a command for us; shared queue of commands?
        # assume command queue top is current task(s)? or do we do top k concurrently
        if not raw_bytes:
            return

        frame = self._decode_image(self, raw_bytes)

        if frame is None:
            return

        commands = self._get_active_commands()

        # we have frame & commands; do basic stuff and depending on commands we'll do further processing
        # basic: maybe yolo & major change measure?

        # returns result; but what form?
