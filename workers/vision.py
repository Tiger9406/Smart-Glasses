import multiprocessing as mp
import time

import cv2
import numpy as np

from workers.vision_utils.facial_processing.inspireface_processor import (
    InspireFaceProcessor,
)
from workers.vision_utils.facial_processing.tracking import FaceTracker


class VisionWorker(mp.Process):
    def __init__(
        self,
        input_queue: mp.Queue,
        output_queue: mp.Queue,
        vision_command_queue: mp.Queue,
    ):
        super().__init__(daemon=True)
        self.input_queue = input_queue
        self.output_queue = output_queue

        print("[Vision] Worker setting up")
        self.command_queue = vision_command_queue
        self.tracker = FaceTracker(iou_threshold=0.3, untracked_before_delete=10)
        self.processor = InspireFaceProcessor(
            model_path="Megatron", confidence_threshold=0.5, download_model=False
        )
        print("[Vision] Ready")

    def run(self):
        while True:
            if not self.input_queue.empty():
                item = self.input_queue.get()
                result = self.process(item)

                if result:
                    self.output_queue.put(result)

            # have a time out to prevent overworking cpu upon task spike
            time.sleep(0.001)

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

        raw_detection_faces = self.processor.detect_faces(frame)
        if not raw_detection_faces:
            return

        # use newly detected faces to update current holding
        bboxes = [tuple(map(int, face.location)) for face in raw_detection_faces]
        current_tracks = self.tracker.update(bboxes)

        for i, track in enumerate(current_tracks):
            raw_face_obj = raw_detection_faces[i]
            should_identify = False
            if track.identity == "Unkown":
                should_identify = True

            elif track.frames_since_recognition > 15:
                should_identify = True

            elif not track.is_confirmed:
                should_identify = True

            if should_identify:
                embedding = self.processor.extract_embedding(frame, raw_face_obj)
                name, score = self.processor.identify_embedding(embedding)

                # TODO: deal with given score and names

        commands = self._get_active_commands()

        # we have frame & commands; do basic stuff and depending on commands we'll do further processing
        # basic: maybe yolo & major change measure?

        # returns result; but what form?
