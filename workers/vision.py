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
        self.tracker = FaceTracker(iou_threshold=0.3, untracked_before_delete=20)
        self.processor = InspireFaceProcessor(
            model_path="Megatron", confidence_threshold=0.5, download_model=False
        )
        self.RECOGNITION_INTERVAL = 30  # re-check identity every 30 frames
        self.CONFIDENCE_THRESHOLD = 0.60  # Score to confirm is someone
        self.UNLOCK_THRESHOLD = (
            0.40  # average score thereafter to maintain identity
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

            # if unknown, too long since we last recognized, 
            # or unconfirmed, we should do identification
            should_identify = (
                track.identity == "Unkown"
                or track.frames_since_recognition > self.RECOGNITION_INTERVAL
                or not track.is_confirmed
            )

            if should_identify:
                embedding = self.processor.extract_embedding(frame, raw_face_obj)

                if track.is_confirmed:
                    # we only care about score against the locked identity
                    score = self.processor.compare_to_person(track.identity)
                    track.update_identity_score(score)

                    if score < 0.01 or track.average_score < self.UNLOCK_THRESHOLD:
                        print(f"[Vision] Lost lock on {track.identity} (avg: {track.average_score:..2f})")
                        track.update_identity() #empty to reset
                
                #unknown / still looking for match
                else:
                    name, score = self.processor.identify_embedding(embedding)
                    if score > self.CONFIDENCE_THRESHOLD:
                        track.update_identity(name, True, score)
                    else: # still unknown
                        pass

        commands = self._get_active_commands()

        # we have frame & commands; do basic stuff and depending on commands we'll do further processing
        # basic: maybe yolo & major change measure?

        # returns result; but what form?
