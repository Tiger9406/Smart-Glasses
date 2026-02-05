import multiprocessing as mp
import queue
import time

import cv2
import numpy as np

from workers.vision_utils.facial_processing.inspireface_processor import (
    InspireFaceProcessor,
)
from workers.base import IngestionWorker


class VisionWorker(IngestionWorker):
    def __init__(
        self,
        input_queue: mp.Queue,
        output_queue: mp.Queue,
        vision_command_queue: mp.Queue,
    ):
        super().__init__(input_queue, output_queue)
        self.command_queue = vision_command_queue

    def setup(self):
        print("[Vision] Worker setting up")
        self.processor = InspireFaceProcessor()
        self.processor.session.set_track_lost_recovery_mode(True)
        self.active_identities = {}
        self.RECHECK_INTERVAL = 2.0  # seconds between re-verifying identification
        print("[Vision] Ready")

    def run(self):
        self.setup()

        try:
            while self.running.is_set():
                try:
                    raw_bytes = self.input_queue.get(timeout=0.01)
                except queue.Empty:
                    continue
                frame = cv2.imdecode(
                    np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR
                )
                if frame is None:
                    continue

                raw_detection_faces = self.processor.detect_faces(frame)

                result = []
                current_frame_ids = set()

                for face in raw_detection_faces:
                    track_id = face.track_id
                    current_frame_ids.add(track_id)

                    if (
                        track_id not in self.active_identities
                    ):  # check if we know this dude
                        self.active_identities[track_id] = {
                            "name": "Unknown",
                            "score": 0.0,
                            "checked_ts": 0,
                        }

                    # get our stored data on this guy
                    identity_data = self.active_identities[track_id]

                    now = time.time()
                    should_recognize = (
                        identity_data["name"] == "Unknown"
                        or (now - identity_data["checked_ts"]) > self.RECHECK_INTERVAL
                    )

                    if should_recognize:
                        emb = self.processor.extract_embedding(frame, face)
                        name, score = self.processor.identify_embedding(emb)

                        if score > 0.6:
                            self.active_identities[track_id] = {
                                "name": name,
                                "score": score,
                                "checked_ts": now,
                            }
                        else:
                            self.active_identities[track_id]["checked_ts"] = now

                    x, y, w, h = map(int, face.location)
                    result.append(
                        {
                            "track_id": track_id,
                            "bbox": (x, y, w, h),
                            "name": self.active_identities[track_id]["name"],
                            "score": self.active_identities[track_id]["score"],
                        }
                    )

                expired_ids = [
                    track_id
                    for track_id in self.active_identities
                    if track_id not in current_frame_ids
                ]
                for track_id in expired_ids:
                    del self.active_identities[track_id]

                try:
                    self.output_queue.put({"type": "vision_result", "faces": result})
                    print("[Vision] added to ouput queue")
                except queue.Full:
                    print("Queue Full; passing")
                    pass

        finally:
            print("[Vision] Releasing resources")
            if hasattr(self, "processor") and self.processor.session:
                self.processor.session.release()


    def _get_active_commands(self) -> list:
        commands = []
        while not self.command_queue.empty():
            commands.append(self.command_queue.get_nowait())

        # deal with registering face for instance
        return commands

        # we have frame & commands; do basic stuff and depending on commands we'll do further processing
        # basic: maybe yolo & major change measure?

        # returns result; but what form?
