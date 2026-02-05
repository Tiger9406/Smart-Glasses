import multiprocessing as mp
import os
import queue
import time

import cv2
import numpy as np

from workers.base import IngestionWorker
from workers.vision_utils.facial_processing.inspireface_processor import (
    InspireFaceProcessor,
)


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
        self.video_writer = None
        self.active_identities = {}
        self.RECHECK_INTERVAL = 2.0  # seconds between re-verifying identification
        self.CONFIDENCE_THRESHOLD = 0.5
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

                # for testing purposes: if we wanna see bounding box behavior
                # if self.video_writer is None:
                #     self._init_video_writer(frame)

                raw_detection_faces = self.processor.detect_faces(frame)

                result = []
                current_frame_ids = set()

                for face in raw_detection_faces:
                    track_id = face.track_id
                    current_frame_ids.add(track_id)

                    if (
                        track_id not in self.active_identities
                    ):  # new box; not previously tracked
                        self.active_identities[track_id] = {
                            "name": "Unknown",
                            "score": 0.0,
                            "checked_ts": 0,
                        }

                    # get our stored data on this guy
                    identity_data = self.active_identities[track_id]

                    # determine if we should try to identify him (compare to known people)
                    now = time.time()

                    emb = None

                    should_recognize = (
                        identity_data["name"] == "Unknown"
                        or (now - identity_data["checked_ts"]) > self.RECHECK_INTERVAL
                    )
                    if should_recognize:
                        emb = self.processor.extract_embedding(frame, face)
                        name, score = self.processor.identify_embedding(emb)

                        # if strongly looks like someone we know
                        if score > self.CONFIDENCE_THRESHOLD:
                            self.active_identities[track_id] = {
                                "name": name,
                                "score": score,
                                "checked_ts": now,
                            }
                        else:  # still don't know
                            self.active_identities[track_id]["checked_ts"] = now

                    # form result to send back to coordinator
                    x1, y1, x2, y2 = map(int, face.location)
                    result.append(
                        {
                            "track_id": track_id,
                            "bbox": (x1, y1, x2, y2),
                            "name": self.active_identities[track_id]["name"],
                            "score": self.active_identities[track_id]["score"],
                            "emb": emb,
                        }
                    )

                    if self.video_writer:
                        current_name = self.active_identities[track_id]["name"]
                        label_text = f"{current_name} (ID: {track_id})"
                        self._draw_face_label(frame, (x1, y1, x2, y2), label_text)

                if self.video_writer:
                    self.video_writer.write(frame)

                # remove expired ids (untracked for a while)
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
            if self.video_writer:
                self.video_writer.release()
                print("[Vision] VideoWriter released")

    def _get_active_commands(self) -> list:
        commands = []
        while not self.command_queue.empty():
            commands.append(self.command_queue.get_nowait())

        # deal with registering face for instance
        return commands

        # we have frame & commands; do basic stuff and depending on commands we'll do further processing
        # basic: maybe yolo & major change measure?

        # returns result; but what form?

    def _init_video_writer(
        self, frame, output_path="workers/vision_utils/annotated_video.mp4", fps=15.0
    ):
        """Initializes the VideoWriter based on the first frame's dimensions."""
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        print(f"[Vision] VideoWriter initialized: {output_path} ({w}x{h} @ {fps}fps)")

    def _draw_face_label(self, frame, bbox, text):
        # Unpack the coordinates directly
        x1, y1, x2, y2 = bbox

        # 1. Draw Bounding Box
        # (x1, y1) is top-left, (x2, y2) is bottom-right
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 2. Draw Label Background
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        )

        # Ensure label doesn't go off-screen at the top
        text_y_start = max(y1 - 20, 0)

        # Draw filled box for text
        cv2.rectangle(
            frame, (x1, text_y_start), (x1 + text_w, text_y_start + 20), (0, 255, 0), -1
        )

        # 3. Draw Text
        text_y = max(y1 - 5, 15)
        cv2.putText(
            frame,
            text,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
