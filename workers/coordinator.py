# coordinator: looks at global queue and processes output from sub workers vision and audio
# kinda like the decision making part

import multiprocessing as mp
import queue
import time
import inspireface as isf
import numpy as np

from core.config import TEST_REGISTER_IDENTITY, VLM_ACTIVE, SAMPLE_EMBEDDING_PATHS
from workers.base import BaseWorker


class Coordinator(BaseWorker):
    def __init__(self, results_queue: mp.Queue, commands_queue: mp.Queue):
        super().__init__()
        self.results_queue = results_queue
        self.commands_queue = commands_queue
        # self.audio_events
        # self.vision_events

        # maybe more; hold past actions taken by self maybe? or a state, like what's happening in the world rn?
        # again, decision making module given the initial processing by the workers

    def setup(self):
        self.start_time = time.time()
        self.request_number = 0

    def run(self):
        print("[Coordinator] Started")
        self.setup()

        try:
            while self.running.is_set():
                try:
                    event = self.results_queue.get(timeout=0.1)
                    self._handle_event(event)
                    self._test_VLM()
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    break
        finally:
            print("[Coordinator] Shutting down")

    def _test_VLM(self):
        # comment return statement to test VLM funcitonality
        if not VLM_ACTIVE:
            return
        if time.time() - self.start_time > 5 and self.request_number == 0:
            command = {
                "cmd": "GET_VIDEO_CONTEXT",
                "prompt": "Summarize the video in a sentence",
                "request_id": self.request_number,
            }
            self.request_number += 1
            print("[Coordinator] Put command in vision queue")
            self.commands_queue.put_nowait(command)

    def _test_register_identity(self, event: list[dict]):
        if not TEST_REGISTER_IDENTITY:
            return
        faces = event.get("faces", [])
        for name, emb_path in SAMPLE_EMBEDDING_PATHS.items():
            emb = np.load(emb_path)
            for face in faces:
                new_emb = face.get('emb', None)
                if new_emb is None:
                    continue
                score = isf.feature_comparison(emb, new_emb)
                if score > 0.5:
                    command = {
                        "cmd": "REGISTER_FACE",
                        "track_id": face.get("track_id"), 
                        "name": name, 
                        "emb": new_emb
                    }
                    self.request_number+=1
                    print(f"[Coordinator] Put command to register {name} in vision queue")
                    self.commands_queue.put_nowait(command)

    def _handle_event(self, event):
        # handling events; gotta coordinate event data format
        # for instance if event type is a face in view, we throw it on the picture or sum

        event_type = event.get("type", "unknown")

        if event_type == "vision_result":
            # given list of the following:
            """{
                "track_id": track_id,
                "bbox": (x1, y1, x2, y2),
                "name": self.active_identities[track_id]["name"],
                "score": self.active_identities[track_id]["score"],
                "emb": emb # could be none if embedding not extracted on this frame
            }"""

            # potential code to work with input faces; nothing for now, too many frames
            """
            faces = event.get("faces", [])
            if faces:
                print(f"\n [Coordinator] Vision Event: detected {len(faces)} faces")
                for face in faces:
                    _name = face.get("name", DEFAULT_NAME)
                    _score = face.get("score", 0.0)
                    _bbox = face.get("bbox")
                    # print(f" - ID: {face['track_id']} | Name: {name} ({score:.2f}) | Loc: {bbox}")

            """
            self._test_register_identity(event)
            pass

        elif event_type == "speech":
            # given:
            """
            "type": "speech",
            "text": text,    (would be the audio transcription)
            "id": session_id,
            "timestamp": time.time(),
            "final": False,
            "name": Unkown,
            "embedding":
            """
            print(f"[Coordinator] {event['name']}: {event['text']}")

        elif event_type == "vlm_result":
            """"
            "type": "vlm_result",
            "request_id": request_id,
            "text": response_text,
            "timestamp": time.time(),
            """
            print(f"[Coordinator] Received VLM output: {event['text']}")

        else:
            print("\n[Coordinator] got other event")

        return
