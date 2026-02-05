# coordinator: looks at global queue and processes output from sub workers vision and audio
# kinda like the decision making part

import multiprocessing as mp
import time
import queue
from workers.base import BaseWorker


class Coordinator(BaseWorker):
    def __init__(self, results_queue: mp.Queue):
        super().__init__()
        self.results_queue = results_queue
        # self.audio_events
        # self.vision_events

        # maybe more; hold past actions taken by self maybe? or a state, like what's happening in the world rn?
        # again, decision making module given the initial processing by the workers

    def run(self):
        print("[Coordinator] Started")
        try:
            while self.running.is_set():
                try:
                    event = self.results_queue.get(timeout=0.1)
                    self._handle_event(event)
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    break
        finally:
            print("[Coordinator] Shutting down")

    def _handle_event(self, event):
        # handling events; gotta coordinate event data format
        # for instance if event type is a face in view, we throw it on the picture or sum

        event_type = event.get("type", "unknown")

        if event_type == "vision_result":
            faces = event.get("face", [])
            if faces:
                print(f"\n [Coordinator] Vision Event: detected {len(faces)} faces")
                for face in faces:
                    name = face.get("name", "Unknown")
                    score = face.get("score", 0.0)
                    bbox = face.get("bbox")
                    print(f" - ID: {face['track_id']} | Name: {name} ({score:.2f}) | Loc: {bbox}")

        else:
            print("\n[Coordinator] got other event")

        return
