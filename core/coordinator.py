# coordinator: looks at global queue and processes output from sub workers vision and audio
# kinda like the decision making part

import multiprocessing as mp
import time


class Coordinator(mp.Process):
    def __init__(self, results_queue):
        super().__init__(daemon=True)
        self.results_queue = results_queue
        self.past_events = {}
        # self.audio_events
        # self.vision_events

        # maybe more; hold past actions taken by self maybe? or a state, like what's happening in the world rn?
        # again, decision making module given the initial processing by the workers

    def run(self):
        while True:
            if not self.results_queue.empty():
                event = self.results_queue.get()
                self._handle_event(event)
            time.sleep(0.001)

    def _handle_event(self, event):
        # handling events; gotta coordinate event data format
        # for instance if event type is a face in view, we throw it on the picture or sum
        return
