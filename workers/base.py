# base worked (blueprint)
# we're doing oop; this is base for vision and audio
# its own little process
# ingests from an input queue stored in a global mem system and does whatever it needs for processing

import multiprocessing as mp

class BaseWorker(mp.Process):
    def __init__(self):  # gotta comm via queue; could look into shared memory or pipe if queue too slow
        super().__init__(daemon=True)
        self.running = mp.Event()
        self.running.set()

    def run(self):
        raise NotImplementedError

    def shutdown(self):
        self.running.clear()


class IngestionWorker(BaseWorker):
    def __init__(self, input_queue, output_queue):
        super().__init__()
        self.input_queue=input_queue
        self.output_queue = output_queue