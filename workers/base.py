# base worked (blueprint)
# we're doing oop; this is base for vision and audio
# its own little process
# ingests from an input queue stored in a global mem system and does whatever it needs for processing

import multiprocessing as mp
import time


class BaseWorker(mp.Process):
    def __init__(
        self, input_queue, output_queue
    ):  # gotta comm via queue; could look into shared memory or pipe if queue too slow
        super().__init__(daemon=True)
        self.input_queue = input_queue
        self.output_queue = output_queue

    def run(self):
        self.setup()
        while True:
            if not self.input_queue.empty():
                item = self.input_queue.get()
                result = self.process(item)

                if result:
                    self.output_queue.put(result)

            # have a time out to prevent overworking cpu upon task spike
            time.sleep(0.001)

    def process(self, data):
        # to be defined in children class
        raise NotImplementedError

    def setup(self):
        # to be overriden to load any resources
        pass
