from workers.base import BaseWorker


class VisionWorker(BaseWorker):
    def setup(self, vision_command_queue):
        # load whatever initial vision model u need; keep in mind single core so not gonna be performant
        print("Vision Worker setting up")
        self.command_queue = vision_command_queue

    def process(self, raw_bytes):
        # given raw bytes, we could do several things
        # better if coordinator has a command for us; shared queue of commands?
        # assume command queue top is current task(s)? or do we do top k concurrently

        pass
