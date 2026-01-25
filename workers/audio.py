from workers.base import BaseWorker

class AudioWorker(BaseWorker):
    def setup(self):
        #prolly load an audio model or sum
        #either that or have some sort of initial ai to measure importance
        print("Audio Worker setting up")

    def process(self, raw_bytes):
        #preprocess incoming data; dk form yet
        pass
