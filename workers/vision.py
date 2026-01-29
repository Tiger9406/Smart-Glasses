from workers.base import BaseWorker

class VisionWorker(BaseWorker):
    def setup(self):
        #load whatever initial vision model u need; keep in mind single core so not gonna be performant
        print("Vision Worker setting up")

    def process(self, raw_bytes):
        #preprocess incoming data; dk form yet
        print(f"vision worker received & processing frame of size {len(raw_bytes)}")
        pass
