from workers.base import IngestionWorker
import queue


class AudioWorker(IngestionWorker):
    def setup(self):
        # prolly load an audio model or sum
        # either that or have some sort of initial ai to measure importance
        print("Audio Worker setting up")

    def run(self):
        try:
            while self.running.is_set():
                try:
                    raw_bytes = self.input_queue.get(timeout=0.01)
                    self.process(raw_bytes)
                except queue.Empty:
                    continue
        finally:
            print("[Audio] Releasing resources")

    def process(self, raw_bytes):
        # preprocess incoming data; dk form yet
        print(f"[Audio] worker received & processing audio file of size {len(raw_bytes)}")
        pass
