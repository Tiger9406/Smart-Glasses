import multiprocessing as mp


class SharedMem:
    def __init__(self):
        self.vision_queue = mp.Queue(maxsize=100)
        self.audio_queue = mp.Queue(maxsize=100)

        self.results_queue = mp.Queue()
