import multiprocessing as mp


class SharedMem:
    def __init__(self):
        self.vision_queue = mp.Queue(maxsize=100)
        self.audio_queue = mp.Queue(maxsize=100)

        self.results_queue = mp.Queue()
        self.vision_command_queue = mp.Queue()

    def shutdown(self):
        self.vision_queue.cancel_join_thread() 
        self.audio_queue.cancel_join_thread() 
        self.results_queue.cancel_join_thread() 
        self.vision_command_queue.cancel_join_thread() 
        self.vision_queue.close() 
        self.audio_queue.close() 
        self.results_queue.close() 
        self.vision_command_queue.close() 
        print("[Shared Mem] Queues closed")
