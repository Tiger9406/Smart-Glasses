import sys
import threading
import time

import numpy as np

from api.udp_receiver import udp_receiver_thread, udp_shutdown_event
from core import config
from core.shared_mem import SharedMem
from database.database import DatabaseManager
from workers.api_worker import APIWorker
from workers.audio import AudioWorker
from workers.coordinator import Coordinator
from workers.vision import VisionWorker


def main():
    shared_mem = SharedMem()
    db = DatabaseManager()
    if config.START_WITH_SAMPLE_DATA:
        for name, emb_path in config.SAMPLE_FACE_EMBEDDING_PATHS.items():
            db.save_face_embedding(name, np.load(emb_path))
        for name, emb_paths in config.SAMPLE_VOICE_EMBEDDING_PATHS.items():
            for emb_path in emb_paths:
                db.save_voice_embedding(name, np.load(emb_path))

    brain = Coordinator(
        shared_mem.results_queue,
        shared_mem.gemini_command_queue,
        shared_mem.vision_command_queue,
        shared_mem.audio_command_queue,
    )
    audio_worker = AudioWorker(
        shared_mem.audio_queue, shared_mem.results_queue, shared_mem.audio_command_queue
    )
    vision_worker = VisionWorker(
        shared_mem.vision_queue,
        shared_mem.results_queue,
        shared_mem.vision_command_queue,
    )

    api_worker = APIWorker(shared_mem.gemini_command_queue, shared_mem.results_queue)

    print("[System] Starting background workers")

    brain.start()
    audio_worker.start()
    vision_worker.start()
    api_worker.start()

    # udp ingestion to be a thread since inherently stateless and pretty low resource
    udp_shutdown_event.clear()
    udp_thread = threading.Thread(
        target=udp_receiver_thread,
        args=(shared_mem,),
        daemon=True,
        name="UDPReceiverThread",
    )
    udp_thread.start()

    print("\n[System] All systems started; waiting for stream; press ctrl + C to stop")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[System] Shutdown from user; cleaning up resources")
    finally:
        print("[System] Stopping UDP thread")
        udp_shutdown_event.set()
        udp_thread.join(timeout=2.0)

        if config.START_WITH_SAMPLE_DATA:
            print("[System] Clearing Sample Data")
            db.clear_db()

        print("[System] Shutting down workers")
        workers = [audio_worker, vision_worker, brain, api_worker]
        for w in workers:
            w.shutdown()

        # queues not empty when we stop it; instead we just stop it from joining & chuck away data
        print("[System] Shutting down queues")
        shared_mem.shutdown()

        print("[System] Waiting for workers to join...")
        for w in workers:
            w.join(timeout=1.0)
            if w.is_alive():
                print(f"Force killing {w.name}...")
                w.terminate()
                w.join()

        print("[System] All workers stopped")
        sys.exit(0)


if __name__ == "__main__":
    main()
