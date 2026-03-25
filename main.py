import sys
import threading
import time
import os
import subprocess
import imageio_ffmpeg

import numpy as np
import uuid

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
        sample_uuids = {}
        for name, emb_path in config.SAMPLE_FACE_EMBEDDING_PATHS.items():
            if name not in sample_uuids:
                sample_uuids[name] = str(uuid.uuid4())
                db.create_user(sample_uuids[name], name)
            db.save_face_embedding(sample_uuids[name], np.load(emb_path))
            
        # Load sample voices
        for name, emb_paths in config.SAMPLE_VOICE_EMBEDDING_PATHS.items():
            if name not in sample_uuids:
                sample_uuids[name] = str(uuid.uuid4())
                db.create_user(sample_uuids[name], name)
            for emb_path in emb_paths:
                db.save_voice_embedding(sample_uuids[name], np.load(emb_path))

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

        video_path = config.VIDEO_OUTPUT_PATH 
        audio_path = config.AUDIO_OUTPUT_PATH
        final_output_path = config.COMBINED_OUTPUT_PATH

        if os.path.exists(video_path) and os.path.exists(audio_path):
            try:
                # Get the path to the downloaded ffmpeg binary
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                
                command = [
                    ffmpeg_exe, "-y",            
                    "-i", video_path,            
                    "-i", audio_path,            
                    "-c:v", "copy",              
                    "-c:a", "aac",               
                    final_output_path
                ]
                
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"[System] Merge complete: {final_output_path}")
                
            except Exception as e:
                print(f"[System] FFmpeg merge failed: {e}")
        else:
            print("[System] Skipped merge: Missing audio or video file.")

        sys.exit(0)


if __name__ == "__main__":
    main()
