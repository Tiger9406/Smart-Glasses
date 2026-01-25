#entry point

import uvicorn
import multiprocessing as mp

import asyncio
from fastapi import FastAPI
from contextlib import asynccontextmanager

from core.shared_mem import SharedMem
from core.coordinator import Coordinator
from workers.audio import AudioWorker
from workers.vision import VisionWorker

# defines lifespan; handles startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize resources here
    # shared mem
    shared_mem = SharedMem()
    brain = Coordinator(shared_mem.results_queue)
    audio_worker = AudioWorker(shared_mem.audio_queue)
    vision_worker = VisionWorker(shared_mem.vision_queue)

    brain.start()
    audio_worker.start()
    vision_worker.start()

    yield #app running after this

    brain.terminate()
    audio_worker.terminate()
    vision_worker.terminate()


def start_server():
    app=FastAPI(lifespan=lifespan)

    #gotta define the endpoints for receiving & publishing results here

    return app



app=start_server()
if __name__ == "__main__":
    #start subprocesses & then start server
    try:
        uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="error")
    except KeyboardInterrupt:
        print("Shutting down server...")
        mp.current_process().terminate()
    finally:
        print("Server has been shut down.")
