#entry point

import uvicorn
import multiprocessing as mp

from fastapi import FastAPI
from contextlib import asynccontextmanager

from api.routes import setup_routes

from core.shared_mem import SharedMem
from core.coordinator import Coordinator
from workers.audio import AudioWorker
from workers.vision import VisionWorker

# defines lifespan; handles startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):

    shared_mem = SharedMem() #shared mem is 3 queues; again can look into shared_mem or direct pipies if this too slow

    app.state.system=shared_mem

    brain = Coordinator(shared_mem.results_queue)
    audio_worker = AudioWorker(shared_mem.audio_queue, shared_mem.results_queue)
    vision_worker = VisionWorker(shared_mem.vision_queue, shared_mem.results_queue)

    brain.start()
    audio_worker.start()
    vision_worker.start()

    yield #app running after this

    print("Cleaning resources")

    brain.terminate()
    audio_worker.terminate()
    vision_worker.terminate()


def start_server():
    app=FastAPI(lifespan=lifespan)

    #define routes; right now only websocket streamed in
    setup_routes(app)
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
