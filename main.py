# entry point

import multiprocessing as mp
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from api.routes import setup_routes
from core import config

from core.shared_mem import SharedMem
from workers.audio import AudioWorker
from workers.vision import VisionWorker
from workers.coordinator import Coordinator


# defines lifespan; handles startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    shared_mem = SharedMem()  # shared mem is 3 queues; again can look into shared_mem or direct pipies if this too slow

    app.state.system = shared_mem

    brain = Coordinator(shared_mem.results_queue)
    audio_worker = AudioWorker(shared_mem.audio_queue, shared_mem.results_queue)
    vision_worker = VisionWorker(
        shared_mem.vision_queue,
        shared_mem.results_queue,
        shared_mem.vision_command_queue,
    )

    brain.start()
    audio_worker.start()
    vision_worker.start()

    yield  # app running after this

    print("[System] Shutting down workers")
    workers = [audio_worker, vision_worker, brain]
    for w in workers:
        w.shutdown()

    #queues not empty when we stop it; instead we just stop it from joining & chuck away data
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


def start_server():
    app = FastAPI(lifespan=lifespan)

    # define routes; right now only websocket streamed in
    setup_routes(app)
    return app


app = start_server()
if __name__ == "__main__":
    # start subprocesses & then start server
    try:
        uvicorn.run("main:app", host=config.HOST, port=config.PORT, log_level="error")
    except KeyboardInterrupt:
        print("Shutting down server...")
        #uvicorn gone call shutdowns
    finally:
        print("Server has been shut down.")
