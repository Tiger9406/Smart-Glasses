#entry point

import uvicorn
from api.server import app
import multiprocessing as mp

import asyncio
from fastapi import FastAPI
from contextlib import asynccontextmanager

from core.shared_mem import SharedMem
from core.coordinator import Coordinator

# defines lifespan; handles startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize resources here
    # shared mem
    shared_mem = SharedMem()
    brain = Coordinator(shared_mem.results_queue)
    
    #vision
    #audio

    brain.start()



    yield #app running after this

    # Cleanup resources here


def start_server():
    app=FastAPI(lifespan=lifespan)
    return app

    #initialize different workers
    #initialize a coordinator



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
