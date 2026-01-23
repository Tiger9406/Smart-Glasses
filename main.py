#entry point

import uvicorn
from api.server import app
import multiprocessing as mp

import asyncio
from fastapi import FastAPI
from contextlib import asynccontextmanager

# defines lifespan; handles startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize resources here
    # shared mem
    #vision
    #audio
    #coordinator

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
