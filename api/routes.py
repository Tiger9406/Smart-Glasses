
#define endpoints

from fastapi import APIRouter, WebSocket, Request
from core import config

router = APIRouter()

#we only have a websocket as of right now
#maybe we can make it so REST request to start connection & then websocket until REST close connection request
#for now just do as if it's always websocketed


@router.websocket("/stream")
async def stream_ingest(websocket: WebSocket):
    await websocket.accept()
    print("Client connected to stream endpoint")

    system = websocket.app.state.system

    try:
        while True:
            data = await websocket.receive_bytes()
            if not data: continue
            header = data[0:1]
            payload=data[1:]
            if header==config.HEADER_VIDEO:
                system.vision_queue.put(payload)
            elif header == config.HEADER_AUDIO:
                system.audio_queue.put(payload)
            else:
                print("Unkonwn header type")

    except Exception:
         pass

def setup_routes(app):
     app.include_router(router)