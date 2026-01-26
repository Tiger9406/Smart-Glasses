
#define endpoints

from fastapi import APIRouter, WebSocket, Request

router = APIRouter()

#we only have a websocket as of right now
#maybe we can make it so REST request to start connection & then websocket until REST close connection request
#for now just do as if it's always websocketed

def is_vision_data(data: bytes):
     #logic for differentiating vision vs audio data        
     #first byte is padded and for vision data == 1
     return data[0] == 1

@router.websocket("/stream")
async def stream_ingest(websocket: WebSocket):
    await websocket.accept()

    system = websocket.app.state.system

    try:
        while True:
            data = await websocket.receive_bytes()
            if data:
                if is_vision_data(data):
                    system.video_queue.put(data)
                else:
                    system.audio_queue.put(data)

    except Exception:
         pass

def setup_routes(app):
     app.include_router(router)