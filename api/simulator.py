import asyncio
import websockets
import time
import os
import wave

SERVER_URL = "ws://localhost:8000/stream"
FPS = 20
FRAME_DELAY = 1.0/FPS
RESOLUTION = (800, 600)
TARGET_IMAGE = "./api/800x600.jpeg"
TARGET_AUDIO = "api/OSR_us_11.wav"
CHUNK_SIZE = 1024


async def visual_stream(websocket):
    with open(TARGET_IMAGE, "rb") as f:
        image_bytes = f.read()
    print(f"Loaded {len(image_bytes)} bytes of image from {TARGET_IMAGE}")
    while True:
        start_time = time.time()
        await websocket.send(b'\x01'+ image_bytes)
        process_time = time.time()-start_time
        sleep_time = max(0, FRAME_DELAY-process_time)
        await asyncio.sleep(sleep_time)

async def audio_stream(websocket):
    print(f"Starting pcm stream target audio file {TARGET_AUDIO}")
    wf = wave.open(TARGET_AUDIO, 'rb')

    if wf.getnchannels() != 1 or wf.getsampwidth() != 2: # should match mono and two bytes (16 bit)
        print("WAV should be Mono 16-bit PCM to simulate smart glasses")
        return
    
    sample_rate = wf.getframerate()
    chunk_duration = CHUNK_SIZE / sample_rate

    print(f"Audio streaming at {sample_rate}Hz")

    while True:
        start_time = time.time()
        data = wf.readframes(CHUNK_SIZE)
        if len(data)< (CHUNK_SIZE*2): #end of file; rewind
            wf.rewind()
            data=wf.readframes(CHUNK_SIZE)

        await websocket.send(b'\x02'+data)

        process_time = time.time()-start_time
        sleep_time=max(0, chunk_duration - process_time)
        await asyncio.sleep(sleep_time)
        
async def stream_glasses_data():
    print("Simulating Smart Glasses Server")
    #repeatedly send over same deafult image

    if not os.path.exists(TARGET_IMAGE):
        print(f"File not found: {TARGET_IMAGE}")
        return
    
    if not os.path.exists(TARGET_AUDIO):
        print(f"File not found: {TARGET_AUDIO}")
        return
    
    print("Connecting to server")

    async with websockets.connect(SERVER_URL) as websocket:
        print("Streaming now")
        await asyncio.gather( #runs these two async funcs on same thread
            visual_stream(websocket),
            audio_stream(websocket)
        )

    return

if __name__=="__main__":
    try:
        asyncio.run(stream_glasses_data())
    except KeyboardInterrupt:
        print("\n Stream stopped by user")
    except ConnectionRefusedError:
        print(f"\n Error: Could not connect to {SERVER_URL}")