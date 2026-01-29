import asyncio
import websockets
import time
import os
import wave
from core import config

#defining vision
async def vision_stream(websocket):
    with open(config.TARGET_IMAGE, "rb") as f:
        image_bytes = f.read()
    print(f"Loaded {len(image_bytes)} bytes of image from {config.TARGET_IMAGE}")
    while True:
        start_time = time.time()
        await websocket.send(config.HEADER_VISION+ image_bytes)
        process_time = time.time()-start_time
        sleep_time = max(0, config.FRAME_DELAY-process_time)
        await asyncio.sleep(sleep_time)

#async func definint audio stream output
async def audio_stream(websocket):
    print(f"Starting pcm stream target audio file {config.TARGET_AUDIO}")
    wf = wave.open(config.TARGET_AUDIO, 'rb')

    if wf.getnchannels() != config.CHANNELS or wf.getsampwidth() != config.SAMPLE_WIDTH: # should match mono and two bytes (16 bit)
        print(f"WAV should be {config.CHANNELS} channel, {config.SAMPLE_WIDTH*8}-bit to simulate smart glasses")
        return
    
    chunk_duration = config.CHUNK_SIZE / config.SAMPLE_RATE

    print(f"Audio streaming at {config.SAMPLE_RATE}Hz")

    while True:
        start_time = time.time()
        data = wf.readframes(config.CHUNK_SIZE)
        if len(data)< (config.CHUNK_SIZE*2): #end of file; rewind
            wf.rewind()
            data=wf.readframes(config.CHUNK_SIZE)

        await websocket.send(config.HEADER_AUDIO+data)

        process_time = time.time()-start_time
        sleep_time=max(0, chunk_duration - process_time)
        await asyncio.sleep(sleep_time)
        
async def stream_glasses_data():
    print("Simulating Smart Glasses Server")
    #repeatedly send over same deafult image

    if not os.path.exists(config.TARGET_IMAGE):
        print(f"File not found: {config.TARGET_IMAGE}")
        return
    
    if not os.path.exists(config.TARGET_AUDIO):
        print(f"File not found: {config.TARGET_AUDIO}")
        return
    
    print("Connecting to server")

    async with websockets.connect(config.SERVER_URL) as websocket:
        print("Streaming now")
        await asyncio.gather( #runs these two async funcs on same thread
            vision_stream(websocket),
            audio_stream(websocket)
        )

    return

if __name__=="__main__":
    try:
        asyncio.run(stream_glasses_data())
    except KeyboardInterrupt:
        print("\n Stream stopped by user")
    except ConnectionRefusedError:
        print(f"\n Error: Could not connect to {config.SERVER_URL}")