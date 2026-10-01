import asyncio
import websockets
import cv2
import numpy as np
import pyaudio
import queue
import threading

# Frame headers matching config.h
WS_FRAME_VIDEO = 0x01
WS_FRAME_AUDIO_TX = 0x02

# Queues to absorb network jitter
video_queue = queue.Queue(maxsize=10)
audio_queue = queue.Queue(maxsize=50)

# --- 1. Audio Playback Thread ---
def audio_worker():
    p = pyaudio.PyAudio()
    # Matching ESP32 setup: 16000 Hz, 16-bit PCM, Mono
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, output=True)
    
    while True:
        audio_data = audio_queue.get()
        if audio_data is None:  # Exit signal
            break
        try:
            stream.write(audio_data)
        except Exception as e:
            print(f"Audio playback error: {e}")
            
    stream.stop_stream()
    stream.close()
    p.terminate()

# --- 2. Async WebSocket Server ---
async def handle_esp32_stream(websocket):
    print(f"ESP32 Connected from {websocket.remote_address}")
    
    try:
        async for message in websocket:
            if not isinstance(message, bytes):
                continue
                
            frame_type = message[0]
            payload = message[1:]

            if frame_type == WS_FRAME_VIDEO:
                try:
                    # Drop the oldest frame if the video decoder is falling behind
                    if video_queue.full():
                        video_queue.get_nowait() 
                    video_queue.put_nowait(payload)
                except queue.Full:
                    pass
                    
            elif frame_type == WS_FRAME_AUDIO_TX:
                try:
                    audio_queue.put_nowait(payload)
                except queue.Full:
                    pass 
                    
    except websockets.exceptions.ConnectionClosed:
        print("ESP32 Disconnected")

async def run_server():
    # Use the modern async context manager to start the server
    async with websockets.serve(handle_esp32_stream, "0.0.0.0", 8000):
        print("WebSocket Server listening on ws://0.0.0.0:8000")
        await asyncio.Future()  # Keeps the server running forever

def start_ws_server():
    # asyncio.run() automatically creates and manages the running event loop
    asyncio.run(run_server())

# --- 3. Main Thread (Video Decoding & UI) ---
if __name__ == "__main__":
    # Start background threads
    threading.Thread(target=audio_worker, daemon=True).start()
    threading.Thread(target=start_ws_server, daemon=True).start()

    print("Waiting for stream... Press 'q' to quit.")
    
    # OpenCV MUST run in the main thread
    while True:
        try:
            # Wait up to 50ms for a new video frame
            payload = video_queue.get(timeout=0.05)
            
            # Decode and display
            nparr = np.frombuffer(payload, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                cv2.imshow('ESP32 Live Stream', img)
                
        except queue.Empty:
            pass # No new frame yet, just keep checking for the 'q' key
            
        # UI refresh and exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Closing...")
            break

    # Cleanup
    audio_queue.put(None) # Tell audio thread to exit
    cv2.destroyAllWindows()