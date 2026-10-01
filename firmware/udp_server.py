import socket
import cv2
import numpy as np
import pyaudio
import queue
import threading

# Frame headers matching config.h
WS_FRAME_VIDEO = 0x01
WS_FRAME_AUDIO_TX = 0x02

UDP_IP = "0.0.0.0"
UDP_PORT = 8000

video_queue = queue.Queue(maxsize=10)
audio_queue = queue.Queue(maxsize=50)

# --- 1. Audio Playback Thread ---
def audio_worker():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, output=True)
    
    while True:
        audio_data = audio_queue.get()
        if audio_data is None: 
            break
        try:
            stream.write(audio_data)
        except Exception:
            pass
            
    stream.stop_stream()
    stream.close()
    p.terminate()

# --- 2. Fast UDP Receiver Thread ---
def udp_receiver():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Maximize OS receive buffer to prevent dropping large video packets
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"UDP Server listening on {UDP_IP}:{UDP_PORT}")

    while True:
        try:
            # 65507 is the absolute max data size for a UDP datagram
            data, addr = sock.recvfrom(65507) 
            if not data: 
                continue
                
            frame_type = data[0]
            payload = data[1:]

            if frame_type == WS_FRAME_VIDEO:
                if video_queue.full():
                    video_queue.get_nowait() 
                video_queue.put_nowait(payload)
                
            elif frame_type == WS_FRAME_AUDIO_TX:
                if audio_queue.full():
                    audio_queue.get_nowait()
                audio_queue.put_nowait(payload)
                
        except Exception as e:
            print(f"UDP Error: {e}")

# --- 3. Main Thread (Video Decoding & UI) ---
if __name__ == "__main__":
    threading.Thread(target=audio_worker, daemon=True).start()
    threading.Thread(target=udp_receiver, daemon=True).start()

    print("Waiting for stream... Press 'q' to quit.")
    
    while True:
        try:
            payload = video_queue.get(timeout=0.05)
            nparr = np.frombuffer(payload, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                cv2.imshow('ESP32 Live Stream', img)
                
        except queue.Empty:
            pass 
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    audio_queue.put(None) 
    cv2.destroyAllWindows()