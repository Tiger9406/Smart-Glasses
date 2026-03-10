import socket
import threading
import queue

from core import config
from core.shared_mem import SharedMem

udp_shutdown_event = threading.Event()


def udp_receiver_thread(shared_mem: SharedMem):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)

    sock.settimeout(1.0)  # time-out to check udp_shutdown; to address blocking sock.recvfrom
    sock.bind(
        (config.HOST, config.PORT)
    )

    print(f"[System] UDP Server listening on {config.HOST}:{config.PORT}")

    while not udp_shutdown_event.is_set():
        try:
            data, addr = sock.recvfrom(65507)
            if not data:
                continue

            header = data[0:1]
            payload = data[1:]

            if header == config.HEADER_VISION:
                if shared_mem.vision_queue.full():
                    try:
                        shared_mem.vision_queue.get_nowait()
                    except queue.Empty:
                        pass
                shared_mem.vision_queue.put_nowait(payload)

            elif header == config.HEADER_AUDIO:
                if shared_mem.audio_queue.full():
                    try:
                        shared_mem.audio_queue.get_nowait()
                    except queue.Empty:
                        pass
                shared_mem.audio_queue.put_nowait(payload)

        except socket.timeout:
            continue
        except Exception as e:
            if not udp_shutdown_event.is_set():
                print(f"[System] UDP Error: {e}")

    sock.close()
    print("[System] UDP socket closed.")
