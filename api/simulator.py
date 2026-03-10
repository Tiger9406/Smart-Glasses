import asyncio
import os
import socket
import time
import wave

import cv2

from core import config


# defining vision
async def vision_stream(sock: socket.socket, target_addr: tuple):
    print(f"Opening video file: {config.TARGET_VIDEO}")
    cap = cv2.VideoCapture(config.TARGET_VIDEO)

    if not cap.isOpened():
        print(f"Error opening video file: {config.TARGET_VIDEO}")
        return

    # frame delay & such based on video itself instead of glob param; otherwise have to
    # process video further each time

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = 1.0 / fps if fps > 0 else config.FRAME_DELAY
    print(f"Video streaming at {fps if fps > 0 else 'default'} FPS")

    jpeg_quality = 70

    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()

            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            try:
                frame = cv2.resize(frame, config.RESOLUTION)
                # maximal 65507 bytes
                # around 100 bytes for udp headers, we can fit like 65400 bytes in actual payload
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
                success, buffer = cv2.imencode(".jpg", frame, encode_param)
                if not success:
                    print("Failed to encode frame")
                    continue

                image_bytes = buffer.tobytes()
                payload = config.HEADER_VISION + image_bytes

                if len(payload) > 65400:
                    print(f"Frame too large ({len(payload)} bytes). gotta drop quality")
                    jpeg_quality = max(10, jpeg_quality - 10)
                    continue
            except cv2.error as e:
                print(f"OpenCV specific error occurrsed: {e}")
                continue

            try:
                sock.sendto(payload, target_addr)
            except Exception as e:
                print(f"UDP send error: {e}")

            process_time = time.time() - start_time
            sleep_time = max(0, frame_delay - process_time)
            await asyncio.sleep(sleep_time)
    except asyncio.CancelledError:
        print("Vision stream task cancelled")
        raise
    except Exception as e:
        raise e
    finally:
        cap.release()


# async func definint audio stream output
async def audio_stream(sock: socket.socket, target_addr: tuple):
    print(f"Starting pcm stream target audio file {config.TARGET_AUDIO}")
    with wave.open(config.TARGET_AUDIO, "rb") as wf:
        if (
            wf.getnchannels() != config.CHANNELS
            or wf.getsampwidth() != config.SAMPLE_WIDTH
        ):  # should match mono and two bytes (16 bit)
            print(
                f"WAV should be {config.CHANNELS} channel, {config.SAMPLE_WIDTH * 8}-bit to simulate smart glasses"
            )
            return

        chunk_duration = config.CHUNK_SIZE / config.SAMPLE_RATE
        bytes_per_frame = config.CHANNELS * config.SAMPLE_WIDTH
        expected_bytes = config.CHUNK_SIZE * bytes_per_frame

        print(f"Audio streaming at {config.SAMPLE_RATE}Hz")

        try:
            while True:
                start_time = time.time()
                data = wf.readframes(config.CHUNK_SIZE)

                if len(data) < expected_bytes:  # end of file; rewind
                    wf.rewind()
                    data = wf.readframes(config.CHUNK_SIZE)

                payload = config.HEADER_AUDIO + data

                try:
                    sock.sendto(payload, target_addr)
                except Exception as e:
                    print(f"Audio UDP send error: {e}")

                process_time = time.time() - start_time
                sleep_time = max(0, chunk_duration - process_time)
                await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            print("Audio stream task cancelled")
            raise
        except Exception as e:
            raise e

    print("Audio file closed")


async def stream_glasses_data():
    print("Simulating Smart Glasses Server")
    # repeatedly send over same deafult image

    if not os.path.exists(config.TARGET_VIDEO):
        print(f"File not found: {config.TARGET_VIDEO}")
        return

    if not os.path.exists(config.TARGET_AUDIO):
        print(f"File not found: {config.TARGET_AUDIO}")
        return

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
    target_ip = "127.0.0.1" if config.HOST == "0.0.0.0" else config.HOST
    target_addr = (target_ip, config.PORT)
    print(f"Streaming to {config.HOST}:{config.PORT}")

    try:
        vision_task = asyncio.create_task(vision_stream(sock, target_addr))
        audio_task = asyncio.create_task(audio_stream(sock, target_addr))

        done, pending = await asyncio.wait(
            [vision_task, audio_task], return_when=asyncio.FIRST_COMPLETED
        )

        for task in done:
            if task.exception():
                print(f"Stream crashed w/ error: {task.exception()}")
            else:
                print("Stream finished normally")

        # kill remaining process
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    except Exception as e:
        print(f"Global error: {e}")
    finally:
        sock.close()


if __name__ == "__main__":
    try:
        asyncio.run(stream_glasses_data())
    except KeyboardInterrupt:
        print("\n Stream stopped by user")