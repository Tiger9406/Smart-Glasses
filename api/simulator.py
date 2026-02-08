import asyncio
import os
import time
import wave

import cv2
import websockets

from core import config


# defining vision
async def vision_stream(websocket):
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

    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()

            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                print("Failed to encode frame")
                continue

            image_bytes = buffer.tobytes()

            try:
                await websocket.send(config.HEADER_VISION + image_bytes)
            except websockets.exceptions.ConnectionClosed:
                print("Vision stream connection closed by server")
                break

            process_time = time.time() - start_time
            sleep_time = max(0, frame_delay - process_time)
            await asyncio.sleep(sleep_time)
    except asyncio.CancelledError:
        print("Vision stream task cancelled")
        raise
    except Exception as e:
        print(f"Error in vision stream: {e}")
        raise e
    finally:
        cap.release()


# async func definint audio stream output
async def audio_stream(websocket):
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

                try:
                    await websocket.send(config.HEADER_AUDIO + data)
                except websockets.exceptions.ConnectionClosed:
                    print("audio stream connection closed by server")
                    break

                process_time = time.time() - start_time
                sleep_time = max(0, chunk_duration - process_time)
                await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            print("Audio stream task cancelled")
            raise
        except Exception as e:
            print(f"Error in audio stream: {e}")
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

    print("Connecting to server")

    try:
        async with websockets.connect(config.SERVER_URL) as websocket:
            print("Streaming now")

            # create separate tasks
            vision_task = asyncio.create_task(vision_stream(websocket))
            audio_task = asyncio.create_task(audio_stream(websocket))

            _, pending = await asyncio.wait(
                [vision_task, audio_task], return_when=asyncio.FIRST_COMPLETED
            )

            # kill remaining process
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    except websockets.exceptions.ConnectionRefusedError:
        print("Connection failed & could not connect to target server url")
    except Exception as e:
        print(f"Global error: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(stream_glasses_data())
    except KeyboardInterrupt:
        print("\n Stream stopped by user")