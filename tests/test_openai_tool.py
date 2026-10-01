import asyncio

import cv2

from api.openai_client import OpenAIClient


async def main() -> None:
    print("Initializing OpenAIClient...")
    client = OpenAIClient()

    try:
        print("\n--- Testing analyze_memory ---")
        history = "Steve: Hello! How can I help?\nTiger: Actually, I'm working on a C++ project."
        facts = "User is building a high-performance system."
        memory_result = await client.analyze_memory(history, facts)

        # Prompt asks for a JSON list of {subject, fact}
        assert isinstance(memory_result, list), (
            f"Expected list, got {type(memory_result)}"
        )
        assert len(memory_result) > 0, "Memory result list is empty"
        print("Memory Result verified.")

        print("\n--- Testing analyze_video_frames ---")
        frames = get_video_clip()

        # Assert frame extraction worked properly
        assert frames is not None, "Failed to extract frames"
        assert isinstance(frames[0], bytes), "Frames must be encoded as bytes"

        prompt = "Describe what happens in this sequence of images."
        vision_result = await client.analyze_video_frames(frames, prompt)

        assert isinstance(vision_result, str), (
            f"Expected string, got {type(vision_result)}"
        )
        assert len(vision_result.strip()) > 0, "Vision result string is empty"
        print("Vision Result verified.")

        print("\n--- Testing parse_intent (register_identity) ---")
        # Self Introduction
        prompt_self_intro = (
            "Speaker: Sarah: Hi Tiger, I don't think we've met. My name is Sarah."
        )
        result_self_intro = await client.parse_intent(prompt_self_intro)
        assert result_self_intro is not None
        assert result_self_intro["cmd"] == "REGISTER_IDENTITY", (
            f"Wrong command: {result_self_intro.get('cmd')}"
        )
        assert result_self_intro["args"]["name"] == "Sarah", (
            "Failed to extract name 'Sarah'"
        )
        assert result_self_intro["args"]["speaker_name"] == "Sarah", (
            "Failed to identify speaker 'Sarah'"
        )
        assert result_self_intro["args"]["is_self_introduction"] is True, (
            "Failed to identify self-introduction"
        )
        print("Self Introduction parsed correctly.")

        print("\n--- Testing parse_intent (register_identity) ---")
        prompt_self_intro = (
            "Speaker: Sarah: Hi Tiger, I don't think we've met. My name is Sarah."
        )
        print(f"Prompt: {prompt_self_intro}")
        result_self_intro = await client.parse_intent(prompt_self_intro)
        print("Result:", result_self_intro)

        prompt_hello = "Speaker: Tiger: Hey Sean."
        result_third_party = await client.parse_intent(prompt_hello)
        assert result_third_party is not None
        assert result_third_party["cmd"] == "REGISTER_IDENTITY", (
            f"Wrong command: {result_third_party.get('cmd')}"
        )
        assert result_third_party["args"]["name"] == "Sean", (
            "Failed to extract name 'Sean'"
        )
        assert result_third_party["args"]["speaker_name"] == "Tiger", (
            "Failed to identify speaker 'Tiger'"
        )
        assert result_third_party["args"]["is_self_introduction"] is False, (
            "Failed to identify non-self-intro"
        )
        print("Greetings parsed correctly.")

    except Exception as e:
        print("\nError during testing: ", e)

    finally:
        print("\nClosing client session...")
        await client.close()


def get_video_clip() -> list[bytes] | None:
    video_path = "api/simulator_resources/LongVideo.mp4"

    # 1. Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    # We want 1 second of video, but only extract 3 evenly spaced frames to save tokens
    target_frames_to_extract = 3
    frame_interval = int(fps / target_frames_to_extract)

    frames_bytes = []
    frame_count = 0
    extracted_count = 0

    while cap.isOpened() and frame_count < fps:
        ret, frame = cap.read()
        if not ret:
            break

        if (
            frame_count % frame_interval == 0
            and extracted_count < target_frames_to_extract
        ):
            # Encode frame as JPEG
            success, buffer = cv2.imencode(".jpg", frame)
            if success:
                frames_bytes.append(buffer.tobytes())
            extracted_count += 1

        frame_count += 1

    cap.release()

    if not frames_bytes:
        print("Error: No frames extracted.")
        return

    print(f"Successfully extracted {len(frames_bytes)} frames.")
    return frames_bytes


if __name__ == "__main__":
    asyncio.run(main())