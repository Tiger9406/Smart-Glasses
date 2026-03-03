# coordinator: looks at global queue and processes output from sub workers vision and audio
# kinda like the decision making part

import asyncio
import json
import multiprocessing as mp
import queue
import threading
import time

from core import config
from core.config import VLM_ACTIVE
from workers.base import BaseWorker
from workers.vision_utils.VLM import VLMClient


class Coordinator(BaseWorker):
    def __init__(self, results_queue: mp.Queue, commands_queue: mp.Queue):
        super().__init__()
        self.results_queue = results_queue
        self.commands_queue = commands_queue
        # self.audio_events
        # self.vision_events

        # maybe more; hold past actions taken by self maybe? or a state, like what's happening in the world rn?
        # again, decision making module given the initial processing by the workers
        with open("workers/coordinator_utils/ContextDict.json") as f:
            self.memory_db = json.load(f)

    def setup(self):
        self.start_time = time.time()
        self.request_number = 0

        self.vlm_client = VLMClient(
            api_key=config.GEMINI_API_KEY, url=config.GEMINI_API_LINK
        )

        self.loop = asyncio.new_event_loop()
        self.async_thread = threading.Thread(  # assigns loop to thread
            target=self._start_background_loop, daemon=True
        )
        self.async_thread.start()

    def _start_background_loop(self):
        """
        spins a separate thread, keeps an event loop open
        so we can reuse the VLMClient session across multiple requests.
        """
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()  # blocking; awaits something thrown at self.loop

    def run(self):
        print("[Coordinator] Started")
        self.setup()

        try:
            while self.running.is_set():
                try:
                    event = self.results_queue.get(timeout=0.1)
                    self._handle_event(event)
                    self._test_VLM()
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    break
        finally:
            print("[Coordinator] Shutting down")

    def _test_VLM(self):
        # comment return statement to test VLM funcitonality
        if not VLM_ACTIVE:
            return
        if time.time() - self.start_time > 5 and self.request_number == 0:
            command = {
                "cmd": "GET_VIDEO_CONTEXT",
                "prompt": "Summarize the video in a sentence",
                "request_id": self.request_number,
            }
            self.request_number += 1
            print("[Coordinator] Put command in vision queue")
            self.commands_queue.put_nowait(command)

    async def _handle_gemini(self, prompt, request_id: int):  # handling api request
        try:
            name, new_facts = await self.vlm_client.analyze_memory(prompt)

            if new_facts:
                self.results_queue.put(
                    {"type": "llm_result", "name": name, "new_facts": new_facts}
                )
                print(
                    f"[LLM Worker] Sent {len(new_facts)} new facts for {name} to Coordinator."
                )
            else:
                print(f"[LLM Worker] No new facts found for {name}.")
        except Exception as e:
            print(f"[Coordinator] LLM task error: {e}")

    def _handle_event(self, event):
        # handling events; gotta coordinate event data format
        # for instance if event type is a face in view, we throw it on the picture or sum

        event_type = event.get("type", "unknown")

        if event_type == "vision_result":
            # given list of the following:
            """{
                "track_id": track_id,
                "bbox": (x1, y1, x2, y2),
                "name": self.active_identities[track_id]["name"],
                "score": self.active_identities[track_id]["score"],
                "emb": emb # could be none if embedding not extracted on this frame
            }"""

            # potential code to work with input faces; nothing for now, too many frames
            """
            faces = event.get("faces", [])
            if faces:
                print(f"\n [Coordinator] Vision Event: detected {len(faces)} faces")
                for face in faces:
                    _name = face.get("name", DEFAULT_NAME)
                    _score = face.get("score", 0.0)
                    _bbox = face.get("bbox")
                    # print(f" - ID: {face['track_id']} | Name: {name} ({score:.2f}) | Loc: {bbox}")

            """
            pass

        elif event_type == "speech":
            if not event.get("final", False):
                return
            name = event.get("name", "Unknown")
            speech_text = event.get("text", "").strip()
            print(f"[Coordinator] {name}: {speech_text}")

            if not speech_text:
                return

            current_facts = self.memory_db.get(name, [])
            if current_facts:
                facts_string = "\n".join([f"- {fact}" for fact in current_facts])

                payload = {
                    "cmd": "UPDATE_MEMORY",
                    "name": name,
                    "speech_text": speech_text,
                    "facts_string": facts_string,
                    "request_id": event.get("id", -1),
                }

                asyncio.run_coroutine_threadsafe(
                    self._handle_gemini(payload, event.get("id", -1)),
                    self.loop,
                )

        elif event_type == "llm_result":
            name = event.get("name", "Unknown")
            new_facts_str = event.get("new_facts", "").strip()

            if new_facts_str:
                # update the DB
                current_facts = self.memory_db.get(name, [])

                fact_lines = new_facts_str.split("\n")

                for line in fact_lines:
                    # Strip out whitespace, dashes, and asterisks from the start/end
                    clean_line = line.strip(" -*")

                    if clean_line:  # Make sure it's not an empty line
                        self.memory_db[name].append(clean_line)

                # save to disk safely, maybe do this on coord shutdown but its prob good to keep updating the mem
                # could dispatch the saving every x seconds or something also
                try:
                    with open("workers/coordinator_utils/ContextDict.json", "w") as f:
                        json.dump(self.memory_db, f, indent=2)
                    print(
                        f"[Coordinator] Successfully saved memory for {name}: {new_facts_str}"
                    )
                except Exception as e:
                    print(f"[Coordinator] Error writing to ContextDict.json: {e}")

        elif event_type == "vlm_result":
            """"
            "type": "vlm_result",
            "request_id": request_id,
            "text": response_text,
            "timestamp": time.time(),
            """
            print(f"[Coordinator] Received VLM output: {event['text']}")

        else:
            print("\n[Coordinator] got other event")

        return
