import asyncio
import multiprocessing as mp
import queue
import time

from api.gemini_client import GeminiClient
from workers.base import IngestionWorker


class APIWorker(IngestionWorker):
    """Serial Gemini request worker.
    Input queue: gemini_command_queue
    Output queue: results_queue events consumed by Coordinator
    """

    def __init__(self, input_queue: mp.Queue, output_queue: mp.Queue):
        super().__init__(input_queue, output_queue)
        self.client = GeminiClient()
        self.loop = None

    def run_async(self, routine):
        if self.loop is None:
            raise RuntimeError("No apiworker loop")
        return self.loop.run_until_complete(routine)

    def run(self):
        print("[API Worker] Started")
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            while self.running.is_set():
                try:
                    command = self.input_queue.get(timeout=0.1)
                    self._process_command(command)
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.output_queue.put_nowait(
                        {
                            "type": "api_error",
                            "error": str(e),
                            "timestamp": time.time(),
                        }
                    )
                    print(f"[API Worker] Command error: {e}")
        finally:
            try:
                if self.loop is not None and not self.loop.is_closed():
                    self.run_async(self.client.close())
                    self.loop.run_until_complete(self.loop.shutdown_asyncgens())

            except Exception as e:
                print(f"[API Worker] Error closing API Worker: {e}")
            finally:
                if self.loop is not None and not self.loop.is_closed():
                    self.loop.close()
                asyncio.set_event_loop(None)
            print("[API Worker] Shutting down")

    def _process_command(self, command: dict):
        cmd = (command.get("cmd") or "").upper()

        if cmd == "PARSE_INTENT":
            prompt = command.get("text", "")
            if not prompt:
                return

            result = self.run_async(self.client.parse_intent(prompt))
            if result is None:
                return

            self.output_queue.put_nowait(
                {
                    "type": "intent",
                    "cmd": result.get("cmd", "CHAT"),
                    "args": result.get("args", {}),
                    "timestamp": command.get("timestamp", time.time()),
                    "voice_embedding": command.get("voice_embedding"),
                }
            )
            return

        # not being used by coordinator yet tho

        if cmd == "ANALYZE_MEMORY":
            conversation_history = command.get("conversation_history", "")
            known_facts = command.get("known_facts", "None")
            subject = command.get("subject", "Unknown")

            if not conversation_history:
                return

            result = self.run_async(
                self.client.analyze_memory(
                    conversation_history=conversation_history,
                    known_facts=known_facts,
                )
            )

            self.output_queue.put_nowait(
                {
                    "type": "memory_result",
                    "subject": subject,
                    "facts": result,
                    "timestamp": time.time(),
                }
            )
            return

        # Unknown command types are ignored to keep worker resilient.
