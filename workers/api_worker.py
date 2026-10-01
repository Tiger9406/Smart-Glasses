import asyncio
import multiprocessing as mp
import queue
import time
from collections.abc import Coroutine
from typing import Any, Protocol, TypeVar

from api.openai_client import OpenAIClient
from core.config import DEFAULT_NAME
from core.log_interceptor import install_log_interceptor
from workers.api_utils.commands import command_name, intent_event, memory_event
from workers.base import IngestionWorker

_T = TypeVar("_T")


class LLMClient(Protocol):
    async def parse_intent(self, prompt: str) -> dict[str, Any] | None: ...

    async def analyze_memory(
        self, conversation_history: str, known_facts: str = "None"
    ) -> Any: ...

    async def close(self) -> None: ...


class APIWorker(IngestionWorker):
    """Serial LLM request worker.
    Input queue: llm_command_queue
    Output queue: results_queue events consumed by Coordinator
    """

    client: LLMClient
    loop: asyncio.AbstractEventLoop | None

    def __init__(
        self, input_queue: mp.Queue, output_queue: mp.Queue, log_queue: mp.Queue
    ) -> None:
        super().__init__(input_queue, output_queue, log_queue=log_queue)
        self.client = OpenAIClient()
        self.loop = None

    def run_async(self, routine: Coroutine[Any, Any, _T]) -> _T:
        if self.loop is None:
            raise RuntimeError("No apiworker loop")
        return self.loop.run_until_complete(routine)

    def run(self) -> None:
        install_log_interceptor(self.log_queue, "[API Worker]")
        print("[API Worker] Started")
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            while self.running.is_set():
                try:
                    # one llm call at a time, in the order they were queued
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

    def _process_command(self, command: dict[str, Any]) -> None:
        cmd = command_name(command)
        now = time.time()

        # turn the latest transcript into an intent the coordinator can act on
        if cmd == "PARSE_INTENT":
            if not command.get("text"):
                return
            result = self.run_async(self.client.parse_intent(command.get("text", "")))
            event = intent_event(command, result, now)
            if event is not None:
                self.output_queue.put_nowait(event)
            return

        # pull facts out of the conversation
        if cmd == "ANALYZE_MEMORY":
            conversation_history = command.get("conversation_history", "")
            if not conversation_history:
                return
            result = self.run_async(
                self.client.analyze_memory(
                    conversation_history=conversation_history,
                    known_facts=command.get("known_facts", "None"),
                )
            )
            event = memory_event(command, result, now, DEFAULT_NAME)
            if event is not None:
                self.output_queue.put_nowait(event)
