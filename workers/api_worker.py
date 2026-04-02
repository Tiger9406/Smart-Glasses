import asyncio
import multiprocessing as mp
import queue
import time

# from api.gemini_client import GeminiClient
from api.openai_client import OpenAIClient
from core.log_interceptor import install_log_interceptor
from database.database import DatabaseManager
from workers.base import IngestionWorker
from core.config import DEFAULT_NAME

# Tool names that require DB execution and a follow-up LLM call
_DB_TOOLS = {"QUERY_CONVERSATIONS", "GET_KNOWN_PEOPLE"}


class APIWorker(IngestionWorker):
    """Serial Gemini request worker.
    Input queue: gemini_command_queue
    Output queue: results_queue events consumed by Coordinator
    """

    def __init__(self, input_queue: mp.Queue, output_queue: mp.Queue, log_queue: mp.Queue = None):
        super().__init__(input_queue, output_queue, log_queue=log_queue)
        # self.client = GeminiClient()
        self.client = OpenAIClient()
        self.db = DatabaseManager()
        self.loop = None

    def run_async(self, routine):
        if self.loop is None:
            raise RuntimeError("No apiworker loop")
        return self.loop.run_until_complete(routine)

    def run(self):
        if self.log_queue is not None:
            install_log_interceptor(self.log_queue, "[API Worker]")
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

            result = self.run_async(self._run_intent_with_tools(prompt))
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
            subject = command.get("subject", DEFAULT_NAME)

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

    async def _run_intent_with_tools(self, prompt: str) -> dict:
        """Agentic loop: execute DB tools until LLM reaches a terminal command (speak/register/vision/chat)."""
        messages = [
            {"role": "system", "content": self.client.sys_instruct},
            {"role": "user", "content": prompt},
        ]

        for iteration in range(5):
            raw_message, parsed = await self.client._call_tools_api(messages)

            if parsed is None:
                return None

            cmd = parsed.get("cmd", "")

            # Terminal commands — pass straight through to coordinator
            if cmd not in _DB_TOOLS:
                return parsed

            # Execute the DB tool locally and feed results back to the LLM
            tool_result = self._execute_db_tool(cmd, parsed.get("args", {}))
            print(f"[API Worker] Steve DB tool [{cmd}] → {len(tool_result)} chars (iteration {iteration + 1})")

            # Append assistant's tool-call message, then the tool result(s)
            messages.append(raw_message)
            for tc in parsed.get("tool_calls", []):
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": tool_result,
                })

        # Fallback: return whatever the last parsed result was
        return parsed

    def _execute_db_tool(self, cmd: str, args: dict) -> str:
        if cmd == "QUERY_CONVERSATIONS":
            results = self.db.search_conversations(
                person_name=args.get("person_name"),
                date_str=args.get("date"),
                limit=args.get("limit", 20),
            )
            if not results:
                return "No conversations found matching the criteria."
            lines = [f"[{r['timestamp']}] {r['transcript']}" for r in results]
            return f"Found {len(results)} conversation entries:\n" + "\n".join(lines)

        elif cmd == "GET_KNOWN_PEOPLE":
            names = self.db.get_all_user_names()
            if not names:
                return "No known people in the system yet."
            return f"Known people: {', '.join(names)}"

        return f"Unknown tool: {cmd}"
