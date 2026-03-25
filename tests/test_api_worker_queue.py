import multiprocessing as mp
import queue
import time

from workers.api_worker import APIWorker


class FakeGeminiClient:
    async def parse_intent(self, prompt: str):
        return {
            "cmd": "REGISTER_IDENTITY",
            "args": {
                "name": "John",
                "speaker_name": "Unknown",
                "is_self_introduction": True,
            },
        }

    async def analyze_memory(
        self, conversation_history: str, known_facts: str = "None"
    ):
        return [{"subject": "Alice", "fact": "Likes sushi"}]

    async def close(self):
        return None


class FailingGeminiClient:
    async def parse_intent(self, prompt: str):
        raise RuntimeError("boom")

    async def analyze_memory(
        self, conversation_history: str, known_facts: str = "None"
    ):
        raise RuntimeError("boom")

    async def close(self):
        return None


class TestableAPIWorker(APIWorker):
    def __init__(self, input_queue: mp.Queue, output_queue: mp.Queue, client):
        super().__init__(input_queue, output_queue)
        self.client = client


def assert_queue_empty(q: mp.Queue):
    try:
        q.get_nowait()
        raise AssertionError("Expected queue to be empty")
    except queue.Empty:
        pass


def test_process_command_happy_paths():
    input_q = mp.Queue()
    output_q = mp.Queue()

    worker = APIWorker(input_q, output_q)
    worker.client = FakeGeminiClient()

    t0 = time.time()
    worker._process_command(
        {
            "cmd": "PARSE_INTENT",
            "text": "Hi, I am John",
            "timestamp": t0,
            "voice_embedding": [0.1, 0.2],
        }
    )

    evt = output_q.get(timeout=1)
    assert evt["type"] == "intent"
    assert evt["cmd"] == "REGISTER_IDENTITY"
    assert evt["args"]["name"] == "John"
    assert evt["timestamp"] == t0
    assert evt["voice_embedding"] == [0.1, 0.2]

    worker._process_command(
        {
            "cmd": "ANALYZE_MEMORY",
            "conversation_history": "Alice said she likes sushi",
            "known_facts": "None",
            "subject": "Alice",
        }
    )

    mem_evt = output_q.get(timeout=1)
    assert mem_evt["type"] == "memory_result"
    assert mem_evt["subject"] == "Alice"
    assert isinstance(mem_evt["facts"], list)
    assert mem_evt["facts"][0]["subject"] == "Alice"
    assert "timestamp" in mem_evt


def test_process_command_negative_paths():
    input_q = mp.Queue()
    output_q = mp.Queue()

    worker = APIWorker(input_q, output_q)
    worker.client = FakeGeminiClient()

    worker._process_command({"cmd": "PARSE_INTENT", "text": ""})
    assert_queue_empty(output_q)

    worker._process_command({"cmd": "ANALYZE_MEMORY", "conversation_history": ""})
    assert_queue_empty(output_q)

    worker._process_command({"cmd": "UNKNOWN_COMMAND"})
    assert_queue_empty(output_q)


def test_run_loop_emits_api_error_on_exception():
    input_q = mp.Queue()
    output_q = mp.Queue()

    worker = TestableAPIWorker(input_q, output_q, FailingGeminiClient())
    worker.start()

    input_q.put(
        {"cmd": "PARSE_INTENT", "text": "this should fail", "timestamp": time.time()}
    )

    err_evt = output_q.get(timeout=3)
    assert err_evt["type"] == "api_error"
    assert "boom" in err_evt["error"]
    assert "timestamp" in err_evt

    worker.shutdown()
    worker.join(timeout=2)
    if worker.is_alive():
        worker.terminate()
        worker.join(timeout=1)


def run_tests():
    test_process_command_happy_paths()
    test_process_command_negative_paths()
    test_run_loop_emits_api_error_on_exception()
    print("API worker queue tests passed")


if __name__ == "__main__":
    run_tests()
