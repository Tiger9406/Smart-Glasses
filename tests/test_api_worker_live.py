import multiprocessing as mp
import os
import time

from workers.api_worker import APIWorker


def run_live_test():
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not set")

    in_q = mp.Queue()
    out_q = mp.Queue()

    worker = APIWorker(in_q, out_q)
    worker.start()

    try:
        in_q.put(
            {
                "cmd": "PARSE_INTENT",
                "text": "Hi, my name is John.",
                "timestamp": time.time(),
                "voice_embedding": None,
            }
        )

        event = out_q.get(timeout=20)
        assert event["type"] in ("intent", "api_error"), event
        print("Live event:", event)

    finally:
        worker.shutdown()
        worker.join(timeout=2)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=1)


if __name__ == "__main__":
    run_live_test()
