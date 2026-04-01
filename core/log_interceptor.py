import builtins
import functools
import time
import multiprocessing as mp


def install_log_interceptor(log_queue: mp.Queue, source_tag: str):
    """
    Monkey-patches builtins.print in the calling process to also put
    structured log dicts onto log_queue. Original print is preserved
    so terminal output still works.

    Must be called from inside the target process (e.g., first line of run()).
    """
    original_print = builtins.print

    @functools.wraps(original_print)
    def _intercepted_print(*args, **kwargs):
        original_print(*args, **kwargs)
        text = " ".join(str(a) for a in args)
        try:
            log_queue.put_nowait({
                "source": source_tag,
                "text": text,
                "ts": time.time(),
            })
        except Exception:
            pass  # never block a worker on a full log queue

    builtins.print = _intercepted_print
