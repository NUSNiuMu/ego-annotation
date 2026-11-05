"""Two independent TAKE workers submit windows to one model-owning thread."""
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Empty, Queue
import threading


class WindowQueue:
    def __init__(self):
        self.runtime = {}
        self.requests = Queue()
        self.pending = set()
        self.mutex = threading.Lock()
        self.closed = False

    def infer(self, window, samples, images):
        future = Future()
        with self.mutex:
            if self.closed:
                raise KeyboardInterrupt('Qwen batch owner stopped')
            self.pending.add(future)
            self.requests.put(({'window': window, 'samples': samples, 'images': images}, future))
        try:
            return future.result()
        finally:
            with self.mutex:
                self.pending.discard(future)

    def close(self):
        with self.mutex:
            self.closed = True
            for future in self.pending:
                if not future.done():
                    future.set_exception(KeyboardInterrupt('Qwen batch owner stopped'))


def run_batched(admit, get_worker, complete):
    """admit returns (key, compute(proxy)); complete runs per TAKE, without a barrier.

    CPU preparation/checkpointing may overlap. Only this calling thread touches
    the model, CUDA or global attention context. Missing peers never block a TAKE.
    """
    proxy, jobs = WindowQueue(), {}
    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='qwen-take')
    try:
        while True:
            for key, future in list(jobs.items()):
                if future.done():
                    result = future.result()
                    del jobs[key]
                    complete(key, result)
            while len(jobs) < 2:
                item = admit()
                if item is None:
                    break
                key, compute = item
                if key in jobs:
                    raise ValueError('Duplicate admitted Qwen TAKE')
                jobs[key] = pool.submit(compute, proxy)
            if not jobs:
                return
            try:
                selected = [proxy.requests.get(timeout=.05)]
            except Empty:
                continue
            if len(jobs) > 1:
                try:
                    # Full-resolution JPEG decode/checkpointing can exceed 20ms.
                    # Bound this wait so an absent/preparing peer never stalls work.
                    selected.append(proxy.requests.get(timeout=.2))
                except Empty:
                    pass
            worker = get_worker()
            proxy.runtime = worker.runtime
            results = worker.infer_batch([request for request, _ in selected])
            if len(results) != len(selected):
                raise RuntimeError('Qwen batch result count differs')
            for (_, future), result in zip(selected, results):
                future.set_result(result)
    finally:
        proxy.close()
        pool.shutdown(wait=True, cancel_futures=True)
