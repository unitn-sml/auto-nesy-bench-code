"""Fixed-size pool of worker processes for background model counting.

Lets :class:`fonesy.statistics.metrics.Metrics` count models while the LLM
generates. Unlike ``multiprocessing.Pool`` (which uses SIGTERM), workers are
stopped with SIGKILL, the only reliable way to stop native SAT/ApproxMC code.
"""

import multiprocessing as mp
import queue
import time
from typing import Any, Callable, Dict, Tuple

from fonesy.evaluation.procutil import set_pdeathsig


def _pool_worker(in_queue, out_queue, ready_queue) -> None:
    """Worker loop: run ``(job_id, fn, args)`` jobs and report ``(job_id, status, payload)``.

    Signals *ready_queue* once protected by PDEATHSIG; runs until killed.
    """
    set_pdeathsig()
    ready_queue.put(None)
    while True:
        job_id, fn, args = in_queue.get()
        try:
            out_queue.put((job_id, "ok", fn(*args)))
        except Exception as e:  # noqa: BLE001 - forwarded to the parent as a message
            out_queue.put((job_id, "error", f"{type(e).__name__}: {e}"))


class McWorkerPool:
    """A fixed-size pool of persistent ``spawn`` workers (the parent may hold a CUDA context)."""

    def __init__(self, n_workers: int, ready_timeout: float = 30.0):
        if n_workers < 1:
            raise ValueError("n_workers must be >= 1")
        self._ctx = mp.get_context("spawn")
        self._in_queue = self._ctx.Queue()
        self._out_queue = self._ctx.Queue()
        ready_queue = self._ctx.Queue()
        # Not daemonic: jobs spawn their own hard-timeout subprocesses.
        self._processes = [
            self._ctx.Process(
                target=_pool_worker, args=(self._in_queue, self._out_queue, ready_queue)
            )
            for _ in range(n_workers)
        ]
        self._shutdown_done = False
        for p in self._processes:
            p.start()
        try:
            # Wait until every worker is protected by PDEATHSIG.
            for _ in range(n_workers):
                ready_queue.get(timeout=ready_timeout)
        except Exception:
            self.shutdown()
            raise
        self._next_job_id = 0

    def submit(self, fn: Callable, args: tuple) -> int:
        """Queue ``fn(*args)`` (non-blocking) and return its job id."""
        job_id = self._next_job_id
        self._next_job_id += 1
        self._in_queue.put((job_id, fn, args))
        return job_id

    def collect(self, deadline: float, n_expected: int) -> Dict[int, Tuple[str, Any]]:
        """Collect finished jobs until *n_expected* are done or *deadline* passes.

        *deadline* is a ``time.monotonic()`` value (``time.monotonic()`` itself
        polls without waiting). Returns ``{job_id: ("ok" | "error", payload)}``.
        """
        results: Dict[int, Tuple[str, Any]] = {}
        while len(results) < n_expected:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                job_id, status, payload = self._out_queue.get(timeout=remaining)
            except queue.Empty:
                break
            results[job_id] = (status, payload)
        return results

    def shutdown(self) -> None:
        """SIGKILL and reap every worker (idempotent, safe in signal handlers)."""
        if self._shutdown_done:
            return
        self._shutdown_done = True
        for p in self._processes:
            if p.is_alive():
                p.kill()
        for p in self._processes:
            p.join(timeout=5)
