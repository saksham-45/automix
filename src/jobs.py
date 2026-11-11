"""
Bounded render queue for the club server.

A fixed pool of worker threads (``workers``) runs render jobs; at most ``max_queue``
further jobs may wait. Everything is in-process (the server runs one gunicorn
worker), so this is a plain Condition-guarded deque — no broker needed.

Cancellation is cooperative: ``cancel()`` sets the job's event, and the job's own
callbacks call ``job.check()``, which raises :class:`Cancelled`. ``Cancelled``
derives from ``BaseException`` on purpose, so a stray ``except Exception`` inside
the audio engine cannot swallow it — the render unwinds all the way out and the
slot frees immediately.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Callable, Dict, Optional

log = logging.getLogger("automix.jobs")


class Cancelled(BaseException):
    """Raised inside a running job once it has been cancelled."""


class QueueFull(Exception):
    """Raised by submit() when every slot is busy and the wait queue is full."""


class Job:
    __slots__ = ("id", "fn", "cancel_event", "state", "enqueued_at", "started_at",
                 "finished_at")

    def __init__(self, job_id: str, fn: Callable[["Job"], None]):
        self.id = job_id
        self.fn = fn
        self.cancel_event = threading.Event()
        self.state = "queued"            # queued | running | done | cancelled
        self.enqueued_at = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def check(self) -> None:
        """Cooperative cancellation point: raise Cancelled if cancel() was called."""
        if self.cancel_event.is_set():
            raise Cancelled()


class JobQueue:
    def __init__(self, workers: int = 1, max_queue: int = 4, name: str = "render"):
        self.workers = max(1, int(workers))
        self.max_queue = max(0, int(max_queue))
        self.name = name
        self._cv = threading.Condition()
        self._pending: deque = deque()
        self._running: Dict[str, Job] = {}
        self._jobs: Dict[str, Job] = {}
        self._threads: list = []
        self._closed = False

    # ------------------------------------------------------------------ #
    def _ensure_workers(self) -> None:
        # Started lazily (first submit) so importing the server never spawns threads
        # before gunicorn forks, and tests can swap queues freely.
        alive = [t for t in self._threads if t.is_alive()]
        self._threads = alive
        for i in range(len(alive), self.workers):
            t = threading.Thread(target=self._worker, name=f"{self.name}-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def _worker(self) -> None:
        while True:
            with self._cv:
                while not self._pending and not self._closed:
                    self._cv.wait()
                if self._closed:
                    return
                job = self._pending.popleft()
                job.state, job.started_at = "running", time.time()
                self._running[job.id] = job
            try:
                if not job.cancelled:
                    job.fn(job)
            except Cancelled:
                pass
            except Exception:  # the job owns its error reporting; never kill the worker
                log.exception("job %s crashed", job.id)
            finally:
                with self._cv:
                    self._running.pop(job.id, None)
                    self._jobs.pop(job.id, None)
                    job.state = "cancelled" if job.cancelled else "done"
                    job.finished_at = time.time()
                    self._cv.notify_all()

    # ------------------------------------------------------------------ #
    def submit(self, job_id: str, fn: Callable[[Job], None]) -> Job:
        with self._cv:
            if self._closed:
                raise QueueFull("server is shutting down")
            # Jobs not yet picked up by an idle worker still hold a slot, so count
            # running + pending against the total capacity (slots + wait places).
            if len(self._running) + len(self._pending) >= self.workers + self.max_queue:
                raise QueueFull()
            job = Job(job_id, fn)
            self._jobs[job_id] = job
            self._pending.append(job)
            self._ensure_workers()
            self._cv.notify()
            return job

    def position(self, job_id: str) -> Optional[int]:
        """1-based place in the wait line; 0 while running; None if unknown/finished."""
        with self._cv:
            if job_id in self._running:
                return 0
            free = max(0, self.workers - len(self._running))
            for i, j in enumerate(self._pending):
                if j.id == job_id:
                    # Within the free slots it is about to start (0); otherwise its
                    # 1-based place among the jobs that really have to wait.
                    return 0 if i < free else i - free + 1
            return None

    def cancel(self, job_id: str) -> bool:
        """Cancel a queued (removed immediately) or running (unwinds at its next
        check()) job. Returns True if the job was known."""
        with self._cv:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            job.cancel_event.set()
            if job in self._pending:
                self._pending.remove(job)
                self._jobs.pop(job_id, None)
                job.state, job.finished_at = "cancelled", time.time()
            self._cv.notify_all()
            return True

    def wait_idle(self, job_id: str, timeout: float = 10.0) -> bool:
        """Block until the job has left the queue and its slot (tests / shutdown)."""
        end = time.time() + timeout
        with self._cv:
            while job_id in self._jobs:
                left = end - time.time()
                if left <= 0:
                    return False
                self._cv.wait(left)
        return True

    def stats(self) -> dict:
        with self._cv:
            return {"workers": self.workers, "running": len(self._running),
                    "queued": len(self._pending), "max_queue": self.max_queue}

    def saturated(self) -> bool:
        with self._cv:
            return len(self._running) + len(self._pending) >= self.workers + self.max_queue

    def shutdown(self) -> None:
        """Cancel everything (running jobs unwind at their next check)."""
        with self._cv:
            self._closed = True
            for job in list(self._jobs.values()):
                job.cancel_event.set()
            self._pending.clear()
            self._cv.notify_all()
