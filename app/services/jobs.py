from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Optional

from ..config import SETTINGS


logger = logging.getLogger(__name__)


class Job:
    def __init__(self):
        self.id = uuid.uuid4().hex
        self.status = 'queued'
        self.progress = 0.0
        self.model_id: Optional[str] = None
        self.detail: Optional[str] = None


class JobManager:
    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=SETTINGS.jobs.max_workers,
            thread_name_prefix="trainer",
        )

    def create(self) -> Job:
        job = Job()
        with self._lock:
            self.jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def run_in_thread(self, job: Job, target: Callable, *args, **kwargs):
        job.detail = job.detail or 'queued'
        self._executor.submit(self._wrap, job, target, *args, **kwargs)

    def _wrap(self, job: Job, target: Callable, *args, **kwargs):
        job.status = 'running'
        try:
            target(job, *args, **kwargs)
            if job.status not in ('failed', 'error'):
                job.status = 'finished'
                job.progress = 1.0
        except Exception as exc:  # pragma: no cover - exercised via tests
            job.status = 'failed'
            job.detail = str(exc)
            logger.exception("Job %s failed", job.id, exc_info=exc)

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)


JOBS = JobManager()
