from __future__ import annotations
import threading, time, uuid, json
from pathlib import Path
from typing import Callable, Dict, Optional

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

    def create(self) -> Job:
        job = Job()
        with self._lock:
            self.jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def run_in_thread(self, job: Job, target: Callable, *args, **kwargs):
        th = threading.Thread(target=self._wrap, args=(job, target, *args), kwargs=kwargs, daemon=True)
        th.start()

    def _wrap(self, job: Job, target: Callable, *args, **kwargs):
        job.status = 'running'
        try:
            target(job, *args, **kwargs)
            if job.status not in ('failed', 'error'):
                job.status = 'finished'
                job.progress = 1.0
        except Exception as e:
            job.status = 'failed'
            job.detail = str(e)

JOBS = JobManager()
