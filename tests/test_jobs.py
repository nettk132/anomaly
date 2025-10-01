from __future__ import annotations

import pytest

from app.services.jobs import JobManager


def test_job_manager_success():
    manager = JobManager()
    job = manager.create()

    def worker(job_obj):
        job_obj.progress = 1.0

    try:
        manager.run_in_thread(job, worker)
    finally:
        manager.shutdown()

    assert job.status == "finished"
    assert job.progress == pytest.approx(1.0)


def test_job_manager_failure(caplog):
    caplog.set_level("ERROR")
    manager = JobManager()
    job = manager.create()

    def worker(_job):
        raise RuntimeError("boom")

    try:
        manager.run_in_thread(job, worker)
    finally:
        manager.shutdown()

    assert job.status == "failed"
    assert "boom" in (job.detail or "")
    assert any("Job" in record.message for record in caplog.records)
