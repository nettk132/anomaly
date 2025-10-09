from __future__ import annotations

import time
from threading import Thread

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file as save_safetensors

from app.services import training
from app.services.training import TMP_DIR


class DummyJob:
    def __init__(self):
        self.id = "job"
        self.status = "queued"
        self.progress = 0.0
        self.detail = None
        self.model_id = None


def _write_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=(128, 0, 128)).save(str(path), format="PNG")


def test_train_job_requires_minimum_images(temp_env):
    raw_dir = temp_env["datasets"] / "scene-small" / "raw"
    for idx in range(3):
        _write_image(raw_dir / f"img_{idx}.png")

    job = DummyJob()
    with pytest.raises(ValueError):
        training.train_job(job, "scene-small", raw_dir, img_size=64, epochs=1, lr=1e-3)


def test_resolve_model_checkpoint_missing(temp_env):
    with pytest.raises(FileNotFoundError):
        training.resolve_model_checkpoint("does-not-exist")


def test_resolve_model_checkpoint_prefers_safetensors(temp_env):
    scene_dir = temp_env["models"] / "scene-one"
    scene_dir.mkdir(parents=True, exist_ok=True)
    safepath = scene_dir / "model.safetensors"
    ptpath = scene_dir / "model.pt"

    save_safetensors({"weight": torch.zeros(1)}, str(safepath))
    ptpath.write_bytes(b"legacy")

    ref = training.resolve_model_checkpoint("scene-one")
    assert ref.path == safepath
    assert ref.format == "safetensors"


def test_resolve_model_checkpoint_pt_fallback(temp_env):
    scene_dir = temp_env["models"] / "scene-two"
    scene_dir.mkdir(parents=True, exist_ok=True)
    ptpath = scene_dir / "model.pt"
    ptpath.write_bytes(b"legacy")

    ref = training.resolve_model_checkpoint("scene-two")
    assert ref.path == ptpath
    assert ref.format == "pt"


def test_training_requires_approval(monkeypatch, temp_env):
    job = DummyJob()
    job.id = "job-approval"
    job.detail = "queued"

    monkeypatch.setattr(training.SETTINGS.training, "require_approval", True, raising=False)

    approval_dir = TMP_DIR / "approvals" / "training"
    approval_dir.mkdir(parents=True, exist_ok=True)
    approved_file = approval_dir / f"{job.id}.approved"
    pending_file = approval_dir / f"{job.id}.pending"
    approved_file.unlink(missing_ok=True)
    pending_file.unlink(missing_ok=True)

    approved_file.write_text("ok", encoding="utf-8")

    training._await_training_approval(job)

    assert job.status == "running"
    assert "Approved" in (job.detail or "")
    assert not pending_file.exists()
