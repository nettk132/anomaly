from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from app.services import training


class DummyJob:
    def __init__(self):
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
