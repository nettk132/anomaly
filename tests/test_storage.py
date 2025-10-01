from __future__ import annotations

from io import BytesIO
from pathlib import Path

from fastapi import UploadFile
from PIL import Image

from app.services import storage


def _make_image_upload(
    filename: str,
    *,
    size: tuple[int, int] = (32, 32),
    color: tuple[int, int, int] = (255, 0, 0),
) -> UploadFile:
    bio = BytesIO()
    Image.new("RGB", size, color=color).save(bio, format="PNG")
    bio.seek(0)
    return UploadFile(filename=filename, file=BytesIO(bio.getvalue()))


def _make_binary_upload(filename: str, payload: bytes) -> UploadFile:
    bio = BytesIO(payload)
    bio.seek(0)
    return UploadFile(filename=filename, file=bio)


def test_save_uploads_enforces_limits(temp_env):
    files = [
        _make_image_upload(f"img_{i}.png", color=(i * 40 % 255, 64, 128))
        for i in range(3)
    ]
    saved = storage.save_uploads("scene-a", files)
    assert saved == 3

    scene_raw = temp_env["datasets"] / "scene-a" / "raw"
    assert len(list(scene_raw.glob("*.png"))) == 3


def test_save_uploads_skips_invalid_and_too_large(temp_env, monkeypatch):
    monkeypatch.setattr(storage, "MAX_FILE_SIZE_BYTES", 100)
    monkeypatch.setattr(storage, "MAX_TOTAL_UPLOAD_BYTES", 150)

    tiny = _make_image_upload("ok.png")
    oversized = _make_binary_upload("large.png", b"x" * 200)
    bogus = _make_binary_upload("bad.png", b"not an image")

    saved = storage.save_uploads("scene-b", [tiny, oversized, bogus])
    assert saved == 1

    scene_raw = temp_env["datasets"] / "scene-b" / "raw"
    files = list(scene_raw.glob("*.png"))
    assert len(files) == 1
    assert files[0].name == "ok.png"


def test_save_project_uploads_dedup(temp_env):
    file1 = _make_image_upload("dup.png")
    file2 = _make_image_upload("dup.png")
    saved = storage.save_project_uploads("proj-1", [file1, file2])
    assert saved == 1

    proj_raw = temp_env["projects"] / "proj-1" / "raw"
    files = list(proj_raw.glob("*.png"))
    assert len(files) == 1
