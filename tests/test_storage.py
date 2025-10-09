from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pytest
import torch
from fastapi import UploadFile
from PIL import Image
from safetensors.torch import save_file as save_safetensors

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
    assert (scene_raw / "_index.json").exists()

    listing = storage.list_scene_images("scene-a")
    assert len(listing) == 3
    assert {item["display_name"] for item in listing} == {f"img_{i}.png" for i in range(3)}
    assert all(item["filename"] != item["display_name"] for item in listing)
    assert all(len(Path(item["filename"]).stem) == 32 for item in listing)
    metadata = json.loads((scene_raw / "_index.json").read_text(encoding="utf-8"))
    assert {info["display_name"] for info in metadata["files"].values()} == {f"img_{i}.png" for i in range(3)}


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
    listing = storage.list_scene_images("scene-b")
    assert len(listing) == 1
    assert listing[0]["display_name"] == "ok.png"
    assert listing[0]["filename"].endswith(".png")
    assert listing[0]["filename"] != listing[0]["display_name"]


def test_save_project_uploads_dedup(temp_env):
    file1 = _make_image_upload("dup.png")
    file2 = _make_image_upload("dup.png")
    saved = storage.save_project_uploads("proj-1", [file1, file2])
    assert saved == 1

    proj_raw = temp_env["projects"] / "proj-1" / "raw"
    files = list(proj_raw.glob("*.png"))
    assert len(files) == 1
    assert (proj_raw / "_index.json").exists()
    listing = storage.list_project_images("proj-1")
    assert len(listing) == 1
    assert listing[0]["display_name"] == "dup.png"
    assert listing[0]["filename"] != "dup.png"
    assert len(Path(listing[0]["filename"]).stem) == 32
    metadata = json.loads((proj_raw / "_index.json").read_text(encoding="utf-8"))
    assert list(metadata["files"].values())[0]["display_name"] == "dup.png"


def test_save_project_base_model_requires_safetensors(temp_env):
    proj_dir = temp_env["projects"] / "proj-import"
    proj_dir.mkdir(parents=True, exist_ok=True)

    bad = _make_binary_upload("bad.pt", b"not allowed")
    with pytest.raises(ValueError):
        storage.save_project_base_model("proj-import", bad)

    weights_path = temp_env["tmp"] / "weights.safetensors"
    save_safetensors({"w": torch.zeros(1)}, str(weights_path))
    good = _make_binary_upload("weights.safetensors", weights_path.read_bytes())
    info = storage.save_project_base_model("proj-import", good)
    assert info["filename"] == "weights.safetensors"
