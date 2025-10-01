from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from app.models.autoencoder import LegacyConvAE
from app.services import inference


def _create_legacy_model(dst: Path) -> None:
    model = LegacyConvAE()
    torch.save(model.state_dict(), dst / "model.pt")

    with (dst / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump({"img_size": 64}, f)

    with (dst / "threshold.json").open("w", encoding="utf-8") as f:
        json.dump({"threshold_mse": 0.05}, f)


def _to_bytes_image(color=(0, 255, 0)) -> bytes:
    img = Image.new("RGB", (48, 48), color=color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_test_images_generates_preview(temp_env):
    model_id = "legacy-model"
    model_dir = temp_env["models"] / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    _create_legacy_model(model_dir)

    buffers = [("sample.png", _to_bytes_image())]
    items = inference.test_images(model_id, temp_env["models"], buffers)

    assert len(items) == 1
    item = items[0]
    assert item["filename"] == "sample.png"
    assert "score" in item and "thr" in item
    assert item["image_url"].startswith("/static/preview/")

    preview_dir = temp_env["data"] / "preview" / model_id
    saved = list(preview_dir.glob("sample_*"))
    # We expect three outputs: input, heatmap, overlay
    assert len(saved) >= 3
