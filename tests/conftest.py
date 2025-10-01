from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest

import app.config as config
import app.services.storage as storage
import app.services.projects as projects
import app.services.inference as inference
import app.services.training as training


@pytest.fixture()
def temp_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Dict[str, Path]:
    data_dir = tmp_path / "data"
    datasets_dir = data_dir / "datasets"
    projects_dir = data_dir / "projects"
    models_dir = data_dir / "models"
    tmp_dir = tmp_path / "tmp"

    for folder in (datasets_dir, projects_dir, models_dir, tmp_dir):
        folder.mkdir(parents=True, exist_ok=True)

    patch_matrix = (
        (config, "DATA_DIR", data_dir),
        (config, "DATASETS_DIR", datasets_dir),
        (config, "PROJECTS_DIR", projects_dir),
        (config, "MODELS_DIR", models_dir),
        (config, "TMP_DIR", tmp_dir),
        (storage, "DATASETS_DIR", datasets_dir),
        (storage, "PROJECTS_DIR", projects_dir),
        (projects, "PROJECTS_DIR", projects_dir),
        (inference, "DATASETS_DIR", datasets_dir),
        (inference, "PROJECTS_DIR", projects_dir),
        (inference, "MODELS_DIR", models_dir),
        (training, "MODELS_DIR", models_dir),
        (training, "PROJECTS_DIR", projects_dir),
    )

    for module, attr, value in patch_matrix:
        monkeypatch.setattr(module, attr, value)

    monkeypatch.setattr(storage, "MAX_FILE_SIZE_BYTES", 512 * 1024)
    monkeypatch.setattr(storage, "MAX_TOTAL_UPLOAD_BYTES", 1024 * 1024)

    return {
        "data": data_dir,
        "datasets": datasets_dir,
        "projects": projects_dir,
        "models": models_dir,
        "tmp": tmp_dir,
    }
