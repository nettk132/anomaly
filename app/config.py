from __future__ import annotations

from pathlib import Path

from .settings import SETTINGS

ROOT: Path = SETTINGS.root_dir
DATA_DIR: Path = SETTINGS.data_dir
DATASETS_DIR: Path = SETTINGS.datasets_dir
MODELS_DIR: Path = SETTINGS.models_dir
TMP_DIR: Path = SETTINGS.tmp_dir
PROJECTS_DIR: Path = SETTINGS.projects_dir

ALLOWED_EXTS = set(SETTINGS.uploads.allowed_extensions)
MAX_FILES_PER_UPLOAD = SETTINGS.uploads.max_files_per_upload
MAX_FILE_SIZE_BYTES = SETTINGS.upload_max_file_bytes
MAX_TOTAL_UPLOAD_BYTES = SETTINGS.upload_max_total_bytes

__all__ = [
    "ROOT",
    "DATA_DIR",
    "DATASETS_DIR",
    "MODELS_DIR",
    "TMP_DIR",
    "PROJECTS_DIR",
    "ALLOWED_EXTS",
    "MAX_FILES_PER_UPLOAD",
    "MAX_FILE_SIZE_BYTES",
    "MAX_TOTAL_UPLOAD_BYTES",
    "SETTINGS",
]
