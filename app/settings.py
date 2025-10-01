from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class UploadSettings(BaseModel):
    allowed_extensions: List[str] = Field(default_factory=lambda: [".jpg", ".jpeg", ".png"])
    max_files_per_upload: int = Field(default=200, ge=1, description="Maximum number of files per upload batch")
    max_file_size_mb: int = Field(default=32, ge=1, description="Per-file size limit in megabytes")
    max_total_size_mb: int = Field(default=512, ge=1, description="Total request size limit in megabytes")

    @field_validator("allowed_extensions")
    @classmethod
    def _normalize_exts(cls, value: List[str]) -> List[str]:
        exts = []
        for ext in value:
            if not ext:
                continue
            ext = ext.strip().lower()
            if not ext:
                continue
            if not ext.startswith('.'):
                ext = f'.{ext}'
            exts.append(ext)
        if not exts:
            raise ValueError("allowed_extensions must contain at least one extension")
        return sorted(set(exts))


class TrainingSettings(BaseModel):
    batch_size: int = Field(default=32, ge=1)
    val_ratio: float = Field(default=0.2, gt=0.0, lt=0.9)
    patience: Optional[int] = Field(default=10, ge=1)
    min_epochs_before_early_stop: int = Field(default=5, ge=1)


class JobsSettings(BaseModel):
    max_workers: int = Field(default=2, ge=1)
    poll_interval_seconds: float = Field(default=0.5, gt=0)


class YamlConfigSettingsSource(PydanticBaseSettingsSource):
    def __init__(self, settings_cls: type[BaseSettings], config_path: Path):
        super().__init__(settings_cls)
        self._config_path = config_path
        self._data_cache: Optional[Dict[str, Any]] = None

    def _load(self) -> Dict[str, Any]:
        if self._data_cache is not None:
            return self._data_cache
        if not self._config_path.exists():
            self._data_cache = {}
            return self._data_cache
        data = yaml.safe_load(self._config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"config file {self._config_path} must contain a mapping")
        self._data_cache = data
        return self._data_cache

    def get_field_value(self, field, field_name: str):  # type: ignore[override]
        data = self._load()
        if field_name in data:
            return data[field_name], field_name
        return None, None

    def __call__(self) -> Dict[str, Any]:
        return dict(self._load())


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    root_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_dir: Optional[Path] = None
    models_dir: Optional[Path] = None
    datasets_dir: Optional[Path] = None
    projects_dir: Optional[Path] = None
    tmp_dir: Optional[Path] = None
    uploads: UploadSettings = Field(default_factory=UploadSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    jobs: JobsSettings = Field(default_factory=JobsSettings)

    @staticmethod
    def _default_config_path() -> Path:
        override = os.environ.get("APP_CONFIG_FILE")
        if override:
            return Path(override)
        return Path(__file__).resolve().parents[1] / "config.yaml"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_source = YamlConfigSettingsSource(settings_cls, cls._default_config_path())
        # Priority: explicit init > env vars > .env > yaml file > secrets
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            yaml_source,
            file_secret_settings,
        )

    @field_validator(
        "root_dir",
        mode="after",
    )
    @classmethod
    def _resolve_root(cls, value: Path) -> Path:
        return value.resolve()

    @model_validator(mode="after")
    def _populate_paths(self) -> "AppSettings":
        root = self.root_dir.resolve()
        self.data_dir = (self.data_dir or root / "data").resolve()
        self.models_dir = (self.models_dir or self.data_dir / "models").resolve()
        self.datasets_dir = (self.datasets_dir or self.data_dir / "datasets").resolve()
        self.projects_dir = (self.projects_dir or self.data_dir / "projects").resolve()
        self.tmp_dir = (self.tmp_dir or root / "tmp").resolve()

        for folder in (self.data_dir, self.models_dir, self.datasets_dir, self.projects_dir, self.tmp_dir):
            folder.mkdir(parents=True, exist_ok=True)

        return self

    @property
    def upload_max_file_bytes(self) -> int:
        return self.uploads.max_file_size_mb * 1024 * 1024

    @property
    def upload_max_total_bytes(self) -> int:
        return self.uploads.max_total_size_mb * 1024 * 1024


SETTINGS = AppSettings()
