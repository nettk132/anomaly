from __future__ import annotations

import json
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import List, Optional

from fastapi import UploadFile

import yaml

from ..config import PROJECTS_DIR, TMP_DIR, ALLOWED_EXTS
from ..utils import PathTraversalError, safe_join, validate_slug

ALLOWED_IMAGE_EXTS = {ext.lower() for ext in ALLOWED_EXTS}
SUMMARY_FILE = "summary.json"
DATA_YAML_NAME = "data.yaml"
NAMES_TXT = "names.txt"


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECTS_DIR.parent).as_posix()
    except ValueError:
        return path.as_posix()


def _project_yolo_dir(project_id: str) -> Path:
    validate_slug(project_id, name="project_id")
    try:
        project_root = safe_join(PROJECTS_DIR, project_id, must_exist=True)
    except (FileNotFoundError, PathTraversalError):
        raise FileNotFoundError("project not found")
    pdir = project_root / "yolo"
    if not pdir.exists():
        raise FileNotFoundError("project not found")
    pdir.mkdir(exist_ok=True)
    (pdir / "dataset").mkdir(exist_ok=True)
    return pdir


def dataset_dir(project_id: str) -> Path:
    return _project_yolo_dir(project_id) / "dataset"


def _write_upload_to_tmp(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "dataset.zip").suffix or ".zip"
    tmp = Path(tempfile.mkstemp(prefix="yolo_ds_", suffix=suffix, dir=TMP_DIR)[1])
    with tmp.open("wb") as f:
        upload.file.seek(0)
        while True:
            chunk = upload.file.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    upload.file.seek(0)
    return tmp


def _clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _maybe_flatten_root(path: Path) -> None:
    entries = [p for p in path.iterdir() if p.name != "__MACOSX"]
    if len(entries) == 1 and entries[0].is_dir():
        inner = entries[0]
        for child in inner.iterdir():
            shutil.move(str(child), path)
        shutil.rmtree(inner, ignore_errors=True)



def _infer_classes(labels_root: Path, names_hint: Optional[List[str]] = None) -> List[str]:
    if names_hint:
        return names_hint
    txt = labels_root / NAMES_TXT
    if txt.exists():
        names = [line.strip() for line in txt.read_text(encoding="utf-8").splitlines() if line.strip()]
        if names:
            return names
    ids: set[int] = set()
    for label_file in labels_root.rglob("*.txt"):
        try:
            for line in label_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                parts = line.strip().split()
                try:
                    idx = int(float(parts[0]))
                except Exception:
                    continue
                ids.add(idx)
        except Exception:
            continue
    if not ids:
        return []
    max_id = max(ids)
    names = [f"class{i}" for i in range(max_id + 1)]
    return names


def _normalise_names(names: List[str]) -> List[str]:
    cleaned = []
    for name in names:
        n = str(name).strip()
        if n:
            cleaned.append(n)
    if not cleaned:
        return []
    return cleaned


def _load_existing_yaml(path: Path) -> tuple[Optional[List[str]], Optional[dict]]:
    if not path.exists():
        return None, None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None, None
    names = data.get("names")
    if isinstance(names, dict):
        names = [str(names[k]) for k in sorted(names.keys())]
    elif isinstance(names, list):
        names = [str(n) for n in names]
    else:
        names = None
    return names, data


def _write_data_yaml(root: Path, train_rel: str, val_rel: str, names: List[str], extra: Optional[dict] = None) -> Path:
    payload = {
        "path": str(root.resolve()),
        "train": train_rel,
        "val": val_rel,
        "nc": len(names),
        "names": names,
    }
    if extra:
        payload.update({k: v for k, v in extra.items() if k not in payload})
    data_path = root / DATA_YAML_NAME
    data_path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    return data_path


def _count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in ALLOWED_IMAGE_EXTS)


def save_yolo_dataset(project_id: str, upload: UploadFile) -> dict:
    yolo_root = _project_yolo_dir(project_id)
    ddir = yolo_root / "dataset"
    _clear_dir(ddir)

    tmp_path = _write_upload_to_tmp(upload)
    try:
        with zipfile.ZipFile(tmp_path, "r") as zf:
            zf.extractall(path=ddir)
    except zipfile.BadZipFile as exc:
        raise ValueError("invalid zip archive") from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    _maybe_flatten_root(ddir)

    train_dir = ddir / "train" / "images"
    labels_dir = ddir / "train" / "labels"
    if not train_dir.exists() or not labels_dir.exists():
        raise ValueError("dataset must contain train/images and train/labels directories")
    val_dir = ddir / "val" / "images"
    if not val_dir.exists():
        val_dir = train_dir

    existing_names, existing_yaml = _load_existing_yaml(ddir / DATA_YAML_NAME)
    names = _infer_classes(labels_dir, _normalise_names(existing_names or []))
    if not names:
        raise ValueError("cannot infer class names; provide data.yaml with names or names.txt")

    yaml_path = _write_data_yaml(ddir, "train/images", "val/images", names, existing_yaml)

    summary = {
        "ready": True,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_images": _count_images(train_dir),
        "val_images": _count_images(val_dir),
        "classes": names,
        "yaml_path": _display_path(yaml_path),
    }
    (ddir / SUMMARY_FILE).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def describe_dataset(project_id: str) -> dict:
    ddir = dataset_dir(project_id)
    summary_path = ddir / SUMMARY_FILE
    if summary_path.exists():
        try:
            return json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    yaml_path = ddir / DATA_YAML_NAME
    if not yaml_path.exists():
        return {"ready": False, "train_images": 0, "val_images": 0, "classes": [], "yaml_path": None, "updated_at": None}
    existing_names, _ = _load_existing_yaml(yaml_path)
    train_dir = ddir / "train" / "images"
    val_dir = ddir / "val" / "images"
    summary = {
        "ready": True,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_images": _count_images(train_dir),
        "val_images": _count_images(val_dir),
        "classes": existing_names or [],
        "yaml_path": _display_path(yaml_path),
    }
    return summary


def clear_dataset(project_id: str) -> None:
    ddir = dataset_dir(project_id)
    if ddir.exists():
        shutil.rmtree(ddir)
    ddir.mkdir(parents=True, exist_ok=True)
