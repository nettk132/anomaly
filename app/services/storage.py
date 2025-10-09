# app/services/storage.py
from __future__ import annotations
import hashlib
import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Set, Tuple

from fastapi import UploadFile

import yaml

from PIL import Image, UnidentifiedImageError

from ..config import (
    DATASETS_DIR,
    PROJECTS_DIR,
    ALLOWED_EXTS,
    MAX_FILES_PER_UPLOAD,
    MAX_FILE_SIZE_BYTES,
    MAX_TOTAL_UPLOAD_BYTES,
)
from ..utils import PathTraversalError, safe_join, validate_slug

BASE_MODEL_EXTS = {".safetensors"}
METADATA_FILENAME = "_index.json"
INTERNAL_ID_REGEX = re.compile(r"^[0-9a-f]{32}$")
INTERNAL_FILENAME_REGEX = re.compile(r"^[0-9a-f]{32}\.[a-z0-9]{1,8}$")
LEGACY_FILENAME_REGEX = re.compile(r"^[A-Za-z0-9._-]+$")

logger = logging.getLogger(__name__)


def normalize_stored_filename(name: str) -> str:
    """Normalize filenames saved in project/raw folders, allowing legacy names."""
    if not name:
        raise ValueError("invalid filename")

    cleaned = Path(name).name
    if cleaned in (".", ".."):
        raise ValueError("invalid filename")
    if cleaned != name:
        raise ValueError("invalid filename")
    if "/" in cleaned or "\\" in cleaned:
        raise ValueError("invalid filename")

    ext = Path(cleaned).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise ValueError("not an allowed image")

    if INTERNAL_FILENAME_REGEX.fullmatch(cleaned):
        return cleaned
    if LEGACY_FILENAME_REGEX.fullmatch(cleaned):
        logger.info("Using legacy stored filename: %s", cleaned)
        return cleaned

    raise ValueError("invalid filename")

# ---------- utils ----------
def _write_and_hash(
    src_file,
    dst_path: Path,
    *,
    chunk_size: int = 1 << 20,
    max_bytes: int | None = None,
) -> Tuple[str, int]:
    """
    เขียนไฟล์ลงปลายทางแบบสตรีม พร้อมคำนวณแฮช (blake2b) ในคราวเดียว
    คืนค่า digest hex string
    """
    h = hashlib.blake2b(digest_size=16)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    src_file.seek(0)
    written = 0
    with dst_path.open("wb") as f:
        while True:
            chunk = src_file.read(chunk_size)
            if not chunk:
                break
            written += len(chunk)
            if max_bytes is not None and written > max_bytes:
                raise ValueError("file too large")
            h.update(chunk)
            f.write(chunk)
    src_file.seek(0)  # คืนพอยน์เตอร์ เผื่อ FastAPI จะใช้งานต่อ
    return h.hexdigest(), written


def _verify_image(path: Path) -> None:
    try:
        with Image.open(path) as img:
            img.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("invalid image content") from exc


def _metadata_path(root: Path) -> Path:
    return root / METADATA_FILENAME


def _load_image_metadata(root: Path) -> Dict[str, Dict[str, object]]:
    path = _metadata_path(root)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    files = data.get("files")
    if not isinstance(files, dict):
        return {}
    sanitized: Dict[str, Dict[str, object]] = {}
    for key, value in files.items():
        if isinstance(key, str) and isinstance(value, dict):
            sanitized[key] = value
    return sanitized


def _save_image_metadata(root: Path, metadata: Dict[str, Dict[str, object]]) -> None:
    payload = {
        "files": metadata,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _metadata_path(root).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _cleanup_metadata(root: Path, metadata: Dict[str, Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    """Drop entries whose files disappeared; persist if mutations occur."""
    to_remove = [name for name in metadata if not (root / name).is_file()]
    if not to_remove:
        return metadata
    for name in to_remove:
        metadata.pop(name, None)
    _save_image_metadata(root, metadata)
    return metadata


def _generate_internal_name(root: Path, ext: str, metadata: Dict[str, Dict[str, object]]) -> Tuple[str, str]:
    """Return (file_id, stored_filename)."""
    ext = ext.lower()
    if not ext.startswith("."):
        ext = f".{ext}"
    while True:
        file_id = uuid.uuid4().hex
        stored = f"{file_id}{ext}"
        if stored not in metadata and not (root / stored).exists():
            return file_id, stored


def _store_images(
    root: Path,
    files: Iterable[UploadFile],
    *,
    dedup: bool,
    dedup_content: bool,
) -> int:
    metadata = _cleanup_metadata(root, _load_image_metadata(root))
    existing_hashes = {
        str(info.get("hash"))
        for info in metadata.values()
        if isinstance(info, dict) and info.get("hash")
    }
    seen_display_names: Set[str] = set()
    seen_hashes: Set[str] = set()
    saved = 0
    total_bytes = 0

    for idx, upload in enumerate(files):
        if idx >= MAX_FILES_PER_UPLOAD or total_bytes >= MAX_TOTAL_UPLOAD_BYTES:
            break

        display_name = Path(upload.filename or f"upload_{idx}").name
        ext = Path(display_name).suffix.lower()
        if ext not in ALLOWED_EXTS:
            continue

        if dedup and display_name in seen_display_names:
            continue

        file_id, stored_name = _generate_internal_name(root, ext, metadata)
        dst = root / stored_name

        try:
            file_hash, written = _write_and_hash(upload.file, dst, max_bytes=MAX_FILE_SIZE_BYTES)
        except ValueError:
            dst.unlink(missing_ok=True)
            continue

        if total_bytes + written > MAX_TOTAL_UPLOAD_BYTES:
            dst.unlink(missing_ok=True)
            break

        try:
            _verify_image(dst)
        except ValueError:
            dst.unlink(missing_ok=True)
            continue

        hash_duplicate = file_hash in existing_hashes or file_hash in seen_hashes
        if dedup_content and hash_duplicate:
            dst.unlink(missing_ok=True)
            continue

        seen_display_names.add(display_name)
        seen_hashes.add(file_hash)
        existing_hashes.add(file_hash)
        total_bytes += written
        saved += 1

        metadata[stored_name] = {
            "id": file_id,
            "filename": stored_name,
            "display_name": display_name,
            "original_name": display_name,
            "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hash": file_hash,
            "size": written,
        }

    if saved:
        _save_image_metadata(root, metadata)

    if saved:
        logger.info("Stored %s image(s) under %s", saved, root)
    return saved


def _remove_metadata_entry(root: Path, filename: str) -> None:
    metadata = _load_image_metadata(root)
    if filename in metadata:
        metadata.pop(filename, None)
        _save_image_metadata(root, metadata)


def _list_images(root: Path, url_factory: Callable[[str], str]) -> List[dict]:
    metadata = _cleanup_metadata(root, _load_image_metadata(root))
    items: List[dict] = []

    if metadata:
        for stored_name in sorted(metadata.keys()):
            info = metadata.get(stored_name) or {}
            path = root / stored_name
            if not path.is_file():
                continue
            display = str(
                info.get("display_name")
                or info.get("original_name")
                or stored_name
            )
            items.append(
                {
                    "id": str(info.get("id") or Path(stored_name).stem),
                    "filename": stored_name,
                    "display_name": display,
                    "url": url_factory(stored_name),
                }
            )
        return items

    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTS:
            items.append(
                {
                    "id": path.stem,
                    "filename": path.name,
                    "display_name": path.name,
                    "url": url_factory(path.name),
                }
            )
    return items


# ---------- SCENE (ของเดิม รักษาให้ใช้ได้ต่อ) ----------
def get_raw_dir(scene_id: str) -> Path:
    validate_slug(scene_id, name="scene_id")
    try:
        return safe_join(DATASETS_DIR, scene_id, "raw", must_exist=False)
    except PathTraversalError as exc:
        raise ValueError("invalid scene_id") from exc


def save_uploads(
    scene_id: str,
    files: Iterable[UploadFile],
    dedup: bool = True,
    dedup_content: bool = True,
) -> int:
    """Persist scene uploads using internal filenames and metadata."""
    target = get_raw_dir(scene_id)
    target.mkdir(parents=True, exist_ok=True)
    return _store_images(target, files, dedup=dedup, dedup_content=dedup_content)



# ---------- PROJECT (ของใหม่) ----------
def get_project_raw_dir(project_id: str) -> Path:
    validate_slug(project_id, name="project_id")
    try:
        return safe_join(PROJECTS_DIR, project_id, "raw", must_exist=False)
    except PathTraversalError as exc:
        raise ValueError("invalid project_id") from exc


def save_project_uploads(
    project_id: str,
    files: Iterable[UploadFile],
    dedup: bool = True,
    dedup_content: bool = True,
) -> int:
    """Persist project uploads using internal filenames and metadata."""
    target = get_project_raw_dir(project_id)
    target.mkdir(parents=True, exist_ok=True)
    return _store_images(target, files, dedup=dedup, dedup_content=dedup_content)


def list_project_images(project_id: str) -> list[dict]:
    try:
        raw = safe_join(PROJECTS_DIR, project_id, "raw", must_exist=False)
    except (FileNotFoundError, PathTraversalError):
        return []
    if not raw.exists():
        return []
    return _list_images(raw, lambda name: f"/static/projects/{project_id}/raw/{name}")


def list_scene_images(scene_id: str) -> list[dict]:
    try:
        raw = get_raw_dir(scene_id)
    except (FileNotFoundError, PathTraversalError, ValueError):
        return []
    if not raw.exists():
        return []
    return _list_images(raw, lambda name: f"/static/datasets/{scene_id}/raw/{name}")

def _validate_filename(name: str) -> None:
    """Validate that the provided filename is a safe, known image resource."""
    normalize_stored_filename(name)


def delete_project_image(project_id: str, filename: str) -> bool:
    try:
        normalized = normalize_stored_filename(filename)
    except ValueError:
        return False
    try:
        raw = safe_join(PROJECTS_DIR, project_id, 'raw', must_exist=False)
    except (FileNotFoundError, PathTraversalError):
        return False
    if not raw.exists():
        return False
    try:
        path = safe_join(raw, normalized, must_exist=False)
    except (FileNotFoundError, PathTraversalError):
        return False
    if path.is_file():
        path.unlink()
        _remove_metadata_entry(raw, normalized)
        logger.info("Deleted project image %s/%s", project_id, normalized)
        return True
    return False


def delete_scene_image(scene_id: str, filename: str) -> bool:
    try:
        normalized = normalize_stored_filename(filename)
    except ValueError:
        return False
    try:
        raw = get_raw_dir(scene_id)
    except (FileNotFoundError, PathTraversalError, ValueError):
        return False
    if not raw.exists():
        return False
    try:
        path = safe_join(raw, normalized, must_exist=False)
    except (FileNotFoundError, PathTraversalError):
        return False
    if path.is_file():
        path.unlink()
        _remove_metadata_entry(raw, normalized)
        logger.info("Deleted scene image %s/%s", scene_id, normalized)
        return True
    return False


def save_project_base_model(project_id: str, file: UploadFile) -> dict:
    try:
        project_dir = safe_join(PROJECTS_DIR, project_id, must_exist=True)
    except (FileNotFoundError, PathTraversalError):
        raise FileNotFoundError('project not found')

    filename = Path(file.filename or 'model.safetensors').name
    ext = Path(filename).suffix.lower()
    if ext not in BASE_MODEL_EXTS:
        raise ValueError('file must be .safetensors')

    imports_dir = safe_join(project_dir, 'imports', must_exist=False)
    imports_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime('%Y%m%d-%H%M%S')
    model_id = f"import-{ts}-{uuid.uuid4().hex[:8]}"
    model_id = validate_slug(model_id, name='model_id')
    dest_dir = safe_join(imports_dir, model_id, must_exist=False)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = safe_join(dest_dir, 'model.safetensors', must_exist=False)

    file_hash, _ = _write_and_hash(file.file, dest_path, max_bytes=MAX_FILE_SIZE_BYTES)

    config = {
        'project_id': project_id,
        'created_at': ts,
        'note': f'Uploaded checkpoint ({filename})',
        'base_model_source': 'upload',
        'origin_filename': filename,
        'file_hash': file_hash,
    }
    with (dest_dir / 'config.yaml').open('w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, allow_unicode=True)

    logger.info("Stored base model %s for project %s at %s", model_id, project_id, dest_dir)
    return {'model_id': model_id, 'filename': filename}
