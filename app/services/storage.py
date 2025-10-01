# app/services/storage.py
from __future__ import annotations
import hashlib
import time
import uuid
from pathlib import Path
from typing import Iterable, Set

from fastapi import UploadFile

import yaml

from ..config import (
    DATASETS_DIR,
    PROJECTS_DIR,
    ALLOWED_EXTS,
    MAX_FILES_PER_UPLOAD,
)

BASE_MODEL_EXTS = {".pt", ".pth"}

# ---------- utils ----------
def _write_and_hash(src_file, dst_path: Path, chunk_size: int = 1 << 20) -> str:
    """
    เขียนไฟล์ลงปลายทางแบบสตรีม พร้อมคำนวณแฮช (blake2b) ในคราวเดียว
    คืนค่า digest hex string
    """
    h = hashlib.blake2b(digest_size=16)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    src_file.seek(0)
    with dst_path.open("wb") as f:
        while True:
            chunk = src_file.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
            f.write(chunk)
    src_file.seek(0)  # คืนพอยน์เตอร์ เผื่อ FastAPI จะใช้งานต่อ
    return h.hexdigest()


# ---------- SCENE (ของเดิม รักษาให้ใช้ได้ต่อ) ----------
def get_raw_dir(scene_id: str) -> Path:
    return DATASETS_DIR / scene_id / "raw"


def save_uploads(
    scene_id: str,
    files: Iterable[UploadFile],
    dedup: bool = True,
    dedup_content: bool = True,
) -> int:
    """
    เซฟรูปลง data/datasets/<scene_id>/raw/
    - อนุญาตเฉพาะ ALLOWED_EXTS
    - dedup: กันชื่อซ้ำ
    - dedup_content: กันไฟล์เนื้อหาเหมือนกันด้วยแฮช (ชื่อไม่เหมือนก็กัน)
    """
    target = get_raw_dir(scene_id)
    target.mkdir(parents=True, exist_ok=True)

    saved = 0
    seen_names: Set[str] = set()
    seen_hashes: Set[str] = set()

    for idx, up in enumerate(files):
        if idx >= MAX_FILES_PER_UPLOAD:
            break

        name = Path(up.filename or f"upload_{idx}").name
        ext = Path(name).suffix.lower()
        if ext not in ALLOWED_EXTS:
            continue

        if dedup and name in seen_names:
            continue

        dst = target / name

        # คำนวณแฮชและเขียนไฟล์ในคราวเดียว
        file_hash = _write_and_hash(up.file, dst)

        # กันไฟล์เนื้อหาซ้ำ (แม้ชื่อไม่เหมือน) -> ลบไฟล์ที่เพิ่งเขียนทิ้ง ถ้าซ้ำ
        if dedup_content:
            if file_hash in seen_hashes:
                try:
                    dst.unlink(missing_ok=True)
                finally:
                    continue
            seen_hashes.add(file_hash)

        seen_names.add(name)
        saved += 1

    return saved


# ---------- PROJECT (ของใหม่) ----------
def get_project_raw_dir(project_id: str) -> Path:
    return PROJECTS_DIR / project_id / "raw"


def save_project_uploads(
    project_id: str,
    files: Iterable[UploadFile],
    dedup: bool = True,
    dedup_content: bool = True,
) -> int:
    """
    เซฟรูปลง data/projects/<project_id>/raw/ (เหมือน save_uploads แต่แยกต่อโปรเจกต์)
    """
    target = get_project_raw_dir(project_id)
    target.mkdir(parents=True, exist_ok=True)

    saved = 0
    seen_names: Set[str] = set()
    seen_hashes: Set[str] = set()

    for idx, up in enumerate(files):
        if idx >= MAX_FILES_PER_UPLOAD:
            break

        name = Path(up.filename or f"upload_{idx}").name
        ext = Path(name).suffix.lower()
        if ext not in ALLOWED_EXTS:
            continue

        if dedup and name in seen_names:
            continue

        dst = target / name
        file_hash = _write_and_hash(up.file, dst)

        if dedup_content:
            if file_hash in seen_hashes:
                try:
                    dst.unlink(missing_ok=True)
                finally:
                    continue
            seen_hashes.add(file_hash)

        seen_names.add(name)
        saved += 1

    return saved

def list_project_images(project_id: str) -> list[dict]:
    raw = PROJECTS_DIR / project_id / "raw"
    if not raw.exists():
        return []
    items = []
    for p in raw.iterdir():
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
            items.append({
                "filename": p.name,
                "url": f"/static/projects/{project_id}/raw/{p.name}"
            })
    return items

def list_scene_images(scene_id: str) -> list[dict]:
    raw = get_raw_dir(scene_id)
    if not raw.exists():
        return []
    items = []
    for p in raw.iterdir():
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
            items.append({
                "filename": p.name,
                "url": f"/static/datasets/{scene_id}/raw/{p.name}"
            })
    return items


def _validate_filename(name: str):
    # กัน ../ หรือ โฟลเดอร์ย่อย และไฟล์ที่ไม่ใช่รูป
    if not name or "/" in name or "\\" in name or name in (".", "..") or name.startswith("."):
        raise ValueError("invalid filename")
    if Path(name).suffix.lower() not in ALLOWED_EXTS:
        raise ValueError("not an allowed image")

def delete_project_image(project_id: str, filename: str) -> bool:
    _validate_filename(filename)
    raw = PROJECTS_DIR / project_id / "raw"
    path = raw / filename
    if path.is_file():
        path.unlink()
        return True
    return False

def delete_scene_image(scene_id: str, filename: str) -> bool:
    _validate_filename(filename)
    raw = get_raw_dir(scene_id)
    path = raw / filename
    if path.is_file():
        path.unlink()
        return True
    return False

def save_project_base_model(project_id: str, file: UploadFile) -> dict:
    project_dir = PROJECTS_DIR / project_id
    if not project_dir.exists():
        raise FileNotFoundError('project not found')

    filename = Path(file.filename or 'model.pt').name
    ext = Path(filename).suffix.lower()
    if ext not in BASE_MODEL_EXTS:
        raise ValueError('file must be .pt or .pth')

    imports_dir = project_dir / 'imports'
    imports_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime('%Y%m%d-%H%M%S')
    model_id = f"import-{ts}-{uuid.uuid4().hex[:8]}"
    dest_dir = imports_dir / model_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / 'model.pt'

    file_hash = _write_and_hash(file.file, dest_path)

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

    return {'model_id': model_id, 'filename': filename}
