# app/services/storage.py
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Iterable, Set

from fastapi import UploadFile

from ..config import (
    DATASETS_DIR,
    PROJECTS_DIR,
    ALLOWED_EXTS,
    MAX_FILES_PER_UPLOAD,
)

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
