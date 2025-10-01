from __future__ import annotations
import json, re, time
from pathlib import Path
from typing import Dict, List, Optional

from ..config import PROJECTS_DIR
import shutil

VALID_TRAINING_MODES = {"anomaly", "finetune"}
DEFAULT_TRAINING_MODE = "anomaly"


def _normalize_training_mode(mode: Optional[str], *, strict: bool = False) -> str:
    if mode is None:
        return DEFAULT_TRAINING_MODE
    normalized = str(mode).strip().lower()
    if normalized in VALID_TRAINING_MODES:
        return normalized
    if strict:
        raise ValueError(f"unsupported training_mode={mode!r}")
    return DEFAULT_TRAINING_MODE


def _slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9\-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "project"


def project_dir(project_id: str) -> Path:
    return PROJECTS_DIR / project_id


def create_project(name: str, description: Optional[str] = None, training_mode: str = DEFAULT_TRAINING_MODE) -> Dict:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

    mode = _normalize_training_mode(training_mode, strict=True)

    ts = time.strftime("%Y%m%d-%H%M%S")
    pid = f"{_slugify(name)}-{ts}"
    pdir = project_dir(pid)
    (pdir / "raw").mkdir(parents=True, exist_ok=True)
    (pdir / "models").mkdir(exist_ok=True)
    (pdir / "preview").mkdir(exist_ok=True)

    meta = {
        "project_id": pid,
        "name": name,
        "description": description,
        "created_at": ts,
        "last_model_id": None,
        "training_mode": mode,
    }
    (pdir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


def list_projects() -> List[Dict]:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    out: List[Dict] = []
    for p in PROJECTS_DIR.iterdir():
        if not p.is_dir():
            continue
        meta_path = p / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        meta["training_mode"] = _normalize_training_mode(meta.get("training_mode"))
        meta["num_images"] = len(list((p / "raw").glob("*")))
        out.append(meta)
    out.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return out


def get_project(project_id: str) -> Dict:
    pdir = project_dir(project_id)
    meta_path = pdir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError("project not found")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["training_mode"] = _normalize_training_mode(meta.get("training_mode"))
    meta["num_images"] = len(list((pdir / "raw").glob("*")))
    return meta


def set_last_model(project_id: str, model_id: str) -> None:
    pdir = project_dir(project_id)
    meta_path = pdir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError("project not found")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["training_mode"] = _normalize_training_mode(meta.get("training_mode"))
    meta["last_model_id"] = model_id
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def delete_project(project_id: str) -> None:
    pdir = project_dir(project_id)
    if not pdir.exists():
        raise FileNotFoundError("project not found")
    shutil.rmtree(pdir)
