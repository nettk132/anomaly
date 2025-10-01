from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional

import yaml

from ..config import MODELS_DIR, PROJECTS_DIR, DATA_DIR


def _load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _load_threshold(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        for key in ('threshold_score', 'threshold_mse'):
            if isinstance(data, dict) and key in data:
                try:
                    return float(data[key])
                except Exception:
                    continue
    except Exception:
        pass
    return None


def _parse_created(model_id: str, cfg: dict) -> Optional[str]:
    created = cfg.get('created_at')
    if created:
        return str(created)
    if '-' in model_id:
        parts = model_id.split('-')
        if len(parts) >= 3 and parts[-1].isdigit() and len(parts[-1]) == 6 and parts[-2].isdigit() and len(parts[-2]) == 8:
            return "{}-{}".format(parts[-2], parts[-1])
        tail = parts[-1]
        if tail.isdigit():
            return tail
    return None


def _relative_to_data(path: Path) -> str:
    try:
        rel = path.relative_to(DATA_DIR)
        return rel.as_posix()
    except Exception:
        return path.as_posix()


def _collect_from_dir(model_dir: Path, mode: str, *, scene_id: Optional[str] = None, project_id: Optional[str] = None) -> dict:
    cfg_path = model_dir / 'config.yaml'
    cfg = _load_config(cfg_path)
    thr = _load_threshold(model_dir / 'threshold.json')
    training_mode = cfg.get('training_mode')
    if isinstance(training_mode, str):
        training_mode = training_mode.lower()
    else:
        training_mode = None
    if mode == 'project' and training_mode not in ('anomaly', 'finetune'):
        training_mode = 'anomaly'
    info = {
        'model_id': model_dir.name,
        'mode': mode,
        'scene_id': cfg.get('scene_id', scene_id),
        'project_id': cfg.get('project_id', project_id),
        'created_at': _parse_created(model_dir.name, cfg),
        'img_size': cfg.get('img_size'),
        'epochs_run': cfg.get('epochs_run'),
        'lr': cfg.get('lr'),
        'threshold': thr,
        'note': cfg.get('note'),
        'base_model_id': cfg.get('base_model_id'),
        'training_mode': training_mode,
        'path': _relative_to_data(model_dir),
    }
    return info


def list_models(*, mode: Optional[str] = None, project_id: Optional[str] = None) -> List[dict]:
    if mode not in (None, 'scene', 'project'):
        raise ValueError('mode must be "scene" or "project" if provided')

    items: List[dict] = []

    if mode in (None, 'scene') and MODELS_DIR.exists():
        for entry in MODELS_DIR.iterdir():
            if entry.is_dir():
                items.append(_collect_from_dir(entry, 'scene'))

    if mode in (None, 'project') and PROJECTS_DIR.exists():
        for proj_dir in PROJECTS_DIR.iterdir():
            if not proj_dir.is_dir():
                continue
            pid = proj_dir.name
            if project_id and pid != project_id:
                continue
            models_dir = proj_dir / 'models'
            if models_dir.exists():
                for entry in models_dir.iterdir():
                    if entry.is_dir():
                        items.append(_collect_from_dir(entry, 'project', project_id=pid))
            imports_dir = proj_dir / 'imports'
            if imports_dir.exists():
                for entry in imports_dir.iterdir():
                    if entry.is_dir():
                        items.append(_collect_from_dir(entry, 'project', project_id=pid))

    def sort_key(item: dict):
        created = item.get('created_at') or ''
        return created, item.get('model_id')

    items.sort(key=sort_key, reverse=True)
    return items
