from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

from ..config import PROJECTS_DIR
from ..utils import PathTraversalError, safe_join, validate_slug
from .storage import normalize_stored_filename


def _project_dir(project_id: str) -> Path:
    validate_slug(project_id, name="project_id")
    try:
        return safe_join(PROJECTS_DIR, project_id, must_exist=True)
    except (FileNotFoundError, PathTraversalError):
        raise FileNotFoundError("project not found")


def _labels_dir(project_id: str) -> Path:
    return safe_join(_project_dir(project_id), "labels", must_exist=False)


def _classes_path(project_id: str) -> Path:
    return _labels_dir(project_id) / "classes.json"


def _raw_image_path(project_id: str, filename: str) -> Path:
    normalized = normalize_stored_filename(filename)
    return safe_join(_project_dir(project_id), "raw", normalized, must_exist=False)


def _sanitize_filename(filename: str) -> str:
    return normalize_stored_filename(filename)


def _label_path(project_id: str, filename: str) -> Path:
    safe_name = _sanitize_filename(filename)
    return safe_join(_project_dir(project_id), "labels", f"{safe_name}.json", must_exist=False)


def _ensure_boxes_valid(boxes: List[Dict]) -> List[Dict[str, object]]:
    validated: List[Dict[str, object]] = []
    for idx, raw in enumerate(boxes or []):
        if not isinstance(raw, dict):
            raise ValueError(f"box #{idx+1} must be an object")
        label = str(raw.get("label", "")).strip()
        if not label:
            raise ValueError(f"box #{idx+1} missing label")
        try:
            x = float(raw.get("x"))
            y = float(raw.get("y"))
            w = float(raw.get("width"))
            h = float(raw.get("height"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"box #{idx+1} must include numeric x/y/width/height") from exc
        if w <= 0 or h <= 0:
            raise ValueError(f"box #{idx+1} width/height must be > 0")
        if x < 0 or y < 0 or x > 1 or y > 1 or w > 1 or h > 1:
            raise ValueError(f"box #{idx+1} coordinates must be between 0 and 1")
        if x + w > 1.0001 or y + h > 1.0001:
            raise ValueError(f"box #{idx+1} exceeds image bounds")
        bid = raw.get("id")
        box_id = str(bid) if bid is not None else None
        validated.append(
            {
                "id": box_id,
                "label": label,
                "x": x,
                "y": y,
                "width": w,
                "height": h,
            }
        )
    return validated


def load_label_classes(project_id: str) -> List[str]:
    """Load persisted class list if available."""
    path = _classes_path(project_id)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError("invalid classes file") from exc
    if not isinstance(data, list):
        raise ValueError("invalid classes file format")
    out: List[str] = []
    seen: set[str] = set()
    for raw in data:
        label = str(raw or "").strip()
        if not label:
            continue
        key = label.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(label)
    return out


def save_label_classes(project_id: str, classes: List[str]) -> List[str]:
    """Persist the given class list (deduplicated, trimmed)."""
    labels_dir = _labels_dir(project_id)
    labels_dir.mkdir(parents=True, exist_ok=True)

    cleaned: List[str] = []
    seen: set[str] = set()
    for raw in classes or []:
        label = str(raw or "").strip()
        if not label:
            continue
        key = label.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(label)

    path = _classes_path(project_id)
    path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
    return cleaned


def merge_label_classes(project_id: str, labels: List[str]) -> List[str]:
    """Merge new labels into the persisted class list."""
    try:
        existing = load_label_classes(project_id)
    except ValueError:
        existing = []
    if not labels:
        return existing
    seen = {label.casefold(): label for label in existing}
    updated = list(existing)
    changed = False
    for raw in labels:
        label = str(raw or "").strip()
        if not label:
            continue
        key = label.casefold()
        if key in seen:
            continue
        seen[key] = label
        updated.append(label)
        changed = True
    if changed:
        return save_label_classes(project_id, updated)
    return existing


def save_image_labels(
    project_id: str,
    filename: str,
    image_width: int,
    image_height: int,
    boxes: List[Dict],
) -> Dict:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    safe_name = _sanitize_filename(filename)
    raw_path = _raw_image_path(project_id, safe_name)
    if not raw_path.exists():
        raise FileNotFoundError("image not found in project dataset")

    labels_dir = _labels_dir(project_id)
    labels_dir.mkdir(parents=True, exist_ok=True)

    validated_boxes = _ensure_boxes_valid(boxes)
    payload = {
        "filename": safe_name,
        "image_width": int(image_width),
        "image_height": int(image_height),
        "boxes": validated_boxes,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    label_path = labels_dir / f"{safe_name}.json"
    label_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    merge_label_classes(project_id, [box["label"] for box in validated_boxes])
    return payload


def load_image_labels(project_id: str, filename: str) -> Dict | None:
    path = _label_path(project_id, filename)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError
    except Exception as exc:
        raise ValueError("invalid label file") from exc
    return data


def list_project_labels(project_id: str) -> Tuple[List[Dict], List[str]]:
    labels_dir = _labels_dir(project_id)
    items: List[Dict] = []
    classes: set[str] = set()
    if labels_dir.exists():
        for json_path in labels_dir.glob("*.json"):
            if json_path.name == "classes.json":
                continue
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    continue
                boxes = data.get("boxes") or []
                if isinstance(boxes, list):
                    for b in boxes:
                        if isinstance(b, dict):
                            label = str(b.get("label") or "").strip()
                            if label:
                                classes.add(label)
                items.append(data)
            except Exception:
                continue
    items.sort(key=lambda d: d.get("filename") or "")
    persisted = []
    try:
        persisted = load_label_classes(project_id)
    except ValueError:
        persisted = []
    for label in persisted:
        classes.add(label)
    return items, sorted(classes)
