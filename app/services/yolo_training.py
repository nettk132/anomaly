from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - runtime dependency
    YOLO = None  # type: ignore
    _ultralytics_import_error = exc
else:
    _ultralytics_import_error = None

from ..config import PROJECTS_DIR
from ..utils import PathTraversalError, safe_join, validate_slug
from .yolo_dataset import dataset_dir, describe_dataset, DATA_YAML_NAME

try:
    from .projects import set_last_model  # type: ignore
except Exception:  # pragma: no cover
    set_last_model = None  # type: ignore


class MissingUltralyticsError(RuntimeError):
    pass


def _ensure_ultralytics():
    if YOLO is None:
        raise MissingUltralyticsError(
            "ultralytics package is required for YOLO training. Install with 'pip install ultralytics'."
        ) from _ultralytics_import_error

def ensure_ultralytics_available() -> None:
    """Public guard to confirm the ultralytics dependency is present."""
    _ensure_ultralytics()


def _get_dataset_yaml(project_id: str) -> tuple[Path, Dict[str, Any]]:
    info = describe_dataset(project_id)
    if not info.get("yaml_path"):
        raise ValueError("YOLO dataset is not ready; upload a dataset zip with train/val folders first")
    yaml_path = dataset_dir(project_id) / DATA_YAML_NAME
    if not yaml_path.exists():
        raise ValueError("dataset yaml not found; re-upload the dataset")
    return yaml_path, info


def _relative_to_data(path: Path) -> str:
    try:
        return path.relative_to(PROJECTS_DIR.parent).as_posix()
    except ValueError:
        return path.as_posix()


def _jsonable_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    flat: Dict[str, float] = {}
    for key, value in metrics.items():
        try:
            flat[key] = float(value)
        except Exception:
            continue
    return flat


def train_yolo_job(
    job,
    project_id: str,
    epochs: int,
    img_size: int,
    batch_size: Optional[int],
    model_variant: str,
    lr0: Optional[float],
    conf: float,
    iou: float,
) -> None:
    validate_slug(project_id, name="project_id")
    _ensure_ultralytics()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yaml_path, dataset_info = _get_dataset_yaml(project_id)

    model = YOLO(model_variant)
    job.detail = f"training YOLO ({model_variant}) on {project_id}"

    runs_root = safe_join(PROJECTS_DIR, project_id, "yolo", "runs", must_exist=False)
    runs_root.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"train-{ts}"

    progress_total = max(1, epochs)

    def _on_epoch_end(trainer):  # type: ignore[override]
        try:
            current = getattr(trainer, "epoch", 0) + 1
            total = getattr(trainer, "epochs", progress_total)
            job.progress = min(0.98, current / max(1, total))
        except Exception:
            job.progress = min(0.98, job.progress + (1.0 / progress_total))

    callbacks = {"on_train_epoch_end": _on_epoch_end}

    train_kwargs: Dict[str, Any] = dict(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=img_size,
        project=str(runs_root),
        name=run_name,
        exist_ok=False,
        device=device,
        verbose=False,
        callbacks=callbacks,
    )
    if batch_size:
        train_kwargs["batch"] = batch_size
    if lr0:
        train_kwargs["lr0"] = lr0

    # Run training
    results = model.train(**train_kwargs)
    trainer = model.trainer
    best_path = Path(trainer.best) if trainer and getattr(trainer, "best", None) else None
    if not best_path or not best_path.exists():
        # fall back to last weights saved under runs directory
        weights_dir = runs_root / run_name / "weights"
        if (weights_dir / "best.pt").exists():
            best_path = weights_dir / "best.pt"
        elif (weights_dir / "last.pt").exists():
            best_path = weights_dir / "last.pt"
        else:
            raise RuntimeError("unable to locate YOLO weights after training")

    model_id = f"{project_id}-yolo-{ts}"
    model_id = validate_slug(model_id, name="model_id")
    try:
        out_dir = safe_join(PROJECTS_DIR, project_id, "models", model_id, must_exist=False)
    except PathTraversalError as exc:
        raise RuntimeError(f"Invalid model output path: {exc}") from exc
    out_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(best_path, out_dir / "model.pt")

    metrics = getattr(trainer, "metrics", {}) if trainer else {}
    metrics = _jsonable_metrics(metrics)
    if hasattr(results, "results_dict"):
        try:
            for k, v in results.results_dict.items():
                metrics.setdefault(k, float(v))
        except Exception:
            pass

    config = {
        "project_id": project_id,
        "created_at": ts,
        "training_mode": "yolo",
        "device": device,
        "yolo": {
            "model_variant": model_variant,
            "epochs": getattr(trainer, "epochs", epochs),
            "img_size": img_size,
            "batch_size": getattr(trainer, "batch", batch_size),
            "dataset_yaml": _relative_to_data(yaml_path),
            "dataset_info": dataset_info,
            "classes": dataset_info.get("classes", []),
            "metrics": metrics,
            "conf_threshold": float(conf),
            "iou_threshold": float(iou),
        },
    }
    with (out_dir / "config.yaml").open("w", encoding="utf-8") as f:
        import yaml
        yaml.safe_dump(config, f, allow_unicode=True)

    metrics_payload = {
        "type": "yolo",
        "metrics": metrics,
        "created_at": ts,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if callable(set_last_model):
        try:
            set_last_model(project_id, model_id)  # type: ignore
        except Exception:
            pass

    job.model_id = model_id
    job.detail = f"finished: {model_id}"
    job.progress = 1.0
