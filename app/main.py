from __future__ import annotations
import logging
import os
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Annotated, List

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request, Path as PathParam
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles

# Schemas & configuration
from .schemas import (
    TrainRequest, TrainResponse, JobStatus,
    TestResponse, TestItem,
    ProjectCreate, ProjectInfo, TrainProjectRequest, ModelInfo,
    ImageLabel, LabelListResponse, SaveLabelsRequest,
    LabelClassesRequest, LabelClassesResponse,
    YoloTrainRequest, YoloDatasetInfo,
)
from .config import MODELS_DIR, PROJECTS_DIR, TMP_DIR
from .utils import PathTraversalError, safe_join, SLUG_PATTERN, MAX_SLUG_LENGTH
from .services.storage import (
    save_uploads, get_raw_dir,
    save_project_uploads, get_project_raw_dir,
    save_project_base_model,
    list_project_images, list_scene_images,
    delete_project_image, delete_scene_image,
)
from .services.labels import (
    list_project_labels,
    load_image_labels,
    save_image_labels,
    load_label_classes,
    save_label_classes,
)
from .services.jobs import JOBS
from .services.training import train_job, train_job_project, resolve_model_checkpoint
from .services.yolo_training import train_yolo_job, MissingUltralyticsError, ensure_ultralytics_available
from .services.yolo_dataset import save_yolo_dataset, describe_dataset as describe_yolo_dataset, clear_dataset as clear_yolo_dataset
from .services.inference import test_images
from .services.projects import create_project, delete_project, get_project, list_projects
from .services.models import list_models

app = FastAPI(title=os.getenv("API_TITLE", "Anomaly Detection Backend"))
logger = logging.getLogger("app.security")
audit_logger = logging.getLogger("app.audit")
_RATE_LIMIT_BUCKETS: dict[str, deque[float]] = defaultdict(deque)
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_FAILURES = 8
UPLOAD_RATE_LIMIT = 20
DELETE_RATE_LIMIT = 40


def _register_failed_download(request: Request, reason: str) -> None:
    logger.warning("Blocked model download from %s: %s", request.client.host if request.client else "unknown", reason)
    audit_logger.warning("Download blocked: %s (%s)", reason, request.url.path)
    _enforce_rate_limit(request, "download-fail", RATE_LIMIT_MAX_FAILURES)


def _enforce_rate_limit(request: Request, bucket: str, limit: int, *, window: int = RATE_LIMIT_WINDOW_SECONDS) -> None:
    client = request.client.host if request.client else "unknown"
    key = f"{bucket}:{client}"
    queue = _RATE_LIMIT_BUCKETS[key]
    now = time.time()
    queue.append(now)
    while queue and now - queue[0] > window:
        queue.popleft()
    if len(queue) > limit:
        raise HTTPException(status_code=429, detail="Too many requests, try later")



def _resolve_download_path(base_dir: Path, model_id: str) -> Path:
    for ext in ("model.safetensors", "model.pt"):
        try:
            candidate = safe_join(base_dir, model_id, ext)
        except (FileNotFoundError, PathTraversalError):
            continue
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Model {model_id} not found under {base_dir}")


def _reset_failed_downloads() -> None:
    _RATE_LIMIT_BUCKETS.clear()

@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError):
    if any(error.get('loc', [None])[0] == 'path' for error in exc.errors()):
        try:
            path_value = request.url.path
            if path_value.startswith('/models/') or ('/models/' in path_value and path_value.startswith('/projects/')):
                _register_failed_download(request, f'invalid path parameter: {path_value}')
        except HTTPException as rate_limited:
            return JSONResponse(status_code=rate_limited.status_code, content={'detail': rate_limited.detail})
        return JSONResponse(status_code=404, content={'detail': 'Not found'})
    return JSONResponse(status_code=422, content={'detail': exc.errors()})


SlugPathParam = Annotated[
    str,
    PathParam(..., pattern=SLUG_PATTERN, min_length=1, max_length=MAX_SLUG_LENGTH),
]

# Allow broad CORS during initial development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve bundled UI assets when available.
ROOT = Path(__file__).resolve().parents[1]
UI_DIR = ROOT / "ui"
if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")

# Expose data directory as static assets.
DATA_DIR = ROOT / "data"
if DATA_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(DATA_DIR)), name="static")

def _to_yolo_dataset_info(data: dict) -> YoloDatasetInfo:
    classes = data.get("classes") or []
    return YoloDatasetInfo(
        ready=bool(data.get("ready") or data.get("yaml_path")),
        train_images=int(data.get("train_images") or 0),
        val_images=int(data.get("val_images") or 0),
        classes=[str(c) for c in classes],
        yaml_path=str(data.get("yaml_path")) if data.get("yaml_path") else None,
        note=data.get("note"),
        updated_at=data.get("updated_at"),
    )

@app.get("/")
def root():
    """Redirect to the UI dashboard if available, otherwise the API docs."""
    return RedirectResponse(url="/ui/projects" if UI_DIR.exists() else "/docs")


@app.get("/favicon.ico")
def favicon():
    """Return an empty response so browsers stop requesting /favicon.ico."""
    return Response(status_code=204)

@app.get("/health")
def health():
    """Basic health check for monitoring."""
    return {"ok": True}

# === Legacy scene uploads ===
@app.post("/datasets/{scene_id}/upload")
async def upload(request: Request, scene_id: SlugPathParam, files: List[UploadFile] = File(...)):
    """Upload scene-scoped images used for legacy training."""
    _enforce_rate_limit(request, "scene-upload", UPLOAD_RATE_LIMIT)
    try:
        count = save_uploads(scene_id, files)
    except ValueError:
        audit_logger.warning("Upload rejected (invalid scene) from %s", request.client.host if request.client else "unknown")
        raise HTTPException(status_code=400, detail="Invalid scene_id")
    if count == 0:
        audit_logger.warning(
            "Upload contained no valid images (scene=%s, client=%s)",
            scene_id,
            request.client.host if request.client else "unknown",
        )
        raise HTTPException(status_code=400, detail="No valid images uploaded")
    audit_logger.info(
        "Scene upload success (scene=%s, saved=%d, client=%s)",
        scene_id,
        count,
        request.client.host if request.client else "unknown",
    )
    return {"saved": count, "scene_id": scene_id}

# === Legacy scene training ===
@app.post("/train", response_model=TrainResponse)
async def start_train(req: TrainRequest):
    """Queue a training job for a legacy scene dataset."""
    try:
        raw_dir = get_raw_dir(req.scene_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"No dataset for scene_id={req.scene_id}")
    if not raw_dir.exists():
        raise HTTPException(status_code=404, detail=f"No dataset for scene_id={req.scene_id}")
    job = JOBS.create()
    JOBS.run_in_thread(job, train_job, req.scene_id, raw_dir, req.img_size, req.epochs, req.lr)
    return TrainResponse(job_id=job.id)

# === Project endpoints ===
@app.get("/projects", response_model=List[ProjectInfo])
def list_projects_api():
    """Return all projects with metadata."""
    return [ProjectInfo(**p) for p in list_projects()]

@app.get("/projects/{project_id}", response_model=ProjectInfo)
def get_project_api(project_id: SlugPathParam):
    """Fetch a single project by identifier."""
    try:
        return ProjectInfo(**get_project(project_id))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="project not found")

@app.post("/projects", response_model=ProjectInfo)
def create_project_api(req: ProjectCreate):
    """Create a new project and return its metadata."""
    meta = create_project(req.name, req.description, req.training_mode)
    return ProjectInfo(**meta)

@app.post("/projects/{project_id}/upload")
async def upload_project(request: Request, project_id: SlugPathParam, files: List[UploadFile] = File(...)):
    """Upload project-scoped reference images."""
    _enforce_rate_limit(request, "project-upload", UPLOAD_RATE_LIMIT)
    try:
        count = save_project_uploads(project_id, files)
    except ValueError:
        audit_logger.warning("Project upload rejected (invalid project) client=%s", request.client.host if request.client else "unknown")
        raise HTTPException(status_code=404, detail="project not found")
    if count == 0:
        audit_logger.warning(
            "Project upload empty (project=%s, client=%s)",
            project_id,
            request.client.host if request.client else "unknown",
        )
        raise HTTPException(status_code=400, detail="No valid images uploaded")
    audit_logger.info(
        "Project upload success (project=%s, saved=%d, client=%s)",
        project_id,
        count,
        request.client.host if request.client else "unknown",
    )
    return {"saved": count, "project_id": project_id}

@app.post("/projects/{project_id}/base-models/upload")
async def upload_project_base_model_api(request: Request, project_id: SlugPathParam, file: UploadFile = File(...)):
    """Upload a base model checkpoint for project fine-tuning."""
    _enforce_rate_limit(request, "base-model-upload", UPLOAD_RATE_LIMIT)
    try:
        info = save_project_base_model(project_id, file)
    except FileNotFoundError:
        audit_logger.warning("Base model upload rejected (missing project=%s, client=%s)", project_id, request.client.host if request.client else "unknown")
        raise HTTPException(status_code=404, detail="project not found")
    except ValueError as exc:
        audit_logger.warning(
            "Base model upload rejected (project=%s, client=%s, reason=%s)",
            project_id,
            request.client.host if request.client else "unknown",
            exc,
        )
        raise HTTPException(status_code=400, detail=str(exc))
    audit_logger.info(
        "Base model upload success (project=%s, model_id=%s, client=%s)",
        project_id,
        info.get("model_id"),
        request.client.host if request.client else "unknown",
    )
    return info

@app.get("/projects/{project_id}/yolo/dataset", response_model=YoloDatasetInfo)
def get_yolo_dataset_api(project_id: SlugPathParam):
    """Return YOLO dataset summary for a project."""
    try:
        meta = get_project(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="project not found")
    if (meta.get("training_mode") or "anomaly").lower() != "yolo":
        raise HTTPException(status_code=400, detail="Project is not configured for YOLO training")
    info = describe_yolo_dataset(project_id)
    return _to_yolo_dataset_info(info)


@app.post("/projects/{project_id}/yolo/dataset/upload", response_model=YoloDatasetInfo)
async def upload_yolo_dataset_api(project_id: SlugPathParam, file: UploadFile = File(...)):
    """Upload a ZIP archive containing a YOLO dataset (train/val folders)."""
    try:
        meta = get_project(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="project not found")
    if (meta.get("training_mode") or "anomaly").lower() != "yolo":
        raise HTTPException(status_code=400, detail="Project is not configured for YOLO training")
    try:
        summary = save_yolo_dataset(project_id, file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _to_yolo_dataset_info(summary)


@app.delete("/projects/{project_id}/yolo/dataset")
def clear_yolo_dataset_api(project_id: SlugPathParam):
    """Remove uploaded YOLO dataset for a project."""
    try:
        meta = get_project(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="project not found")
    if (meta.get("training_mode") or "anomaly").lower() != "yolo":
        raise HTTPException(status_code=400, detail="Project is not configured for YOLO training")
    clear_yolo_dataset(project_id)
    return {"ok": True}


@app.post("/projects/{project_id}/yolo/train", response_model=TrainResponse)
async def start_yolo_training(project_id: SlugPathParam, req: YoloTrainRequest):
    """Queue a YOLO training job for the given project."""
    try:
        meta = get_project(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="project not found")
    if (meta.get("training_mode") or "anomaly").lower() != "yolo":
        raise HTTPException(status_code=400, detail="Project is not configured for YOLO training")
    info = describe_yolo_dataset(project_id)
    if not info.get("yaml_path"):
        raise HTTPException(status_code=400, detail="Upload a YOLO dataset before training")
    try:
        ensure_ultralytics_available()
    except MissingUltralyticsError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    job = JOBS.create()
    job.detail = "queued (yolo)"
    JOBS.run_in_thread(
        job,
        train_yolo_job,
        project_id,
        req.epochs,
        req.img_size,
        req.batch,
        req.model_variant,
        req.lr0,
        req.conf,
        req.iou,
    )
    return TrainResponse(job_id=job.id)


@app.get("/models", response_model=List[ModelInfo])
def list_models_api(
    mode: str | None = Query(default=None),
    project_id: str | None = Query(
        default=None,
        min_length=1,
        max_length=MAX_SLUG_LENGTH,
        pattern=SLUG_PATTERN,
    ),
):
    """List models with optional filtering by mode or project."""
    try:
        return list_models(mode=mode, project_id=project_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.post("/projects/{project_id}/train", response_model=TrainResponse)
async def train_project(project_id: SlugPathParam, req: TrainProjectRequest):
    """Queue a training job for the specified project."""
    try:
        raw_dir = get_project_raw_dir(project_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"No dataset for project_id={project_id}")
    if not raw_dir.exists():
        raise HTTPException(status_code=404, detail=f"No dataset for project_id={project_id}")

    try:
        meta = get_project(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="project not found")

    mode = (meta.get("training_mode") or "anomaly").lower()
    if mode == "yolo":
        raise HTTPException(status_code=400, detail="Use /projects/{project_id}/yolo/train for YOLO projects")
    if mode not in ("anomaly", "finetune"):
        mode = "anomaly"

    base_model_id = req.base_model_id
    if mode == "finetune":
        if not base_model_id:
            raise HTTPException(status_code=400, detail="base_model_id is required for finetune projects")
        try:
            resolve_model_checkpoint(base_model_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Base model {base_model_id} not found")
    else:
        base_model_id = None

    job = JOBS.create()
    job.detail = f"queued ({mode})"
    JOBS.run_in_thread(
        job,
        train_job_project,
        project_id,
        raw_dir,
        req.img_size,
        req.epochs,
        req.lr,
        mode,
        base_model_id,
    )
    return TrainResponse(job_id=job.id)

# === Job status and model endpoints ===
@app.get("/jobs/{job_id}", response_model=JobStatus)
async def job_status(job_id: str):
    """Retrieve status details for a background training job."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(status=job.status, progress=job.progress, model_id=job.model_id, detail=job.detail)

@app.get("/models/{model_id}/download")
def download_model(request: Request, model_id: SlugPathParam):
    """Download a scene-mode model checkpoint."""
    try:
        path = _resolve_download_path(MODELS_DIR, model_id)
        audit_logger.info(
            "Download success model=%s client=%s",
            path.name,
            request.client.host if request.client else "unknown",
        )
        return FileResponse(str(path), media_type="application/octet-stream", filename=path.name)
    except PathTraversalError as exc:
        _register_failed_download(request, f"path traversal: model_id={model_id}: {exc}")
        raise HTTPException(status_code=404, detail="Model not found")
    except FileNotFoundError:
        pass

    if PROJECTS_DIR.exists():
        for proj_dir in PROJECTS_DIR.iterdir():
            if not proj_dir.is_dir():
                continue
            models_dir = proj_dir / "models"
            if not models_dir.is_dir():
                continue
            try:
                path = _resolve_download_path(models_dir, model_id)
                return FileResponse(str(path), media_type="application/octet-stream", filename=path.name)
            except PathTraversalError as exc:
                _register_failed_download(request, f"path traversal via project: model_id={model_id}: {exc}")
                raise HTTPException(status_code=404, detail="Model not found")
            except FileNotFoundError:
                continue
    _register_failed_download(request, f"model not found: {model_id}")
    raise HTTPException(status_code=404, detail="Model not found")



@app.get("/projects/{project_id}/models/{model_id}/download")
def download_project_model(request: Request, project_id: SlugPathParam, model_id: SlugPathParam):
    """Download a project-scoped model checkpoint."""
    try:
        path = _resolve_download_path(PROJECTS_DIR / project_id / "models", model_id)
        audit_logger.info(
            "Project download success project=%s model=%s client=%s",
            project_id,
            path.name,
            request.client.host if request.client else "unknown",
        )
        return FileResponse(str(path), media_type="application/octet-stream", filename=path.name)
    except PathTraversalError as exc:
        _register_failed_download(request, f"path traversal: project_id={project_id}, model_id={model_id}: {exc}")
        raise HTTPException(status_code=404, detail="Model not found")
    except FileNotFoundError:
        try:
            path = _resolve_download_path(PROJECTS_DIR / project_id / "imports", model_id)
            audit_logger.info(
                "Project import download success project=%s model=%s client=%s",
                project_id,
                path.name,
                request.client.host if request.client else "unknown",
            )
            return FileResponse(str(path), media_type="application/octet-stream", filename=path.name)
        except (FileNotFoundError, PathTraversalError):
            _register_failed_download(request, f"project model not found: {project_id}/{model_id}")
        raise HTTPException(status_code=404, detail="Model not found")



@app.post("/models/{model_id}/test", response_model=TestResponse)
async def test_model(model_id: SlugPathParam, files: List[UploadFile] = File(...)):
    """Run inference on uploaded images and return structured results."""
    buf_list = []
    for f in files:
        content = await f.read()
        buf_list.append((f.filename, content))

    try:
        results = test_images(model_id, MODELS_DIR, buf_list)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    items = [TestItem(**r) for r in results]
    return TestResponse(model_id=model_id, items=items)

@app.delete("/projects/{project_id}")
def delete_project_api(project_id: SlugPathParam, confirm: str = Query(..., description="Type exact project name to confirm")):
    """Delete a project after the caller confirms its name."""
    try:
        info = get_project(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="project not found")
    if confirm != info["name"]:
        raise HTTPException(status_code=400, detail="Confirmation name mismatch")
    delete_project(project_id)
    return {"ok": True}


@app.get("/projects/{project_id}/images")
def list_project_images_api(project_id: SlugPathParam):
    """List project image metadata for the gallery view."""
    return {"items": list_project_images(project_id)}

@app.get("/projects/{project_id}/labels", response_model=LabelListResponse)
def list_project_labels_api(project_id: SlugPathParam):
    """Return saved bounding-box labels for a project."""
    try:
        items, classes = list_project_labels(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="project not found")
    parsed: list[ImageLabel] = []
    for entry in items:
        try:
            parsed.append(ImageLabel(**entry))
        except Exception:
            continue
    return LabelListResponse(items=parsed, classes=classes)

@app.get("/projects/{project_id}/labels/{filename}", response_model=ImageLabel)
def get_image_labels_api(project_id: SlugPathParam, filename: str):
    """Fetch label data for a single image."""
    try:
        data = load_image_labels(project_id, filename)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="project not found or file missing")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if data is None:
        raise HTTPException(status_code=404, detail="labels not found")
    return ImageLabel(**data)

@app.put("/projects/{project_id}/labels/{filename}", response_model=ImageLabel)
def save_image_labels_api(project_id: SlugPathParam, filename: str, req: SaveLabelsRequest):
    """Persist label data for a project image."""
    try:
        data = save_image_labels(
            project_id,
            filename,
            req.image_width,
            req.image_height,
            [box.model_dump() for box in req.boxes],
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="project or image not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return ImageLabel(**data)

@app.get("/projects/{project_id}/labels/classes", response_model=LabelClassesResponse)
def get_label_classes_api(project_id: SlugPathParam):
    """Return the saved label class palette for a project."""
    try:
        classes = load_label_classes(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="project not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return LabelClassesResponse(classes=classes)

@app.put("/projects/{project_id}/labels/classes", response_model=LabelClassesResponse)
def update_label_classes_api(project_id: SlugPathParam, req: LabelClassesRequest):
    """Persist the label class palette for a project."""
    try:
        classes = save_label_classes(project_id, req.classes)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="project not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return LabelClassesResponse(classes=classes)

@app.get("/datasets/{scene_id}/images")
def list_scene_images_api(scene_id: SlugPathParam):
    """List scene image metadata for the legacy workflow."""
    return {"items": list_scene_images(scene_id)}

@app.delete("/projects/{project_id}/images/{filename}")
def delete_project_image_api(request: Request, project_id: SlugPathParam, filename: str):
    """Delete a project image if it exists."""
    _enforce_rate_limit(request, "delete", DELETE_RATE_LIMIT)
    try:
        ok = delete_project_image(project_id, filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not ok:
        raise HTTPException(status_code=404, detail="file not found")
    audit_logger.info(
        "Deleted project image project=%s filename=%s client=%s",
        project_id,
        filename,
        request.client.host if request.client else "unknown",
    )
    return {"ok": True}

@app.delete("/datasets/{scene_id}/images/{filename}")
def delete_scene_image_api(request: Request, scene_id: SlugPathParam, filename: str):
    """Delete a scene image in the legacy workflow."""
    _enforce_rate_limit(request, "delete", DELETE_RATE_LIMIT)
    try:
        ok = delete_scene_image(scene_id, filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not ok:
        raise HTTPException(status_code=404, detail="file not found")
    audit_logger.info(
        "Deleted scene image scene=%s filename=%s client=%s",
        scene_id,
        filename,
        request.client.host if request.client else "unknown",
    )
    return {"ok": True}

@app.middleware("http")
async def no_cache_ui(request, call_next):
    """Prevent browsers from caching UI assets during iteration."""
    resp = await call_next(request)
    if request.url.path.startswith("/ui/"):
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
    return resp

app.mount("/tmp", StaticFiles(directory=str(TMP_DIR), html=False), name="tmp")











