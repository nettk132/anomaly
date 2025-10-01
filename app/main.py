from __future__ import annotations
import os
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

# Schemas & configuration
from .schemas import (
    TrainRequest, TrainResponse, JobStatus,
    TestResponse, TestItem,
    ProjectCreate, ProjectInfo, TrainProjectRequest, ModelInfo,
)
from .config import MODELS_DIR, PROJECTS_DIR, TMP_DIR
from .services.storage import (
    save_uploads, get_raw_dir,
    save_project_uploads, get_project_raw_dir,
    save_project_base_model,
    list_project_images, list_scene_images,
    delete_project_image, delete_scene_image,
)
from .services.jobs import JOBS
from .services.training import train_job, train_job_project, resolve_model_checkpoint
from .services.inference import test_images
from .services.projects import create_project, delete_project, get_project, list_projects
from .services.models import list_models

app = FastAPI(title=os.getenv("API_TITLE", "Anomaly Detection Backend"))

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

@app.get("/")
def root():
    """Redirect to the UI dashboard if available, otherwise the API docs."""
    return RedirectResponse(url="/ui/projects.html" if UI_DIR.exists() else "/docs")

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
async def upload(scene_id: str, files: List[UploadFile] = File(...)):
    """Upload scene-scoped images used for legacy training."""
    count = save_uploads(scene_id, files)
    if count == 0:
        raise HTTPException(status_code=400, detail="No valid images uploaded")
    return {"saved": count, "scene_id": scene_id}

# === Legacy scene training ===
@app.post("/train", response_model=TrainResponse)
async def start_train(req: TrainRequest):
    """Queue a training job for a legacy scene dataset."""
    raw_dir = get_raw_dir(req.scene_id)
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
def get_project_api(project_id: str):
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
async def upload_project(project_id: str, files: List[UploadFile] = File(...)):
    """Upload project-scoped reference images."""
    count = save_project_uploads(project_id, files)
    if count == 0:
        raise HTTPException(status_code=400, detail="No valid images uploaded")
    return {"saved": count, "project_id": project_id}

@app.post("/projects/{project_id}/base-models/upload")
async def upload_project_base_model_api(project_id: str, file: UploadFile = File(...)):
    """Upload a base model checkpoint for project fine-tuning."""
    try:
        info = save_project_base_model(project_id, file)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="project not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return info

@app.get("/models", response_model=List[ModelInfo])
def list_models_api(mode: str | None = None, project_id: str | None = None):
    """List models with optional filtering by mode or project."""
    try:
        return list_models(mode=mode, project_id=project_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.post("/projects/{project_id}/train", response_model=TrainResponse)
async def train_project(project_id: str, req: TrainProjectRequest):
    """Queue a training job for the specified project."""
    raw_dir = get_project_raw_dir(project_id)
    if not raw_dir.exists():
        raise HTTPException(status_code=404, detail=f"No dataset for project_id={project_id}")

    try:
        meta = get_project(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="project not found")

    mode = (meta.get("training_mode") or "anomaly").lower()
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
def download_model(model_id: str):
    """Download a scene-mode model checkpoint."""
    path = MODELS_DIR / model_id / "model.pt"
    if not path.exists():
        if PROJECTS_DIR.exists():
            for proj_dir in PROJECTS_DIR.iterdir():
                cand = proj_dir / "models" / model_id / "model.pt"
                if cand.exists():
                    path = cand
                    break
    if not path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    return FileResponse(str(path), media_type="application/octet-stream", filename="model.pt")

@app.get("/projects/{project_id}/models/{model_id}/download")
def download_project_model(project_id: str, model_id: str):
    """Download a project-scoped model checkpoint."""
    path = PROJECTS_DIR / project_id / "models" / model_id / "model.pt"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    return FileResponse(str(path), media_type="application/octet-stream", filename="model.pt")

@app.post("/models/{model_id}/test", response_model=TestResponse)
async def test_model(model_id: str, files: List[UploadFile] = File(...)):
    """Run inference on uploaded images and return structured results."""
    buf_list = []
    for f in files:
        content = await f.read()
        buf_list.append((f.filename, content))

    try:
        results = test_images(model_id, MODELS_DIR, buf_list)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    items = [TestItem(**r) for r in results]
    return TestResponse(model_id=model_id, items=items)

@app.delete("/projects/{project_id}")
def delete_project_api(project_id: str, confirm: str = Query(..., description="Type exact project name to confirm")):
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
def list_project_images_api(project_id: str):
    """List project image metadata for the gallery view."""
    return {"items": list_project_images(project_id)}

@app.get("/datasets/{scene_id}/images")
def list_scene_images_api(scene_id: str):
    """List scene image metadata for the legacy workflow."""
    return {"items": list_scene_images(scene_id)}

@app.delete("/projects/{project_id}/images/{filename}")
def delete_project_image_api(project_id: str, filename: str):
    """Delete a project image if it exists."""
    try:
        ok = delete_project_image(project_id, filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not ok:
        raise HTTPException(status_code=404, detail="file not found")
    return {"ok": True}

@app.delete("/datasets/{scene_id}/images/{filename}")
def delete_scene_image_api(scene_id: str, filename: str):
    """Delete a scene image in the legacy workflow."""
    try:
        ok = delete_scene_image(scene_id, filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not ok:
        raise HTTPException(status_code=404, detail="file not found")
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











