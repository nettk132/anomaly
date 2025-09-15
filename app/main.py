from __future__ import annotations
import os, json
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi import Query
from fastapi import HTTPException


from .schemas import (
    TrainRequest, TrainResponse, JobStatus, RoiPayload,
    TestResponse, TestItem,
    ProjectCreate, ProjectInfo, TrainProjectRequest,
)
from .config import DATASETS_DIR, MODELS_DIR, PROJECTS_DIR
from .config import TMP_DIR
from .services.storage import (
    save_uploads, get_raw_dir,
    save_project_uploads, get_project_raw_dir,
)
from .services.jobs import JOBS
from .services.training import train_job, train_job_project
from .services.inference import test_images
from .services.projects import create_project, list_projects, get_project

app = FastAPI(title=os.getenv("API_TITLE", "Anomaly Detection Backend"))

# CORS (เปิดกว้างสำหรับ MVP)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Optional UI: ถ้ามีโฟลเดอร์ ui/ จะเสิร์ฟที่ /ui ---
ROOT = Path(__file__).resolve().parents[1]
UI_DIR = ROOT / "ui"
if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")

# --- Static files (เสิร์ฟไฟล์ในโฟลเดอร์ data ผ่าน /static) ---
DATA_DIR = ROOT / "data"
if DATA_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(DATA_DIR)), name="static")

@app.get("/")
def root():
    # เปิด / แล้วพาไปหน้า UI ถ้ามี ไม่งั้นไป /docs
    return RedirectResponse(url="/ui" if UI_DIR.exists() else "/docs")

@app.get("/favicon.ico")
def favicon():
    # กัน log 404 รก ๆ
    return Response(status_code=204)

@app.get("/health")
def health():
    return {"ok": True}

# === Scene-based (legacy) dataset upload ===
@app.post("/datasets/{scene_id}/upload")
async def upload(scene_id: str, files: List[UploadFile] = File(...)):
    count = save_uploads(scene_id, files)
    if count == 0:
        raise HTTPException(status_code=400, detail="No valid images uploaded")
    return {"saved": count, "scene_id": scene_id}

# === Scene-based training ===
@app.post("/train", response_model=TrainResponse)
async def start_train(req: TrainRequest):
    raw_dir = get_raw_dir(req.scene_id)
    if not raw_dir.exists():
        raise HTTPException(status_code=404, detail=f"No dataset for scene_id={req.scene_id}")
    job = JOBS.create()
    JOBS.run_in_thread(job, train_job, req.scene_id, raw_dir, req.img_size, req.epochs, req.lr)
    return TrainResponse(job_id=job.id)

# === Project-based endpoints ===
@app.get("/projects", response_model=List[ProjectInfo])
def list_projects_api():
    """ดึงรายการโปรเจกต์ทั้งหมด"""
    return [ProjectInfo(**p) for p in list_projects()]

@app.get("/projects/{project_id}", response_model=ProjectInfo)
def get_project_api(project_id: str):
    """รายละเอียดโปรเจกต์เดียว (สำหรับหน้า detail/เปิดงาน)"""
    try:
        return ProjectInfo(**get_project(project_id))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="project not found")

@app.post("/projects", response_model=ProjectInfo)
def create_project_api(req: ProjectCreate):
    """สร้างโปรเจกต์ใหม่"""
    meta = create_project(req.name, req.description)
    return ProjectInfo(**meta)

@app.post("/projects/{project_id}/upload")
async def upload_project(project_id: str, files: List[UploadFile] = File(...)):
    """อัปโหลดรูปเข้าโปรเจกต์"""
    count = save_project_uploads(project_id, files)
    if count == 0:
        raise HTTPException(status_code=400, detail="No valid images uploaded")
    return {"saved": count, "project_id": project_id}

@app.post("/projects/{project_id}/train", response_model=TrainResponse)
async def train_project(project_id: str, req: TrainProjectRequest):
    """เทรนโมเดลสำหรับโปรเจกต์"""
    raw_dir = get_project_raw_dir(project_id)
    if not raw_dir.exists():
        raise HTTPException(status_code=404, detail=f"No dataset for project_id={project_id}")
    job = JOBS.create()
    # ใช้ train_job_project เพื่อบันทึกโมเดลในโฟลเดอร์ project
    JOBS.run_in_thread(job, train_job_project, project_id, raw_dir, req.img_size, req.epochs, req.lr)
    return TrainResponse(job_id=job.id)

# === Job status and model endpoints (เหมือนเดิม) ===
@app.get("/jobs/{job_id}", response_model=JobStatus)
async def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(status=job.status, progress=job.progress, model_id=job.model_id, detail=job.detail)

# ดาวน์โหลดโมเดล (scene-mode)
@app.get("/models/{model_id}/download")
def download_model(model_id: str):
    path = MODELS_DIR / model_id / "model.pt"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    return FileResponse(str(path), media_type="application/octet-stream", filename="model.pt")

# ดาวน์โหลดโมเดล (project-mode)
@app.get("/projects/{project_id}/models/{model_id}/download")
def download_project_model(project_id: str, model_id: str):
    path = PROJECTS_DIR / project_id / "models" / model_id / "model.pt"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    return FileResponse(str(path), media_type="application/octet-stream", filename="model.pt")

# Optional stub for ROI (ไม่ถูกใช้ในเทรน MVP)
@app.post("/scenes/{scene_id}/roi")
async def save_roi(scene_id: str, payload: RoiPayload):
    out = DATASETS_DIR / scene_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "roi.json").write_text(
        json.dumps(payload.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"ok": True}

# ทดสอบโมเดล
@app.post("/models/{model_id}/test", response_model=TestResponse)
async def test_model(model_id: str, files: List[UploadFile] = File(...)):
    # โหลดไฟล์ทั้งหมดเข้าหน่วยความจำ
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

from .services.projects import create_project, list_projects, get_project, delete_project as _delete_project
@app.delete("/projects/{project_id}")
def delete_project_api(project_id: str, confirm: str = Query(..., description="Type exact project name to confirm")):
    try:
        info = get_project(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="project not found")
    if confirm != info["name"]:
        raise HTTPException(status_code=400, detail="Confirmation name mismatch")
    _delete_project(project_id)
    return {"ok": True}


from .services.storage import (
    save_uploads, get_raw_dir,
    save_project_uploads, get_project_raw_dir,
    list_project_images, list_scene_images,   # <- เพิ่ม
)

@app.get("/projects/{project_id}/images")
def list_project_images_api(project_id: str):
    return {"items": list_project_images(project_id)}

@app.get("/datasets/{scene_id}/images")
def list_scene_images_api(scene_id: str):
    return {"items": list_scene_images(scene_id)}

from .services.storage import (
    save_uploads, get_raw_dir,
    save_project_uploads, get_project_raw_dir,
    list_project_images, list_scene_images,
    delete_project_image, delete_scene_image,   # ← เพิ่ม
)

# ลบรูปในโปรเจกต์
@app.delete("/projects/{project_id}/images/{filename}")
def delete_project_image_api(project_id: str, filename: str):
    try:
        ok = delete_project_image(project_id, filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not ok:
        raise HTTPException(status_code=404, detail="file not found")
    return {"ok": True}

# ลบรูปใน scene (โหมดเดิม)
@app.delete("/datasets/{scene_id}/images/{filename}")
def delete_scene_image_api(scene_id: str, filename: str):
    try:
        ok = delete_scene_image(scene_id, filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not ok:
        raise HTTPException(status_code=404, detail="file not found")
    return {"ok": True}

@app.middleware("http")
async def no_cache_ui(request, call_next):
    resp = await call_next(request)
    if request.url.path.startswith("/ui/"):
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
    return resp

app.mount("/tmp", StaticFiles(directory=str(TMP_DIR), html=False), name="tmp")