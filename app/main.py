from __future__ import annotations
import os, json
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from .schemas import (
    TrainRequest, TrainResponse, JobStatus, RoiPayload,
    TestResponse, TestItem,
)
from .config import DATASETS_DIR, MODELS_DIR
from .services.storage import save_uploads, get_raw_dir
from .services.jobs import JOBS
from .services.training import train_job
from .services.inference import test_images

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

@app.post("/datasets/{scene_id}/upload")
async def upload(scene_id: str, files: List[UploadFile] = File(...)):
    count = save_uploads(scene_id, files)
    if count == 0:
        raise HTTPException(status_code=400, detail="No valid images uploaded")
    return {"saved": count, "scene_id": scene_id}

@app.post("/train", response_model=TrainResponse)
async def start_train(req: TrainRequest):
    raw_dir = get_raw_dir(req.scene_id)
    if not raw_dir.exists():
        raise HTTPException(status_code=404, detail=f"No dataset for scene_id={req.scene_id}")
    job = JOBS.create()
    JOBS.run_in_thread(job, train_job, req.scene_id, raw_dir, req.img_size, req.epochs, req.lr)
    return TrainResponse(job_id=job.id)

@app.get("/jobs/{job_id}", response_model=JobStatus)
async def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(status=job.status, progress=job.progress, model_id=job.model_id, detail=job.detail)

@app.get("/models/{model_id}/download")
def download_model(model_id: str):
    path = MODELS_DIR / model_id / "model.pt"
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
