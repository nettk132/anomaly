from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Tuple

class BaseSchema(BaseModel):
    # กัน pydantic v2 เตือนเวลาใช้ชื่อ field ที่ชนกับ reserved prefix
    model_config = ConfigDict(protected_namespaces=())

# ---------- Training (scene mode) ----------
class TrainRequest(BaseModel):
    scene_id: str = Field(..., min_length=1, description="Dataset/scene identifier")
    img_size: int = 256
    epochs: int = 30
    lr: float = 1e-4

class TrainResponse(BaseModel):
    job_id: str

class JobStatus(BaseModel):
    status: str
    progress: float = 0.0
    model_id: Optional[str] = None
    detail: Optional[str] = None

# ---------- Projects ----------
class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1)
    description: Optional[str] = None

class ProjectInfo(BaseModel):
    project_id: str
    name: str
    description: Optional[str] = None
    created_at: str
    num_images: int = 0
    last_model_id: Optional[str] = None

class TrainProjectRequest(BaseModel):
    # ไม่ต้องมี project_id ตรงนี้ เพราะส่งใน path: /projects/{project_id}/train
    img_size: int = 256
    epochs: int = 30
    lr: float = 1e-4

# ---------- ROI ----------
class RoiPayload(BaseModel):
    image_size: Tuple[int, int]  # (H, W)
    polygon: List[Tuple[int, int]]

# ---------- Inference ----------
class TestItem(BaseModel):
    filename: str
    score: Optional[float] = None
    thr: Optional[float] = None
    is_anomaly: Optional[bool] = None
    image_url: Optional[str] = None
    heatmap_url: Optional[str] = None
    overlay_url: Optional[str] = None
    error: Optional[str] = None

class TestResponse(BaseModel):
    model_id: str
    items: List[TestItem]
    
