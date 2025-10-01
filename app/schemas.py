from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Tuple, Literal

class BaseSchema(BaseModel):
    # กัน pydantic v2 เตือนเวลาใช้ชื่อ field ที่ชนกับ reserved prefix
    model_config = ConfigDict(protected_namespaces=())

# ---------- Training (scene mode) ----------
class TrainRequest(BaseSchema):
    scene_id: str = Field(..., min_length=1, description="Dataset/scene identifier")
    img_size: int = 256
    epochs: int = 30
    lr: float = 1e-4

class TrainResponse(BaseSchema):
    job_id: str

class JobStatus(BaseSchema):
    status: str
    progress: float = 0.0
    model_id: Optional[str] = None
    detail: Optional[str] = None

# ---------- Projects ----------
class ProjectCreate(BaseSchema):
    name: str = Field(..., min_length=1)
    description: Optional[str] = None
    training_mode: Literal["anomaly", "finetune"] = Field("anomaly")

class ProjectInfo(BaseSchema):
    project_id: str
    name: str
    description: Optional[str] = None
    created_at: str
    num_images: int = 0
    last_model_id: Optional[str] = None
    training_mode: Literal["anomaly", "finetune"] = "anomaly"

class TrainProjectRequest(BaseSchema):
    # ไม่ต้องมี project_id ตรงนี้ เพราะส่งใน path: /projects/{project_id}/train
    img_size: int = 256
    epochs: int = 30
    lr: float = 1e-4
    base_model_id: Optional[str] = None

class ModelInfo(BaseSchema):
    model_id: str
    mode: str
    scene_id: Optional[str] = None
    project_id: Optional[str] = None
    created_at: Optional[str] = None
    img_size: Optional[int] = None
    epochs_run: Optional[int] = None
    lr: Optional[float] = None
    threshold: Optional[float] = None
    note: Optional[str] = None
    base_model_id: Optional[str] = None
    training_mode: Optional[str] = None
    path: str

# ---------- ROI ----------
class RoiPayload(BaseSchema):
    image_size: Tuple[int, int]  # (H, W)
    polygon: List[Tuple[int, int]]

# ---------- Inference ----------
class TestItem(BaseSchema):
    filename: str
    score: Optional[float] = None
    thr: Optional[float] = None
    is_anomaly: Optional[bool] = None
    image_url: Optional[str] = None
    heatmap_url: Optional[str] = None
    overlay_url: Optional[str] = None
    bboxes: Optional[List[Tuple[int,int,int,int]]] = None  # [(x,y,w,h), ...]
    error: Optional[str] = None

class TestResponse(BaseSchema):
    model_id: str
    items: List[TestItem]
