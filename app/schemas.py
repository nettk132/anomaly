from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Tuple



class BaseSchema(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
class TrainRequest(BaseModel):
    scene_id: str = Field(..., description="Dataset/scene identifier")
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
    
class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None

class ProjectInfo(BaseModel):
    project_id: str
    name: str
    description: Optional[str] = None
    created_at: str
    num_images: int = 0
    last_model_id: Optional[str] = None

class TrainProjectRequest(BaseModel):
    project_id: str
    img_size: int = 256
    epochs: int = 30
    lr: float = 1e-4

class RoiPayload(BaseModel):
    image_size: Tuple[int, int]  # (H, W)
    polygon: List[Tuple[int, int]]


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
    