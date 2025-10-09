from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Tuple, Literal

from .utils import MAX_SLUG_LENGTH, SLUG_PATTERN

class BaseSchema(BaseModel):
    # กัน pydantic v2 เตือนเวลาใช้ชื่อ field ที่ชนกับ reserved prefix
    model_config = ConfigDict(protected_namespaces=())

# ---------- Training (scene mode) ----------
class TrainRequest(BaseSchema):
    scene_id: str = Field(
        ...,
        min_length=1,
        max_length=MAX_SLUG_LENGTH,
        pattern=SLUG_PATTERN,
        description="Dataset/scene identifier",
    )
    img_size: int = 256
    epochs: int = 30
    lr: float = 1e-4

class TrainResponse(BaseSchema):
    job_id: str

class JobStatus(BaseSchema):
    status: str
    progress: float = 0.0
    model_id: Optional[str] = Field(
        default=None, min_length=1, max_length=MAX_SLUG_LENGTH, pattern=SLUG_PATTERN
    )
    detail: Optional[str] = None

# ---------- Projects ----------
class ProjectCreate(BaseSchema):
    name: str = Field(..., min_length=1)
    description: Optional[str] = None
    training_mode: Literal["anomaly", "finetune", "yolo"] = Field("anomaly")

class ProjectInfo(BaseSchema):
    project_id: str = Field(..., min_length=1, max_length=MAX_SLUG_LENGTH, pattern=SLUG_PATTERN)
    name: str
    description: Optional[str] = None
    created_at: str
    num_images: int = 0
    last_model_id: Optional[str] = Field(
        default=None, min_length=1, max_length=MAX_SLUG_LENGTH, pattern=SLUG_PATTERN
    )
    training_mode: Literal["anomaly", "finetune", "yolo"] = "anomaly"

class TrainProjectRequest(BaseSchema):
    # ไม่ต้องมี project_id ตรงนี้ เพราะส่งใน path: /projects/{project_id}/train
    img_size: int = 256
    epochs: int = 30
    lr: float = 1e-4
    base_model_id: Optional[str] = Field(
        default=None, min_length=1, max_length=MAX_SLUG_LENGTH, pattern=SLUG_PATTERN
    )

class YoloTrainRequest(BaseSchema):
    epochs: int = 100
    img_size: int = 640
    batch: Optional[int] = None
    model_variant: str = Field("yolov8n.pt", description="Base YOLO checkpoint to start from")
    lr0: Optional[float] = Field(None, description="Override initial learning rate")
    conf: float = Field(0.25, ge=0.0, le=1.0, description="Confidence threshold for validation/inference previews")
    iou: float = Field(0.45, ge=0.0, le=1.0, description="IoU threshold for NMS during validation")

class YoloDatasetInfo(BaseSchema):
    ready: bool = False
    train_images: int = 0
    val_images: int = 0
    classes: List[str] = Field(default_factory=list)
    yaml_path: Optional[str] = None
    note: Optional[str] = None
    updated_at: Optional[str] = None

class ModelInfo(BaseSchema):
    model_id: str = Field(..., min_length=1, max_length=MAX_SLUG_LENGTH, pattern=SLUG_PATTERN)
    mode: str
    scene_id: Optional[str] = Field(
        default=None, min_length=1, max_length=MAX_SLUG_LENGTH, pattern=SLUG_PATTERN
    )
    project_id: Optional[str] = Field(
        default=None, min_length=1, max_length=MAX_SLUG_LENGTH, pattern=SLUG_PATTERN
    )
    created_at: Optional[str] = None
    img_size: Optional[int] = None
    epochs_run: Optional[int] = None
    lr: Optional[float] = None
    threshold: Optional[float] = None
    note: Optional[str] = None
    base_model_id: Optional[str] = Field(
        default=None, min_length=1, max_length=MAX_SLUG_LENGTH, pattern=SLUG_PATTERN
    )
    training_mode: Optional[str] = None
    path: str

# ---------- Labeling ----------
class LabelBox(BaseSchema):
    id: Optional[str] = None
    label: str = Field(..., min_length=1)
    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)
    width: float = Field(..., gt=0.0, le=1.0)
    height: float = Field(..., gt=0.0, le=1.0)


class ImageLabel(BaseSchema):
    filename: str
    image_width: int = Field(..., gt=0)
    image_height: int = Field(..., gt=0)
    boxes: List[LabelBox] = Field(default_factory=list)
    updated_at: str


class LabelListResponse(BaseSchema):
    items: List[ImageLabel] = Field(default_factory=list)
    classes: List[str] = Field(default_factory=list)


class SaveLabelsRequest(BaseSchema):
    image_width: int = Field(..., gt=0)
    image_height: int = Field(..., gt=0)
    boxes: List[LabelBox] = Field(default_factory=list)


class LabelClassesRequest(BaseSchema):
    classes: List[str] = Field(default_factory=list)


class LabelClassesResponse(BaseSchema):
    classes: List[str] = Field(default_factory=list)

# ---------- Inference ----------
class DetectionBox(BaseSchema):
    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    box_format: Literal["xyxy", "xywh"] = "xyxy"

class TestItem(BaseSchema):
    filename: str
    score: Optional[float] = None
    thr: Optional[float] = None
    is_anomaly: Optional[bool] = None
    image_url: Optional[str] = None
    heatmap_url: Optional[str] = None
    overlay_url: Optional[str] = None
    bboxes: Optional[List[Tuple[int,int,int,int]]] = None  # [(x,y,w,h), ...]
    detections: Optional[List[DetectionBox]] = None
    error: Optional[str] = None

class TestResponse(BaseSchema):
    model_id: str
    items: List[TestItem]

