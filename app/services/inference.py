from __future__ import annotations
import io, json, yaml
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2
from PIL import Image
import torch
from uuid import uuid4
import torch.nn as nn
import torch.nn.functional as F

from safetensors.torch import load_file as load_safetensors

from ..models.autoencoder import ConvAE, build_autoencoder
from .scoring import compute_anomaly_score
from ..config import DATASETS_DIR, MODELS_DIR, PROJECTS_DIR
from ..utils import PathTraversalError, safe_join, validate_slug
from .storage import get_raw_dir, get_project_raw_dir
from .yolo_training import MissingUltralyticsError

# ----- utils -----
def _robust_threshold(errs: np.ndarray) -> float:
    if errs.size == 0:
        return 0.1
    p99  = np.percentile(errs, 99)
    p995 = np.percentile(errs, 99.5) if errs.size >= 50 else p99
    med  = np.median(errs)
    mad  = np.median(np.abs(errs - med)) + 1e-12
    robust = float(med + 3.0 * 1.4826 * mad)
    return float(max(p995, robust))

def _to_tensor_rgb01(img_bgr: np.ndarray, size: int) -> torch.Tensor:
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(img).float().permute(2,0,1) / 255.0
    return x.unsqueeze(0)  # (1,3,H,W)

class _AvgPoolSSIM(nn.Module):
    """SSIM เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเนยเธเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€ เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขย AvgPool เน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเธขย inference เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเนยเธเน€เธโฌเน€เธยเนโฌโ€เน€เธโฌเน€เธยเน€เธโ€ขเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเนโฌเธเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเนยเธเน€เธโฌเน€เธยเนโฌโ€เน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขย"""
    def __init__(self, ksize: int = 7):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=ksize, stride=1, padding=ksize // 2)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mu_x = self.pool(x)
        mu_y = self.pool(y)
        sigma_x = self.pool(x * x) - mu_x * mu_x
        sigma_y = self.pool(y * y) - mu_y * mu_y
        sigma_xy = self.pool(x * y) - mu_x * mu_y
        num = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        den = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        ssim_map = num / (den + 1e-12)
        return ssim_map.flatten(1).mean(1)  # (N,)

def _reconstruct_tta_lr(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Test-time augmentation เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเนยเธเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€: เน€เธโฌเน€เธยเนยเธเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€ฆเน€เธโฌเน€เธยเน€เธโ€ขเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€ฆเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเนยเธเน€เธโฌเน€เธยเนโฌยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€ฆเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€ฆเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเน€เธย-เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธโ€
    เน€เธโฌเน€เธยเน€เธโ€ฆเน€เธโฌเน€เธยเนโฌย noise เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธโ€ขเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเนโฌเธเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเนโฌเธเน€เธโฌเน€เธยเธขย เน€เธยเธขยเนโฌย heatmap เน€เธโฌเน€เธยเนยเธเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€ขเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€“เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเนยเธเน€เธโฌเน€เธยเน€เธโ€ฆเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธย
    """
    xhat1 = model(x)
    x_flip = torch.flip(x, dims=[3])
    xhat2 = torch.flip(model(x_flip), dims=[3])
    return 0.5 * (xhat1 + xhat2)

def _heatmap_from_abs(abs_diff: torch.Tensor) -> np.ndarray:
    """
    abs_diff: (1,3,H,W) เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€ |xhat - x|
    return: heatmap float32 [0..1] (H,W)
    """
    hm = abs_diff.mean(1).squeeze(0)
    hm = hm / (hm.max() + 1e-12)
    return hm.detach().cpu().numpy().astype(np.float32)

def _save_jpeg(path: Path, array: np.ndarray, *, color: str = "bgr", quality: int = 95) -> None:
    """Save numpy array as JPEG with proper RGB ordering for the UI."""
    arr = array
    if arr.ndim == 3:
        if color == "bgr":
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        elif color == "rgb":
            pass
        else:
            raise ValueError(f"unsupported color mode: {color}")
    img = Image.fromarray(arr)
    img.save(path, format="JPEG", quality=quality)

def _colorize_overlay(img_bgr: np.ndarray, heatmap01: np.ndarray, alpha: float = 0.45, min_level: float = 0.2) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    hm = cv2.resize(heatmap01.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    hm = np.clip(hm, 0.0, 1.0)
    if min_level > 0.0:
        scale = 1.0 / max(1e-6, 1.0 - min_level)
        hm = np.where(hm >= min_level, (hm - min_level) * scale, 0.0)
    hm_color = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET).astype(np.float32)
    base = img_bgr.astype(np.float32)
    weight = (hm[..., None] * alpha).astype(np.float32)
    overlay = base * (1.0 - weight) + hm_color * weight
    return np.clip(overlay, 0.0, 255.0).astype(np.uint8)

def _extract_bboxes_from_heatmap(heatmap01: np.ndarray, out_hw: Tuple[int,int],
                                 min_area_ratio: float = 0.001,
                                 min_box_size: int = 6,
                                 max_boxes: int = 5) -> List[Tuple[int,int,int,int]]:
    """
    เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเธขย heatmap [0..1] (Hh,Wh) เน€เธยเธขยเนโฌย เน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธโ€”เน€เธโฌเน€เธยเนโฌยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเนยเธเน€เธโฌเน€เธยเนโฌโ€เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเธขย เน€เธยเธขยเนโฌย เน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเธขย binary mask เน€เธโฌเน€เธยเนโฌยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธย Otsu
    เน€เธยเธขยเนโฌย เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขย noise เน€เธโฌเน€เธยเนโฌยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธย morphology เน€เธยเธขยเนโฌย เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€”เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€ฆเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเนโฌเธเน€เธโฌเน€เธยเธขย bounding boxes (x,y,w,h)
    """
    H, W = out_hw
    hm = cv2.resize(heatmap01.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    # เน€เธโฌเน€เธยเนยเธเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€ฆเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเนยเธเน€เธโฌเน€เธยเน€เธโ€ฆเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขย smooth
    hm = cv2.GaussianBlur(hm, (5,5), 0)
    hm_u8 = np.clip(hm * 255.0, 0, 255).astype(np.uint8)
    # Otsu threshold
    _, bin_ = cv2.threshold(hm_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # morphology: open เน€เธโฌเน€เธยเนยเธเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€”เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธโ€ฆเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเนโฌยเน€เธโฌเน€เธยเนยเธเน€เธโฌเน€เธยเน€เธโ€ฆเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขย เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€ฆเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธย dilate เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเนยเธเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเนโฌยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€“เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขย
    kernel = np.ones((3,3), np.uint8)
    bin_ = cv2.morphologyEx(bin_, cv2.MORPH_OPEN, kernel, iterations=1)
    bin_ = cv2.dilate(bin_, kernel, iterations=1)

    cnts, _ = cv2.findContours(bin_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_min = max(1, int(min_area_ratio * (H * W)))

    boxes: List[Tuple[int,int,int,int]] = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w < min_box_size or h < min_box_size:
            continue
        if w*h < area_min:
            continue
        boxes.append((int(x), int(y), int(w), int(h)))

    # เน€เธโฌเน€เธยเนยเธเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธโ€ขเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเนยเธเน€เธโฌเน€เธยเน€เธโ€ฆเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขย เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€ฆเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเนโฌเธเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเนโฌยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเนยเธเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเธขย max_boxes
    boxes.sort(key=lambda b: b[2]*b[3], reverse=True)
    return boxes[:max_boxes]

# ----- core loading -----
def _resolve_model_dir(model_id: str) -> Path:
    validate_slug(model_id, name="model_id")
    try:
        return safe_join(MODELS_DIR, model_id, must_exist=True)
    except (FileNotFoundError, PathTraversalError):
        pass
    if PROJECTS_DIR.exists():
        for proj_dir in PROJECTS_DIR.iterdir():
            if not proj_dir.is_dir():
                continue
            models_dir = proj_dir / 'models'
            if models_dir.is_dir():
                try:
                    return safe_join(models_dir, model_id, must_exist=True)
                except (FileNotFoundError, PathTraversalError):
                    pass
            imports_dir = proj_dir / 'imports'
            if imports_dir.is_dir():
                try:
                    return safe_join(imports_dir, model_id, must_exist=True)
                except (FileNotFoundError, PathTraversalError):
                    pass
    raise FileNotFoundError(f"Model dir not found: {model_id}")


def _load_model_config(model_dir: Path) -> dict:
    cfg_path = model_dir / 'config.yaml'
    if not cfg_path.exists():
        return {}
    with cfg_path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    return data


def _load_autoencoder_from_dir(model_dir: Path, cfg: dict, device: str):
    img_size = int(cfg.get('img_size', 256))
    pt_path = model_dir / 'model.pt'
    st_path = model_dir / 'model.safetensors'
    if pt_path.exists():
        state = torch.load(pt_path, map_location=device)
    elif st_path.exists():
        state = load_safetensors(st_path, device=device)
    else:
        raise FileNotFoundError(f"No model weights found under {model_dir}")
    model = build_autoencoder(state_dict=state).to(device).eval()

    thr = 0.0
    th_path = model_dir / 'threshold.json'
    if th_path.exists():
        try:
            with th_path.open('r', encoding='utf-8') as f:
                thj = json.load(f)
        except Exception:
            thj = {}
        if isinstance(thj, dict):
            if 'threshold_score' in thj:
                try:
                    thr = float(thj.get('threshold_score', 0.0))
                except Exception:
                    thr = 0.0
            elif 'threshold_mse' in thj:
                try:
                    thr = float(thj.get('threshold_mse', 0.0))
                except Exception:
                    thr = 0.0

    raw_dir = None
    if cfg.get('scene_id'):
        try:
            raw_dir = get_raw_dir(cfg['scene_id'])
        except Exception:
            raw_dir = None
    elif cfg.get('project_id'):
        try:
            raw_dir = get_project_raw_dir(cfg['project_id'])
        except Exception:
            raw_dir = None

    return model, raw_dir, img_size, thr


def _load_model_and_cfg(model_id: str, device: str):
    """Backward-compatible helper used by legacy inference paths.
    Returns: (model, raw_dir_or_None, img_size, threshold)
    """
    model_dir = _resolve_model_dir(model_id)
    cfg = _load_model_config(model_dir)
    return _load_autoencoder_from_dir(model_dir, cfg, device)


def _fallback_calibrate_threshold(raw_dir: Path, model: torch.nn.Module, img_size: int, device: str,
                                  limit: int = 64) -> float:
    """เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเนโฌยเน€เธโฌเน€เธยเน€เธโ€ข threshold เน€เธโฌเน€เธยเนยเธเน€เธโฌเน€เธยเนโฌยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเนยเธเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขย/เน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขย: เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเนยเธเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขย normal เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขย raw_dir เน€เธโฌเน€เธยเนโฌโ€เน€เธโฌเน€เธยเน€เธโ€ขเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธโ€"""
    if not raw_dir.exists():
        return 0.1
    # เน€เธโฌเน€เธยเนยเธเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเนโฌยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเนโฌย limit
    img_paths = []
    for p in raw_dir.glob("*"):
        if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".webp"]:
            img_paths.append(p)
            if len(img_paths) >= limit:
                break
    if not img_paths:
        return 0.1

    errs = []
    ssim_calc = _AvgPoolSSIM(ksize=7).to(device)
    with torch.no_grad():
        for p in img_paths:
            bgr = cv2.imread(str(p))
            if bgr is None: 
                continue
            x = _to_tensor_rgb01(bgr, img_size).to(device)
            # เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขย TTA เน€เธโฌเน€เธยเนยเธเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€ เน€เธโฌเน€เธยเธขย เน€เธโฌเน€เธยเนยเธเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€”เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธโ€ขเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเนโฌเธเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเนโฌเธเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเนยเธเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเนโฌโ€เน€เธโฌเน€เธยเน€เธโ€ขเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€“เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขย
            xhat = _reconstruct_tta_lr(model, x)
            # เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเนยเธเน€เธโฌเน€เธยเนโฌยเน€เธโฌเน€เธยเน€เธโ€ขเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเนโฌเธเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเนยเธเน€เธโฌเน€เธยเนโฌโ€เน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขย: mean/top-k diff + SSIM
            abs_diff = torch.abs(xhat - x)
            ssim_val = ssim_calc(xhat.clamp(0,1), x.clamp(0,1))
            score = compute_anomaly_score(abs_diff, ssim_val)
            errs.extend(score.detach().cpu().tolist())
    errs = np.asarray(errs, dtype=np.float32)
    return _robust_threshold(errs)

# ----- public API used by main.py -----
def test_images(model_id: str, models_root: Path, buf_list: List[Tuple[str, bytes]]):
    validate_slug(model_id, name="model_id")
    """
    เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€”เน€เธโฌเน€เธยเธขย list เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขย dict:
    {
      filename, score, thr, is_anomaly,
      image_url, heatmap_url, overlay_url,
      result_image, threshold  # alias เน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเธขย UI เน€เธโฌเน€เธยเนยเธเน€เธโฌเน€เธยเนโฌยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเน€เธย
    }
    URLs เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€ขเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเนโฌโ€เน€เธโฌเน€เธยเน€เธโ€ขเน€เธโฌเน€เธยเธขย /static/preview/<model_id>/...
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = _resolve_model_dir(model_id)
    cfg = _load_model_config(model_dir)
    training_mode = str(cfg.get("training_mode") or "anomaly").lower()
    if training_mode == "yolo":
        return _test_images_yolo(model_id, model_dir, cfg, buf_list, device)
    model, raw_dir, img_size, thr = _load_autoencoder_from_dir(model_dir, cfg, device)

    # Fallback calibrate เน€เธโฌเน€เธยเนโฌโ€เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€ threshold เน€เธโฌเน€เธยเนยเธเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขย 0/เน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเน€เธย
    if thr <= 1e-12 and raw_dir is not None:
        thr = _fallback_calibrate_threshold(raw_dir, model, img_size, device)

    # ---- เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€ฆเน€เธโฌเน€เธยเนยเธเน€เธโฌเน€เธยเนโฌยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธโ€ขเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเน€เธย (เน€เธโฌเน€เธยเนยเธเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเธขย /static) ----
    # เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขย preview เน€เธโฌเน€เธยเนโฌโ€เน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€ฆเน€เธโฌเน€เธยเนยเธเน€เธโฌเน€เธยเนโฌยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€ฆเน€เธโฌเน€เธยเน€เธย URL เน€เธโฌเน€เธยเนยเธเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€”เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเนโฌเธเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเธขย
    DATA_DIR   = DATASETS_DIR.parent  # data/
    try:
        preview_dir = safe_join(DATA_DIR, "preview", model_id, must_exist=False)
    except PathTraversalError as exc:
        raise ValueError("invalid model_id") from exc
    preview_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"/static/preview/{model_id}"

    items = []
    ssim_calc = _AvgPoolSSIM(ksize=7).to(device)
    with torch.no_grad():
        for fname, content in buf_list:
            arr = np.frombuffer(content, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                items.append({
                    "filename": fname, "score": None, "thr": float(thr),
                    "is_anomaly": None,
                    "image_url": None, "heatmap_url": None, "overlay_url": None,
                    "result_image": None,    # alias
                    "threshold": float(thr), # alias
                    "error": "cannot decode image"
                })
                continue

            x = _to_tensor_rgb01(bgr, img_size).to(device)
            xhat = _reconstruct_tta_lr(model, x)
            abs_diff = torch.abs(xhat - x)

            # เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเนโฌย score เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเนโฌยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€ฆเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเนโฌเธเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเนยเธเน€เธโฌเน€เธยเนโฌโ€เน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขย
            ssim_val = ssim_calc(xhat.clamp(0,1), x.clamp(0,1))
            score = float(compute_anomaly_score(abs_diff, ssim_val).item())

            # heatmap + overlay
            hm      = _heatmap_from_abs(abs_diff)
            overlay = _colorize_overlay(bgr, hm, alpha=0.45)

            # เน€เธโฌเน€เธยเนโฌโ€เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเนยเธเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเธขย threshold เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธโ€ bounding boxes เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเธขย heatmap เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€ฆเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเนโฌยเน€เธโฌเน€เธยเนโฌโ€เน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขย overlay
            bboxes: List[Tuple[int,int,int,int]] = []
            if score > thr:
                H, W = bgr.shape[:2]
                bboxes = _extract_bboxes_from_heatmap(hm, (H, W), min_area_ratio=0.001, min_box_size=6, max_boxes=5)
                for (x, y, w, h) in bboxes:
                    cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 0, 255), 2)
                if bboxes:
                    # เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเนยเธเน€เธโฌเน€เธยเน€เธโ€ฆเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขย เน€เธโฌเน€เธยเธขย เน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€ฆเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเธขย
                    bx, by, bw, bh = bboxes[0]
                    cv2.putText(overlay, 'ANOMALY', (bx, max(0, by-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

            # เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเนโฌโ€เน€เธโฌเน€เธยเน€เธโ€“เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธโ€ฆเน€เธโฌเน€เธยเธขย
            stem    = Path(fname).stem
            out_img = preview_dir / f"{stem}_input.jpg"
            out_hm  = preview_dir / f"{stem}_heatmap.jpg"
            out_ovr = preview_dir / f"{stem}_overlay.jpg"
            _save_jpeg(out_img, bgr)
            _save_jpeg(out_hm, (hm * 255).astype(np.uint8))
            _save_jpeg(out_ovr, overlay)

            version = f"?v={uuid4().hex}"
            url_img = f"{base_url}/{stem}_input.jpg{version}"
            url_hm  = f"{base_url}/{stem}_heatmap.jpg{version}"
            url_ovr = f"{base_url}/{stem}_overlay.jpg{version}"

            items.append({
                "filename": fname,
                "score": round(score, 6),
                "thr": float(thr),
                "is_anomaly": bool(score > thr),
                "image_url":   url_img,
                "heatmap_url": url_hm,
                "overlay_url": url_ovr,
                "bboxes": bboxes,
                # aliases เน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเธขย UI เน€เธโฌเน€เธยเนยเธเน€เธโฌเน€เธยเนโฌยเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเน€เธย:
                "result_image": url_ovr,      # เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขย overlay เน€เธโฌเน€เธยเนยเธเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเธขย เน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเธขยเน€เธโฌเน€เธยเน€เธยเน€เธโฌเน€เธยเน€เธโ€ฆเน€เธโฌเน€เธยเน€เธโ€เน€เธโฌเน€เธยเธขย
                "threshold": float(thr),
            })

    return items



def _test_images_yolo(model_id: str, model_dir: Path, cfg: dict, buf_list: List[Tuple[str, bytes]], device: str):
    validate_slug(model_id, name="model_id")
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise MissingUltralyticsError("ultralytics package is required for YOLO inference. Install with 'pip install ultralytics'.") from exc

    weights_path = model_dir / "model.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"model.pt not found under {model_dir}")

    model = YOLO(str(weights_path))
    yolo_cfg = cfg.get("yolo") or {}
    try:
        conf = float(yolo_cfg.get("conf_threshold", 0.25))
    except Exception:
        conf = 0.25
    try:
        iou = float(yolo_cfg.get("iou_threshold", 0.45))
    except Exception:
        iou = 0.45

    classes = yolo_cfg.get("classes") or yolo_cfg.get("class_names") or []
    if isinstance(classes, dict):
        classes = [str(classes[k]) for k in sorted(classes.keys())]
    elif not isinstance(classes, list):
        classes = []
    classes = [str(c) for c in classes]
    if not classes:
        names = getattr(model, "names", None)
        if isinstance(names, dict):
            classes = [str(names[k]) for k in sorted(names.keys())]
        elif isinstance(names, list):
            classes = [str(n) for n in names]

    data_dir = DATASETS_DIR.parent
    try:
        preview_dir = safe_join(data_dir, "preview", model_id, must_exist=False)
    except PathTraversalError as exc:
        raise ValueError("invalid model_id") from exc
    preview_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"/static/preview/{model_id}"

    items = []
    for fname, content in buf_list:
        arr = np.frombuffer(content, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            items.append({
                "filename": fname,
                "score": None,
                "thr": float(conf),
                "is_anomaly": None,
                "image_url": None,
                "heatmap_url": None,
                "overlay_url": None,
                "result_image": None,
                "threshold": float(conf),
                "bboxes": None,
                "detections": None,
                "error": "cannot decode image",
            })
            continue

        try:
            results = model.predict(bgr, conf=conf, iou=iou, device=device, verbose=False)
        except Exception as exc:  # pragma: no cover
            items.append({
                "filename": fname,
                "score": None,
                "thr": float(conf),
                "is_anomaly": None,
                "image_url": None,
                "heatmap_url": None,
                "overlay_url": None,
                "result_image": None,
                "threshold": float(conf),
                "bboxes": None,
                "detections": None,
                "error": f"inference failed: {exc}",
            })
            continue

        res = results[0] if isinstance(results, (list, tuple)) else results
        boxes = getattr(res, "boxes", None)
        detections = []
        int_bboxes = []
        max_conf = 0.0
        overlay = bgr.copy()
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
            confs = boxes.conf.cpu().numpy() if getattr(boxes, "conf", None) is not None else np.zeros(len(xyxy), dtype=np.float32)
            clses = boxes.cls.cpu().numpy() if getattr(boxes, "cls", None) is not None else np.zeros(len(xyxy), dtype=np.float32)
            for idx in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[idx]
                conf_val = float(confs[idx]) if idx < len(confs) else 0.0
                max_conf = max(max_conf, conf_val)
                cls_idx = int(clses[idx]) if idx < len(clses) else 0
                label = classes[cls_idx] if 0 <= cls_idx < len(classes) else str(cls_idx)
                detections.append({
                    "label": label,
                    "confidence": conf_val,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "box_format": "xyxy",
                })
                box_w = int(max(1.0, x2 - x1))
                box_h = int(max(1.0, y2 - y1))
                int_box = (int(x1), int(y1), box_w, box_h)
                int_bboxes.append(int_box)
                color = (0, 255, 0)
                cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(overlay, f"{label} {conf_val:.2f}", (int(x1), max(0, int(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        else:
            max_conf = 0.0

        stem = Path(fname).stem
        input_path = preview_dir / f"{stem}_input.jpg"
        overlay_path = preview_dir / f"{stem}_overlay.jpg"
        _save_jpeg(input_path, bgr)
        _save_jpeg(overlay_path, overlay)

        version = f"?v={uuid4().hex}"
        img_url = f"{base_url}/{stem}_input.jpg{version}"
        overlay_url = f"{base_url}/{stem}_overlay.jpg{version}"

        items.append({
            "filename": fname,
            "score": float(max_conf),
            "thr": float(conf),
            "is_anomaly": None,
            "image_url": img_url,
            "heatmap_url": None,
            "overlay_url": overlay_url,
            "result_image": overlay_url,
            "threshold": float(conf),
            "bboxes": int_bboxes or None,
            "detections": detections or None,
            "error": None,
        })

    return items





