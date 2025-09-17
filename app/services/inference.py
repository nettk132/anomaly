from __future__ import annotations
import io, json, yaml
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.autoencoder import ConvAE, build_autoencoder
from .scoring import compute_anomaly_score
from ..config import DATASETS_DIR, MODELS_DIR, PROJECTS_DIR
from .storage import get_raw_dir, get_project_raw_dir

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
    """SSIM แบบเบา ใช้ AvgPool สำหรับ inference ให้เทียบกับตอนเทรน"""
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
    Test-time augmentation แบบเบา: เฉลี่ยผลจากรูปเดิมและกลับซ้าย-ขวา
    ลด noise ของการรีคอนสตรัคต์ → heatmap เนียนขึ้นเล็กน้อย
    """
    xhat1 = model(x)
    x_flip = torch.flip(x, dims=[3])
    xhat2 = torch.flip(model(x_flip), dims=[3])
    return 0.5 * (xhat1 + xhat2)

def _heatmap_from_abs(abs_diff: torch.Tensor) -> np.ndarray:
    """
    abs_diff: (1,3,H,W) ค่า |xhat - x|
    return: heatmap float32 [0..1] (H,W)
    """
    hm = abs_diff.mean(1).squeeze(0)
    hm = hm / (hm.max() + 1e-12)
    return hm.detach().cpu().numpy().astype(np.float32)

def _colorize_overlay(img_bgr: np.ndarray, heatmap01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    hm = cv2.resize(heatmap01, (w, h), interpolation=cv2.INTER_CUBIC)
    hm_color = cv2.applyColorMap((hm*255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 1.0, hm_color, alpha, 0.0)
    return overlay

def _extract_bboxes_from_heatmap(heatmap01: np.ndarray, out_hw: Tuple[int,int],
                                 min_area_ratio: float = 0.001,
                                 min_box_size: int = 6,
                                 max_boxes: int = 5) -> List[Tuple[int,int,int,int]]:
    """
    จาก heatmap [0..1] (Hh,Wh) → ยืดให้เท่ารูปจริง → สร้าง binary mask ด้วย Otsu
    → กรอง noise ด้วย morphology → คืนลิสต์ bounding boxes (x,y,w,h)
    """
    H, W = out_hw
    hm = cv2.resize(heatmap01.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    # เบลอเล็กน้อยให้ smooth
    hm = cv2.GaussianBlur(hm, (5,5), 0)
    hm_u8 = np.clip(hm * 255.0, 0, 255).astype(np.uint8)
    # Otsu threshold
    _, bin_ = cv2.threshold(hm_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # morphology: open เพื่อลบจุดเล็ก แล้ว dilate ให้เป็นก้อนชัดขึ้น
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

    # เรียงจากใหญ่ไปเล็ก และตัดไม่เกิน max_boxes
    boxes.sort(key=lambda b: b[2]*b[3], reverse=True)
    return boxes[:max_boxes]

# ----- core loading -----
def _load_model_and_cfg(model_id: str, device: str):
    """โหลดโมเดลจาก scene-mode หรือ project-mode ก็ได้ และคืน raw_dir สำหรับ calibrate

    return: (model, raw_dir_or_None, img_size, thr)
    """
    # 1) scene-mode
    model_dir = MODELS_DIR / model_id
    if not model_dir.exists():
        # 2) project-mode: data/projects/*/models/<model_id>
        for proj_dir in PROJECTS_DIR.iterdir() if PROJECTS_DIR.exists() else []:
            cand = proj_dir / 'models' / model_id
            if cand.exists():
                model_dir = cand
                break
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    # config.yaml
    with open(model_dir / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    img_size = int(cfg.get("img_size", 256))

    # model.pt
    state = torch.load(model_dir / "model.pt", map_location=device)
    model = build_autoencoder(state_dict=state).to(device).eval()

    # threshold.json
    thr = 0.0
    th_path = model_dir / "threshold.json"
    if th_path.exists():
        try:
            with open(th_path, "r", encoding="utf-8") as f:
                thj = json.load(f)
            thr = float(thj.get("threshold_mse", thj.get("threshold_score", 0.0)))
        except Exception:
            thr = 0.0

    # resolve raw_dir for fallback calibrate
    raw_dir = None
    if 'scene_id' in cfg and cfg['scene_id']:
        try:
            raw_dir = get_raw_dir(cfg['scene_id'])
        except Exception:
            raw_dir = None
    elif 'project_id' in cfg and cfg['project_id']:
        try:
            raw_dir = get_project_raw_dir(cfg['project_id'])
        except Exception:
            raw_dir = None

    return model, raw_dir, img_size, thr

def _fallback_calibrate_threshold(raw_dir: Path, model: torch.nn.Module, img_size: int, device: str,
                                  limit: int = 64) -> float:
    """กรณี threshold เดิมเป็นศูนย์/หายไป: ประเมินจากรูป normal ใน raw_dir ที่ให้มา"""
    if not raw_dir.exists():
        return 0.1
    # เก็บรูปได้มากสุด limit
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
            # ใช้ TTA เบา ๆ เพื่อให้รีคอนสตรัคต์เสถียรขึ้น
            xhat = _reconstruct_tta_lr(model, x)
            # ใช้สกอร์เดียวกับตอนเทรน: mean/top-k diff + SSIM
            abs_diff = torch.abs(xhat - x)
            ssim_val = ssim_calc(xhat.clamp(0,1), x.clamp(0,1))
            score = compute_anomaly_score(abs_diff, ssim_val)
            errs.extend(score.detach().cpu().tolist())
    errs = np.asarray(errs, dtype=np.float32)
    return _robust_threshold(errs)

# ----- public API used by main.py -----
def test_images(model_id: str, models_root: Path, buf_list: List[Tuple[str, bytes]]):
    """
    คืน list ของ dict:
    {
      filename, score, thr, is_anomaly,
      image_url, heatmap_url, overlay_url,
      result_image, threshold  # alias รองรับ UI เดิม
    }
    URLs ชี้ไปที่ /static/preview/<model_id>/...
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, raw_dir, img_size, thr = _load_model_and_cfg(model_id, device)

    # Fallback calibrate ถ้า threshold เป็น 0/หาย
    if thr <= 1e-12 and raw_dir is not None:
        thr = _fallback_calibrate_threshold(raw_dir, model, img_size, device)

    # ---- โฟลเดอร์พรีวิว (เสิร์ฟผ่าน /static) ----
    # ใช้ preview ทั้งโฟลเดอร์และ URL เพื่อให้ตรงกัน
    DATA_DIR   = DATASETS_DIR.parent  # data/
    preview_dir = DATA_DIR / "preview" / model_id
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

            # คำนวณ score ให้สอดคล้องกับตอนเทรน
            ssim_val = ssim_calc(xhat.clamp(0,1), x.clamp(0,1))
            score = float(compute_anomaly_score(abs_diff, ssim_val).item())

            # heatmap + overlay
            hm      = _heatmap_from_abs(abs_diff)
            overlay = _colorize_overlay(bgr, hm, alpha=0.45)

            # ถ้าเกิน threshold ให้หา bounding boxes จาก heatmap แล้ววาดทับบน overlay
            bboxes: List[Tuple[int,int,int,int]] = []
            if score > thr:
                H, W = bgr.shape[:2]
                bboxes = _extract_bboxes_from_heatmap(hm, (H, W), min_area_ratio=0.001, min_box_size=6, max_boxes=5)
                for (x, y, w, h) in bboxes:
                    cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 0, 255), 2)
                if bboxes:
                    # ใส่ป้ายกำกับเล็ก ๆ มุมซ้ายบนของกล่องแรก
                    bx, by, bw, bh = bboxes[0]
                    cv2.putText(overlay, 'ANOMALY', (bx, max(0, by-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

            # บันทึกไฟล์
            stem    = Path(fname).stem
            out_img = preview_dir / f"{stem}_input.jpg"
            out_hm  = preview_dir / f"{stem}_heatmap.jpg"
            out_ovr = preview_dir / f"{stem}_overlay.jpg"
            cv2.imwrite(str(out_img), bgr)
            cv2.imwrite(str(out_hm),  (hm * 255).astype(np.uint8))
            cv2.imwrite(str(out_ovr), overlay)

            url_img = f"{base_url}/{stem}_input.jpg"
            url_hm  = f"{base_url}/{stem}_heatmap.jpg"
            url_ovr = f"{base_url}/{stem}_overlay.jpg"

            items.append({
                "filename": fname,
                "score": round(score, 6),
                "thr": float(thr),
                "is_anomaly": bool(score > thr),
                "image_url":   url_img,
                "heatmap_url": url_hm,
                "overlay_url": url_ovr,
                "bboxes": bboxes,
                # aliases สำหรับ UI เดิม:
                "result_image": url_ovr,      # ใช้ overlay เป็นภาพหลัก
                "threshold": float(thr),
            })

    return items

