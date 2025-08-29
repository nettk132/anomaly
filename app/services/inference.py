from __future__ import annotations
import io, json, yaml
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2
import torch
import torch.nn.functional as F

from ..models.autoencoder import ConvAE
from ..config import DATASETS_DIR, MODELS_DIR
from .storage import get_raw_dir

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

def _heatmap_from_diff(diff_sq: torch.Tensor) -> np.ndarray:
    """
    diff_sq: (1,3,H,W) ค่า (xhat - x)**2
    return: heatmap float32 [0..1] (H,W)
    """
    hm = diff_sq.mean(1).squeeze(0)      # (H,W)
    hm = hm / (hm.max() + 1e-12)
    return hm.detach().cpu().numpy().astype(np.float32)

def _colorize_overlay(img_bgr: np.ndarray, heatmap01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    hm = cv2.resize(heatmap01, (w, h), interpolation=cv2.INTER_CUBIC)
    hm_color = cv2.applyColorMap((hm*255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 1.0, hm_color, alpha, 0.0)
    return overlay

# ----- core loading -----
def _load_model_and_cfg(model_id: str, device: str):
    model_dir = MODELS_DIR / model_id
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    # config.yaml: scene_id, img_size
    with open(model_dir / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    scene_id = cfg.get("scene_id")
    img_size = int(cfg.get("img_size", 256))

    # model.pt
    model = ConvAE().to(device).eval()
    state = torch.load(model_dir / "model.pt", map_location=device)
    model.load_state_dict(state)

    # threshold.json
    thr = 0.0
    th_path = model_dir / "threshold.json"
    if th_path.exists():
        try:
            with open(th_path, "r", encoding="utf-8") as f:
                thj = json.load(f)
            thr = float(thj.get("threshold_mse", 0.0))
        except Exception:
            thr = 0.0

    return model, scene_id, img_size, thr

def _fallback_calibrate_threshold(scene_id: str, model: torch.nn.Module, img_size: int, device: str,
                                  limit: int = 64) -> float:
    """กรณี threshold เดิมเป็นศูนย์/หายไป: ประเมินจากรูป normal ใน scene นั้น ๆ"""
    raw_dir = get_raw_dir(scene_id)
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
    with torch.no_grad():
        for p in img_paths:
            bgr = cv2.imread(str(p))
            if bgr is None: 
                continue
            x = _to_tensor_rgb01(bgr, img_size).to(device)
            xhat = model(x)
            err = F.mse_loss(xhat, x, reduction="none").flatten(1).mean(1)
            errs.extend(err.detach().cpu().tolist())
    errs = np.asarray(errs, dtype=np.float32)
    return _robust_threshold(errs)

# ----- public API used by main.py -----
def test_images(model_id: str, models_root: Path, buf_list: List[Tuple[str, bytes]]):
    """
    คืน list ของ dict:
    {
      filename, score, thr, is_anomaly,
      image_url, heatmap_url, overlay_url
    }
    URLs ชี้ไปที่ /static/preview/<model_id>/...
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, scene_id, img_size, thr = _load_model_and_cfg(model_id, device)

    # Fallback calibrate ถ้า threshold เป็น 0/หาย
    if thr <= 1e-12:
        thr = _fallback_calibrate_threshold(scene_id, model, img_size, device)

    # โฟลเดอร์พรีวิว (เสิร์ฟผ่าน /static)
    DATA_DIR = DATASETS_DIR.parent             # data/
    preview_dir = DATA_DIR / "preview" / model_id
    preview_dir.mkdir(parents=True, exist_ok=True)

    items = []
    with torch.no_grad():
        for fname, content in buf_list:
            arr = np.frombuffer(content, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                # รูปพัง
                items.append({
                    "filename": fname, "score": None, "thr": float(thr),
                    "is_anomaly": None, "image_url": None,
                    "heatmap_url": None, "overlay_url": None,
                    "error": "cannot decode image"
                })
                continue

            x = _to_tensor_rgb01(bgr, img_size).to(device)
            xhat = model(x)
            diff_sq = (xhat - x) ** 2

            # สกอร์ภาพ (MSE เฉลี่ย)
            score = float(diff_sq.flatten(1).mean(1).item())

            # heatmap + overlay
            hm = _heatmap_from_diff(diff_sq)
            overlay = _colorize_overlay(bgr, hm, alpha=0.45)

            # บันทึกไฟล์
            stem = Path(fname).stem
            out_img   = preview_dir / f"{stem}_input.jpg"
            out_hm    = preview_dir / f"{stem}_heatmap.jpg"
            out_ovr   = preview_dir / f"{stem}_overlay.jpg"
            # เซฟต้นฉบับไว้ด้วย (เผื่อ UI อยากกดดู)
            cv2.imwrite(str(out_img), bgr)
            cv2.imwrite(str(out_hm),  (hm*255).astype(np.uint8))
            cv2.imwrite(str(out_ovr), overlay)

            base_url = f"/static/preview/{model_id}"
            items.append({
                "filename": fname,
                "score": round(score, 6),
                "thr":    float(thr),
                "is_anomaly": bool(score > thr),
                "image_url":   f"{base_url}/{stem}_input.jpg",
                "heatmap_url": f"{base_url}/{stem}_heatmap.jpg",
                "overlay_url": f"{base_url}/{stem}_overlay.jpg",
            })

    return items
