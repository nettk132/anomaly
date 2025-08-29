# app/services/training.py
from __future__ import annotations
import time, json, yaml
from pathlib import Path
from typing import Optional
import numpy as np

import torch, torch.nn as nn
import torch.nn.functional as F

from ..models.autoencoder import ConvAE
from .dataset import make_loaders
from ..config import MODELS_DIR, PROJECTS_DIR

# (optional) อัปเดต meta ของโปรเจกต์ ถ้ามีไฟล์ services/projects.py แล้ว
try:
    from .projects import set_last_model  # type: ignore
except Exception:
    set_last_model = None  # ไม่บังคับ

# ---------- SSIM (แบบเบา ใช้ AvgPool) ----------
class _AvgPoolSSIM(nn.Module):
    def __init__(self, ksize: int = 7):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=ksize, stride=1, padding=ksize // 2)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # คาดว่า x,y ∈ [0,1]
        mu_x = self.pool(x)
        mu_y = self.pool(y)
        sigma_x = self.pool(x * x) - mu_x * mu_x
        sigma_y = self.pool(y * y) - mu_y * mu_y
        sigma_xy = self.pool(x * y) - mu_x * mu_y

        num = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        den = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        ssim_map = num / (den + 1e-12)
        return ssim_map.flatten(1).mean(1)  # (N,)

def _robust_threshold(errs: np.ndarray) -> float:
    """รวมหลายเกณฑ์: P99–P99.5 และ median+3*MAD แล้วเลือกค่าที่เข้มกว่า"""
    if errs.size == 0:
        return 0.1
    p99  = np.percentile(errs, 99)
    p995 = np.percentile(errs, 99.5) if errs.size >= 50 else p99
    med = np.median(errs)
    mad = np.median(np.abs(errs - med)) + 1e-12
    robust = float(med + 3.0 * 1.4826 * mad)
    return float(max(p995, robust))

# ============================================================
# 1) โหมดเดิม: ตาม scene_id  → เซฟไว้ใต้ data/models/<model_id>/
# ============================================================
def train_job(job, scene_id: str, raw_dir: Path, img_size: int, epochs: int, lr: float):
    """
    เทรน ConvAE + SSIM, EarlyStopping, AMP, Gradient Clipping
    เซฟเวทที่ดีที่สุดบน val และคาลิเบรต threshold แบบ robust (โหมด scene)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # DataLoader (อินพุตควรอยู่ช่วง [0,1])
    train_loader, val_loader, _ = make_loaders(raw_dir, img_size=img_size)

    model = ConvAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    mse = nn.MSELoss(reduction='mean')
    ssim = _AvgPoolSSIM(ksize=7).to(device)

    use_amp = (device == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Early stopping
    best_val = float('inf'); best_state = None
    patience = max(5, epochs // 4); bad = 0

    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, epochs * steps_per_epoch)
    step = 0

    for ep in range(1, epochs + 1):
        model.train(); running = 0.0
        for x in train_loader:
            x = x.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                xhat = model(x)
                ssim_val = ssim(xhat.clamp(0,1), x.clamp(0,1))
                loss = 0.7 * mse(xhat, x) + 0.3 * (1.0 - ssim_val.mean()) * 0.5

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt); scaler.update()

            running += float(loss.item()) * x.size(0)
            step += 1
            job.progress = min(0.99, step / total_steps)

        # ---- Validation ----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device, non_blocking=True)
                xhat = model(x)
                ssim_val = ssim(xhat.clamp(0,1), x.clamp(0,1))
                vloss = 0.7 * mse(xhat, x) + 0.3 * (1.0 - ssim_val.mean()) * 0.5
                val_losses.append(float(vloss.item()))
        mean_val = float(np.mean(val_losses)) if val_losses else float('inf')

        if mean_val < best_val - 1e-6:
            best_val = mean_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Calibrate threshold ----
    model.eval(); errs = []
    with torch.no_grad():
        for x in val_loader:
            x = x.to(device, non_blocking=True)
            xhat = model(x)
            err = ((xhat - x) ** 2).flatten(1).mean(1)
            errs.extend(err.detach().cpu().tolist())
    errs = np.asarray(errs, dtype=np.float32)
    threshold = _robust_threshold(errs)

    # ---- Save artifacts (scene mode) ----
    ts = time.strftime('%Y%m%d-%H%M%S')
    model_id = f'{scene_id}-{ts}'
    out_dir = MODELS_DIR / model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_dir / 'model.pt')
    with open(out_dir / 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(
            {
                'scene_id': scene_id,
                'img_size': img_size,
                'epochs_run': ep,
                'lr': lr,
                'device': device,
                'loss': '0.7*MSE + 0.3*(1-SSIM)/2',
                'early_stopping_patience': patience,
            },
            f, allow_unicode=True,
        )

    stats = {
        'threshold_mse': float(threshold),
        'val_size': int(errs.size),
        'val_err_mean': float(np.mean(errs)) if errs.size else None,
        'val_err_p99': float(np.percentile(errs, 99)) if errs.size else None,
        'val_err_p995': float(np.percentile(errs, 99.5)) if errs.size >= 50 else None,
        'val_err_median': float(np.median(errs)) if errs.size else None,
        'val_err_mad': float(np.median(np.abs(errs - np.median(errs)))) if errs.size else None,
        'best_val_loss': float(best_val),
        'note': 'threshold = max(P99.5, median + 3*1.4826*MAD) (ใช้ P99 ถ้า val เล็กมาก)',
    }
    with open(out_dir / 'threshold.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    job.model_id = model_id
    job.progress = 1.0


# ============================================================
# 2) โหมดโปรเจกต์: ตาม project_id → เซฟใต้ data/projects/<project_id>/models/<model_id>/
# ============================================================
def train_job_project(job, project_id: str, raw_dir: Path, img_size: int, epochs: int, lr: float):
    """
    เทรนเหมือน train_job แต่ผูกกับ project_id และบันทึกผลในโฟลเดอร์โปรเจกต์
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, val_loader, _ = make_loaders(raw_dir, img_size=img_size)

    model = ConvAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    mse = nn.MSELoss(reduction='mean')
    ssim = _AvgPoolSSIM(ksize=7).to(device)

    use_amp = (device == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val = float('inf'); best_state = None
    patience = max(5, epochs // 4); bad = 0

    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, epochs * steps_per_epoch)
    step = 0

    for ep in range(1, epochs + 1):
        model.train(); running = 0.0
        for x in train_loader:
            x = x.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                xhat = model(x)
                ssim_val = ssim(xhat.clamp(0,1), x.clamp(0,1))
                loss = 0.7 * mse(xhat, x) + 0.3 * (1.0 - ssim_val.mean()) * 0.5

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt); scaler.update()

            running += float(loss.item()) * x.size(0)
            step += 1
            job.progress = min(0.99, step / total_steps)

        # ---- Validation ----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device, non_blocking=True)
                xhat = model(x)
                ssim_val = ssim(xhat.clamp(0,1), x.clamp(0,1))
                vloss = 0.7 * mse(xhat, x) + 0.3 * (1.0 - ssim_val.mean()) * 0.5
                val_losses.append(float(vloss.item()))
        mean_val = float(np.mean(val_losses)) if val_losses else float('inf')

        if mean_val < best_val - 1e-6:
            best_val = mean_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Calibrate threshold ----
    model.eval(); errs = []
    with torch.no_grad():
        for x in val_loader:
            x = x.to(device, non_blocking=True)
            xhat = model(x)
            err = ((xhat - x) ** 2).flatten(1).mean(1)
            errs.extend(err.detach().cpu().tolist())
    errs = np.asarray(errs, dtype=np.float32)
    threshold = _robust_threshold(errs)

    # ---- Save artifacts (project mode) ----
    ts = time.strftime('%Y%m%d-%H%M%S')
    model_id = f'{project_id}-{ts}'
    out_dir = PROJECTS_DIR / project_id / 'models' / model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_dir / 'model.pt')
    with open(out_dir / 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(
            {
                'project_id': project_id,
                'img_size': img_size,
                'epochs_run': ep,
                'lr': lr,
                'device': device,
                'loss': '0.7*MSE + 0.3*(1-SSIM)/2',
                'early_stopping_patience': patience,
            },
            f, allow_unicode=True,
        )

    with open(out_dir / 'threshold.json', 'w', encoding='utf-8') as f:
        json.dump({'threshold_mse': float(threshold), 'val_size': int(errs.size)}, f, ensure_ascii=False, indent=2)

    # อัปเดต meta โปรเจกต์ (ถ้ามีฟังก์ชัน)
    if callable(set_last_model):
        try:
            set_last_model(project_id, model_id)  # type: ignore
        except Exception:
            pass

    job.model_id = model_id
    job.progress = 1.0
