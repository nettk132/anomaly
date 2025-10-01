# app/services/training.py
from __future__ import annotations
import time, json, yaml
from pathlib import Path
from typing import Optional
import numpy as np

import torch, torch.nn as nn
import torch.nn.functional as F

from ..models.autoencoder import ConvAE, build_autoencoder
from .scoring import compute_anomaly_score
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


def resolve_model_checkpoint(model_id: str) -> tuple[Path, Optional[str]]:
    scene_path = MODELS_DIR / model_id / "model.pt"
    if scene_path.exists():
        return scene_path, None
    if PROJECTS_DIR.exists():
        for proj_dir in PROJECTS_DIR.iterdir():
            if not proj_dir.is_dir():
                continue
            candidate = proj_dir / "models" / model_id / "model.pt"
            if candidate.exists():
                return candidate, proj_dir.name
            imports_candidate = proj_dir / "imports" / model_id / "model.pt"
            if imports_candidate.exists():
                return imports_candidate, proj_dir.name
    raise FileNotFoundError(f"base model {model_id} not found")


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
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    l1  = nn.L1Loss(reduction='mean')
    ssim = _AvgPoolSSIM(ksize=7).to(device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    use_amp = (device == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Early stopping
    best_val = float('inf'); best_state = None
    patience = max(5, epochs // 4); bad = 0

    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, epochs * steps_per_epoch)
    step = 0
    ep = 0

    for ep in range(1, epochs + 1):
        model.train(); running = 0.0
        for x in train_loader:
            x = x.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                xhat = model(x)
                ssim_val = ssim(xhat.clamp(0,1), x.clamp(0,1))  # (N,)
                loss = 0.7 * l1(xhat, x) + 0.3 * (1.0 - ssim_val.mean()) * 0.5

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
                vloss = 0.7 * l1(xhat, x) + 0.3 * (1.0 - ssim_val.mean()) * 0.5
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
        # step scheduler per epoch
        try:
            scheduler.step()
        except Exception:
            pass

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Calibrate threshold ----
    model.eval(); errs = []
    ssim_cal = _AvgPoolSSIM(ksize=7).to(device)
    with torch.no_grad():
        for x in val_loader:
            x = x.to(device, non_blocking=True)
            xhat = model(x)
            abs_diff = torch.abs(xhat - x)
            l1_per = abs_diff.flatten(1).mean(1)
            ssim_val = ssim_cal(xhat.clamp(0,1), x.clamp(0,1))
            score = compute_anomaly_score(abs_diff, ssim_val)
            errs.extend(score.detach().cpu().tolist())
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
                'created_at': ts,
                'img_size': img_size,
                'epochs_run': ep,
                'lr': lr,
                'device': device,
                'loss': '0.7*L1 + 0.3*(1-SSIM)/2',
                'early_stopping_patience': patience,
            },
            f, allow_unicode=True,
        )

    stats = {
        'threshold_mse': float(threshold),  # kept for compatibility
        'threshold_score': float(threshold),
        'score_type': '0.35*meanL1 + 0.35*top1%L1 + 0.15*(1-SSIM)',
        'val_size': int(errs.size),
        'val_err_mean': float(np.mean(errs)) if errs.size else None,
        'val_err_p99': float(np.percentile(errs, 99)) if errs.size else None,
        'val_err_p995': float(np.percentile(errs, 99.5)) if errs.size >= 50 else None,
        'val_err_median': float(np.median(errs)) if errs.size else None,
        'val_err_mad': float(np.median(np.abs(errs - np.median(errs)))) if errs.size else None,
        'best_val_loss': float(best_val),
        'note': 'score=0.35*meanL1+0.35*top1%L1+0.15*(1-SSIM); thr=max(P99.5, median+3*1.4826*MAD)',
    }
    with open(out_dir / 'threshold.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    job.model_id = model_id
    job.progress = 1.0


# ============================================================
# 2) โหมดโปรเจกต์: ตาม project_id → เซฟใต้ data/projects/<project_id>/models/<model_id>/
# ============================================================
def train_job_project(job, project_id: str, raw_dir: Path, img_size: int, epochs: int, lr: float,
                      training_mode: str = "anomaly", base_model_id: Optional[str] = None):
    """Train autoencoder for a project, with optional fine-tune mode."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, val_loader, _ = make_loaders(raw_dir, img_size=img_size)

    mode = (training_mode or 'anomaly').strip().lower()
    if mode != 'finetune':
        mode = 'anomaly'

    base_origin: Optional[str] = None
    missing_keys: list[str] = []
    unexpected_keys: list[str] = []

    if mode == 'finetune':
        if not base_model_id:
            raise ValueError('base_model_id is required for finetune training')
        base_path, base_origin = resolve_model_checkpoint(base_model_id)
        state = torch.load(base_path, map_location='cpu')
        try:
            model = build_autoencoder(state)
            missing_keys = []
            unexpected_keys = []
        except Exception:
            model = ConvAE()
            load_info = model.load_state_dict(state, strict=False)
            missing_keys = list(getattr(load_info, 'missing_keys', []))
            unexpected_keys = list(getattr(load_info, 'unexpected_keys', []))
        model = model.to(device)
        detail_parts = [f"fine-tune from {base_model_id}"]
        if base_origin:
            detail_parts.append(f"(origin: {base_origin})")
        if missing_keys or unexpected_keys:
            detail_parts.append(f"(missing={len(missing_keys)}, unexpected={len(unexpected_keys)})")
        job.detail = ' '.join(detail_parts)
    else:
        base_model_id = None
        model = ConvAE().to(device)
        job.detail = 'training anomaly autoencoder'

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    l1  = nn.L1Loss(reduction='mean')
    ssim = _AvgPoolSSIM(ksize=7).to(device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    use_amp = (device == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_val = float('inf'); best_state = None
    patience = max(5, epochs // 4); bad = 0

    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, epochs * steps_per_epoch)
    step = 0
    ep = 0

    for ep in range(1, epochs + 1):
        model.train(); running = 0.0
        for x in train_loader:
            x = x.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                xhat = model(x)
                ssim_val = ssim(xhat.clamp(0,1), x.clamp(0,1))
                loss = 0.7 * l1(xhat, x) + 0.3 * (1.0 - ssim_val.mean()) * 0.5

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt); scaler.update()

            running += float(loss.item()) * x.size(0)
            step += 1
            job.progress = min(0.99, step / total_steps)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device, non_blocking=True)
                xhat = model(x)
                ssim_val = ssim(xhat.clamp(0,1), x.clamp(0,1))
                vloss = 0.7 * l1(xhat, x) + 0.3 * (1.0 - ssim_val.mean()) * 0.5
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
        try:
            scheduler.step()
        except Exception:
            pass

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval(); errs = []
    ssim_cal = _AvgPoolSSIM(ksize=7).to(device)
    with torch.no_grad():
        for x in val_loader:
            x = x.to(device, non_blocking=True)
            xhat = model(x)
            abs_diff = torch.abs(xhat - x)
            ssim_val = ssim_cal(xhat.clamp(0,1), x.clamp(0,1))
            score = compute_anomaly_score(abs_diff, ssim_val)
            errs.extend(score.detach().cpu().tolist())
    errs = np.asarray(errs, dtype=np.float32)
    threshold = _robust_threshold(errs)

    ts = time.strftime('%Y%m%d-%H%M%S')
    model_id = f'{project_id}-{ts}'
    out_dir = PROJECTS_DIR / project_id / 'models' / model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_dir / 'model.pt')
    config = {
        'project_id': project_id,
        'created_at': ts,
        'img_size': img_size,
        'epochs_run': ep,
        'lr': lr,
        'device': device,
        'loss': '0.7*L1 + 0.3*(1-SSIM)/2',
        'early_stopping_patience': patience,
        'training_mode': mode,
    }
    if mode == 'finetune':
        config['base_model_id'] = base_model_id
        if base_origin:
            config['base_model_origin'] = base_origin
        if missing_keys:
            config['base_model_missing_keys'] = missing_keys
        if unexpected_keys:
            config['base_model_unexpected_keys'] = unexpected_keys
    with open(out_dir / 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, allow_unicode=True)

    stats = {
        'threshold_mse': float(threshold),
        'threshold_score': float(threshold),
        'score_type': '0.35*meanL1 + 0.35*top1%L1 + 0.15*(1-SSIM)',
        'val_size': int(errs.size),
        'val_err_mean': float(np.mean(errs)) if errs.size else None,
        'val_err_p99': float(np.percentile(errs, 99)) if errs.size else None,
        'val_err_p995': float(np.percentile(errs, 99.5)) if errs.size >= 50 else None,
        'val_err_median': float(np.median(errs)) if errs.size else None,
        'val_err_mad': float(np.median(np.abs(errs - np.median(errs)))) if errs.size else None,
        'best_val_loss': float(best_val),
        'note': 'score=0.35*meanL1+0.35*top1%L1+0.15*(1-SSIM); thr=max(P99.5, median+3*1.4826*MAD)',
    }
    with open(out_dir / 'threshold.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    if callable(set_last_model):
        try:
            set_last_model(project_id, model_id)  # type: ignore
        except Exception:
            pass

    job.model_id = model_id
    job.progress = 1.0
