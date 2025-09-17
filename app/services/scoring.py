from __future__ import annotations

import torch
from torch import Tensor


def compute_anomaly_score(abs_diff: Tensor, ssim_val: Tensor, topk_ratio: float = 0.01) -> Tensor:
    """Aggregate reconstruction error to highlight small localized anomalies.

    abs_diff: (N,C,H,W) absolute difference between reconstruction and input.
    ssim_val: (N,) SSIM scores between reconstruction and input.
    topk_ratio: fraction of highest-error pixels used for the localized term.
    """
    if abs_diff.ndim != 4:
        raise ValueError("abs_diff must be a 4D tensor (N,C,H,W)")
    if ssim_val.ndim != 1 or ssim_val.shape[0] != abs_diff.shape[0]:
        raise ValueError("ssim_val must be shape (N,) and match batch size")

    per_pixel = abs_diff.mean(1)  # (N,H,W)
    flat = per_pixel.flatten(1)
    mean_err = flat.mean(1)

    numel = flat.shape[1]
    k = max(1, int(numel * topk_ratio))
    topk_mean = torch.topk(flat, k=k, dim=1).values.mean(1)

    # Highlight both global deviation and concentrated hotspots; keep SSIM term for texture changes.
    score = 0.35 * mean_err + 0.35 * topk_mean + 0.15 * (1.0 - ssim_val.to(abs_diff.dtype))
    return score
