from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ROOT


# ============================================================
# DINO ConvNeXt backbone + decoder for reconstruction
# ============================================================
# ============================================================
# Utility helpers
# ============================================================

def _calc_groups(channels: int, max_groups: int = 32) -> int:
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class LayerNorm2d(nn.Module):
    """LayerNorm implemented for channels-first tensors."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,C,H,W)
        orig_type = x.dtype
        x_perm = x.permute(0, 2, 3, 1)
        weight = self.weight
        bias = self.bias
        if weight.dtype != x_perm.dtype:
            weight = weight.to(x_perm.dtype)
            bias = bias.to(x_perm.dtype)
        x_norm = F.layer_norm(x_perm, (x_perm.shape[-1],), weight, bias, self.eps)
        return x_norm.permute(0, 3, 1, 2).to(orig_type)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob <= 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_tensor = random_tensor.floor()
        return x.div(keep_prob) * binary_tensor


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim)) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)  # channels-first LN
        x = x.permute(0, 2, 3, 1)  # to channels-last for linear layers
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            gamma = self.gamma
            if gamma.dtype != x.dtype:
                gamma = gamma.to(x.dtype)
            x = x * gamma
        x = x.permute(0, 3, 1, 2)
        x = shortcut + self.drop_path(x)
        return x


class DinoConvNeXtBackbone(nn.Module):
    def __init__(self, in_chans: int = 3,
                 depths = (3, 3, 9, 3),
                 dims = (96, 192, 384, 768),
                 layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.dims = tuple(dims)
        self.depths = tuple(depths)

        self.downsample_layers = nn.ModuleList()
        # stem
        self.downsample_layers.append(nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0]),
        ))
        for i in range(1, 4):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm2d(dims[i-1]),
                nn.Conv2d(dims[i-1], dims[i], kernel_size=2, stride=2),
            ))

        self.stages = nn.ModuleList()
        dp_rates = [0.0] * sum(depths)
        cur = 0
        for stage_id, depth in enumerate(depths):
            blocks = []
            for _ in range(depth):
                blocks.append(ConvNeXtBlock(dims[stage_id], drop_path=dp_rates[cur], layer_scale_init_value=layer_scale_init_value))
                cur += 1
            self.stages.append(nn.Sequential(*blocks))

        # only stage 3 has weights in checkpoint, others acts as identity
        self.norms = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            LayerNorm2d(dims[-1]),
        ])
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)

    def forward(self, x: torch.Tensor):
        feats = []
        for i in range(len(self.stages)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            x = self.norms[i](x)
            feats.append(x)
        return feats


class ConvGNAct(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=padding)
        self.gn = nn.GroupNorm(num_groups=_calc_groups(c_out), num_channels=c_out)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class UpBlock(nn.Module):
    def __init__(self, c_in: int, c_skip: int, c_out: int):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvGNAct(c_in, c_out),
        )
        self.fuse = nn.Sequential(
            ConvGNAct(c_out + c_skip, c_out),
            ConvGNAct(c_out, c_out),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class DinoConvAE(nn.Module):
    """Autoencoder that uses a pretrained DINOv3 ConvNeXt-Tiny encoder."""

    def __init__(self, in_ch: int = 3, freeze_encoder: bool = True, weights_path: Optional[Path] = None):
        super().__init__()
        if in_ch != 3:
            raise ValueError("DINO ConvNeXt backbone currently supports only RGB inputs (3 channels)")

        w_path = Path(weights_path) if weights_path is not None else ROOT / "dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth"
        if not w_path.exists():
            raise FileNotFoundError(f"Pretrained weights not found: {w_path}")

        self.encoder = DinoConvNeXtBackbone(in_chans=in_ch)
        state = torch.load(str(w_path), map_location="cpu")
        missing, unexpected = self.encoder.load_state_dict(state, strict=False)
        # allow missing keys only if they belong to classification head which is absent in this checkpoint
        if unexpected:
            raise RuntimeError(f"Unexpected keys when loading DINO weights: {unexpected[:5]}")
        if missing:
            allowed_missing = {"norm.weight", "norm.bias"}
            if any(m not in allowed_missing for m in missing):
                raise RuntimeError(f"Missing keys when loading DINO weights: {missing}")

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        dims = self.encoder.dims
        self.up3 = UpBlock(dims[3], dims[2], dims[2])
        self.up2 = UpBlock(dims[2], dims[1], dims[1])
        self.up1 = UpBlock(dims[1], dims[0], dims[0])
        self.final = nn.Sequential(
            ConvGNAct(dims[0], dims[0] // 2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvGNAct(dims[0] // 2, dims[0] // 4),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(dims[0] // 4, in_ch, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        self._init_decoder_weights()

    def _init_decoder_weights(self):
        for module in [self.up3, self.up2, self.up1, self.final]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        feats = self.encoder(x)
        z = feats[-1]
        skips = (feats[2], feats[1], feats[0])  # H/16, H/8, H/4
        return z, skips

    def decode(self, enc_out: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        if not isinstance(enc_out, tuple) or len(enc_out) != 2:
            raise ValueError("decode expects a tuple (z, skips)")
        z, skips = enc_out
        s2, s1, s0 = skips
        y = self.up3(z, s2)
        y = self.up2(y, s1)
        y = self.up1(y, s0)
        y = self.final(y)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


# ============================================================
# Legacy lightweight ConvAE kept for backward compatibility
# ============================================================
class LegacyConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1, groups=8):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, p)
        self.gn   = nn.GroupNorm(num_groups=min(groups, c_out), num_channels=c_out)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class LegacyDown(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.block = nn.Sequential(
            LegacyConvBlock(c_in, c_out),
            LegacyConvBlock(c_out, c_out),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        h = self.block(x)
        return self.pool(h), h


class LegacyUp(nn.Module):
    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c_in, c_out, 3, padding=1),
        )
        self.block = nn.Sequential(
            LegacyConvBlock(c_out + c_skip, c_out),
            LegacyConvBlock(c_out, c_out),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class LegacyConvAE(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 32):
        super().__init__()
        self.down1 = LegacyDown(in_ch, base)
        self.down2 = LegacyDown(base, base * 2)
        self.down3 = LegacyDown(base * 2, base * 4)
        self.bottleneck = nn.Sequential(
            LegacyConvBlock(base * 4, base * 8),
            LegacyConvBlock(base * 8, base * 8),
        )
        self.up3 = LegacyUp(base * 8, base * 4, base * 4)
        self.up2 = LegacyUp(base * 4, base * 2, base * 2)
        self.up1 = LegacyUp(base * 2, base, base)
        self.out = nn.Conv2d(base, in_ch, 1)
        self.act_out = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x):
        x1p, x1 = self.down1(x)
        x2p, x2 = self.down2(x1p)
        x3p, x3 = self.down3(x2p)
        z = self.bottleneck(x3p)
        return z, (x3, x2, x1)

    def decode(self, z_and_skips):
        if isinstance(z_and_skips, tuple):
            z, (x3, x2, x1) = z_and_skips
        else:
            z, (x3, x2, x1) = z_and_skips, (0, 0, 0)
        y = self.up3(z, x3)
        y = self.up2(y, x2)
        y = self.up1(y, x1)
        y = self.act_out(self.out(y))
        return y

    def forward(self, x):
        z, skips = self.encode(x)
        return self.decode((z, skips))


def build_autoencoder(state_dict: Optional[dict] = None) -> nn.Module:
    """Instantiate model that matches the given state_dict (legacy vs new)."""
    if state_dict is not None:
        keys = list(state_dict.keys())
        if any(k.startswith("down1.") for k in keys):
            model = LegacyConvAE()
        else:
            model = DinoConvAE()
        model.load_state_dict(state_dict)
        return model
    return DinoConvAE()


# default class used by training/inference
ConvAE = DinoConvAE
ConvAutoencoder = ConvAE
LegacyConvAutoencoder = LegacyConvAE
