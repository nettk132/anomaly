from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- Blocks ---------
class ConvBlock(nn.Module):
    """Conv -> GroupNorm -> ReLU"""
    def __init__(self, c_in, c_out, k=3, s=1, p=1, groups=8):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, p)
        self.gn   = nn.GroupNorm(num_groups=min(groups, c_out), num_channels=c_out)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))

class Down(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(c_in, c_out),
            ConvBlock(c_out, c_out),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        h = self.block(x)
        return self.pool(h), h  # return pooled + skip

class Up(nn.Module):
    """Bilinear upsample -> Conv to reduce channels -> fuse skip -> ConvBlock x2"""
    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c_in, c_out, 3, padding=1),
        )
        self.block = nn.Sequential(
            ConvBlock(c_out + c_skip, c_out),
            ConvBlock(c_out, c_out),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # handle odd sizes (shouldn't happen if input multiple of 8, but safe-guard)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)

# --------- Model ---------
class ConvAE(nn.Module):
    """
    Small U-Net style Autoencoder
    - เหมาะกับข้อมูลน้อย
    - ใช้ GroupNorm (ทน batch เล็ก)
    - ใช้ Upsample+Conv (ลด checkerboard)
    """
    def __init__(self, in_ch: int = 3, base: int = 32):
        super().__init__()
        # Encoder
        self.down1 = Down(in_ch, base)        # -> base,  H/2
        self.down2 = Down(base, base*2)       # -> 2b,   H/4
        self.down3 = Down(base*2, base*4)     # -> 4b,   H/8
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(base*4, base*8),
            ConvBlock(base*8, base*8),
        )
        # Decoder
        self.up3 = Up(base*8, base*4, base*4) # -> 4b,   H/4
        self.up2 = Up(base*4, base*2, base*2) # -> 2b,   H/2
        self.up1 = Up(base*2, base,   base)   # -> b,    H
        self.out = nn.Conv2d(base, in_ch, 1)
        self.act_out = nn.Sigmoid()           # อินพุตควรสเกลเป็น [0,1]

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
        # ส่ง z + skips กลับ เพื่อให้ decode ใช้
        return z, (x3, x2, x1)

    def decode(self, z_and_skips):
        if isinstance(z_and_skips, tuple):
            z, (x3, x2, x1) = z_and_skips
        else:
            # เผื่อใช้แบบเดิม (ไม่มี skip) ให้ up อย่างเดียว
            z, (x3, x2, x1) = z_and_skips, (0,0,0)  # จะไม่ถูกใช้

        y = self.up3(z, x3)
        y = self.up2(y, x2)
        y = self.up1(y, x1)
        y = self.act_out(self.out(y))
        return y

    def forward(self, x):
        z, skips = self.encode(x)
        return self.decode((z, skips))

# เพื่อความเข้ากันได้กับโค้ดเดิม
ConvAutoencoder = ConvAE
