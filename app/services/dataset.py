from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Callable
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from ..config import ALLOWED_EXTS

class ImageFolderDataset(Dataset):
    def __init__(self, root: Path, img_size: int = 256):
        self.paths: List[Path] = []
        for p in root.glob('*'):
            if p.suffix.lower() in ALLOWED_EXTS:
                self.paths.append(p)
        self.paths.sort()
        self.t = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),  # [0,1]
        ])

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        with Image.open(p).convert('RGB') as im:
            x = self.t(im)
        return x

class ImagePathsDataset(Dataset):
    """Dataset จากลิสต์พาธ + ทรานส์ฟอร์มที่กำหนด (ใช้ทำ train/val แยกทรานส์ฟอร์มกัน)"""
    def __init__(self, paths: List[Path], transform: Callable):
        self.paths = paths
        self.t = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        with Image.open(p).convert('RGB') as im:
            x = self.t(im)
        return x

def _build_transform(img_size: int, aug: bool) -> T.Compose:
    # ออกแบบให้เบาและปลอดภัยกับ anomaly detection
    if aug:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
            T.ToTensor(),  # [0,1]
            # noise เล็กน้อยช่วยให้ generalize ดีขึ้น
            T.RandomErasing(p=0.05, scale=(0.01, 0.03), ratio=(0.3, 3.3), value='random'),
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])

def make_loaders(raw_dir: Path, img_size: int, batch_size: int = 32, val_ratio: float = 0.2, seed: int = 42):
    # สแกนพาธทีเดียว เพื่อให้ train/val ใช้ลิสต์เดียวกัน
    paths: List[Path] = []
    for p in raw_dir.glob('*'):
        if p.suffix.lower() in ALLOWED_EXTS:
            paths.append(p)
    paths.sort()

    if len(paths) < 5:
        raise ValueError(f'Not enough images in {raw_dir} (got {len(paths)}, need >=5)')

    val_len = max(1, int(len(paths) * val_ratio))
    train_len = len(paths) - val_len

    g = torch.Generator().manual_seed(seed)
    # สุ่ม index แทนการ split dataset ตัวเดียว เพื่อให้กำหนดทรานส์ฟอร์มต่างกัน
    perm = torch.randperm(len(paths), generator=g).tolist()
    train_idx = perm[:train_len]
    val_idx   = perm[train_len:]

    train_paths = [paths[i] for i in train_idx]
    val_paths   = [paths[i] for i in val_idx]

    train_ds = ImagePathsDataset(train_paths, _build_transform(img_size, aug=True))
    val_ds   = ImagePathsDataset(val_paths,   _build_transform(img_size, aug=False))

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    return train_loader, val_loader, len(paths)
