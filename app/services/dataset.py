from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
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

def make_loaders(raw_dir: Path, img_size: int, batch_size: int = 32, val_ratio: float = 0.2, seed: int = 42):
    ds = ImageFolderDataset(raw_dir, img_size=img_size)
    if len(ds) < 5:
        raise ValueError(f'Not enough images in {raw_dir} (got {len(ds)}, need >=5)')
    val_len = max(1, int(len(ds)*val_ratio))
    train_len = len(ds) - val_len
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [train_len, val_len], generator=g)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, len(ds)
