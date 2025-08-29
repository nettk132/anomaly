import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
DATASETS_DIR = DATA_DIR / 'datasets'
MODELS_DIR = DATA_DIR / 'models'
TMP_DIR = ROOT / 'tmp'
PROJECTS_DIR  = DATA_DIR / "projects"
        
for d in (DATA_DIR, DATASETS_DIR, MODELS_DIR, TMP_DIR):
    d.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTS = {'.jpg', '.jpeg', '.png'}
MAX_FILES_PER_UPLOAD = 500  # adjust as needed
