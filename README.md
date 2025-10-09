# Anomaly Detection Backend

FastAPI backend that trains an anomaly-detection autoencoder on **normal** images and serves
reconstruction-based scoring plus visual previews. The service supports both the legacy
scene_id workflow and a newer project-centric flow with lightweight HTML pages under /ui.

## Features
- Upload normal samples (scene or project scope) and trigger background training jobs.
- DINOv3 ConvNeXt-Tiny encoder + custom decoder autoencoder with SSIM-aware loss, AMP, and early stopping.
- Automatic threshold calibration on validation data and downloadable model artifacts.
- Inference endpoint returns anomaly scores, heatmaps, and overlay images with optional bounding boxes.
- Simple job tracker (/jobs/{job_id}) and static file hosting for previews and saved models.
- Optional YOLO object-detection workflow with dataset upload, training, and inference UI.

## Requirements
- Python 3.10 or newer (tested on 3.11).
- PyTorch 2.x with CUDA if you want GPU acceleration (CPU works but is slower).
- DINOv3 ConvNeXt-Tiny weights file (dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth) placed in the repository root.

## Quick Start
1. Create a virtual environment and install core dependencies:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate

   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. Install the PyTorch stack that matches your platform/GPU. Examples:
   ```bash
   # CUDA 12.1 build
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

   # CPU-only build
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```
3. Download the DINOv3 ConvNeXt-Tiny checkpoint from the official Meta research release
   (requires accepting their license) and place it next to this README:
   ```text
   project-root/
     dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth
     app/
     data/
     ...
   ```
4. Launch the API:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
5. Open http://localhost:8000/ui/projects (if the ui/ folder is present) or inspect
   the OpenAPI docs at http://localhost:8000/docs.

## Configuration

Runtime configuration is handled by an optional [`config.yaml`](./config.yaml) file and environment variables with the
`APP_` prefix. Copy [`config.example.yaml`](./config.example.yaml) to `config.yaml` and tweak paths, upload limits, and
training defaults for your environment. Every key can also be overridden via env vars, for example:

```bash
export APP_DATA_DIR=/mnt/storage/anomaly-data
export APP_UPLOADS__MAX_TOTAL_SIZE_MB=256
```

Key settings:
- `data_dir`, `tmp_dir` – filesystem layout for datasets, models, previews.
- `uploads` – allowed extensions plus per-file and aggregate upload limits (enforced on the backend).
- `training` – default batch size, validation split, and early-stopping patience.
- `jobs.max_workers` – size of the background training thread pool.

All directories are created automatically on startup.

## Testing

The repository now ships with a pytest suite covering the job manager, upload pipeline, and representative API flows.
After installing the requirements run:

```bash
pytest
```

Add new tests alongside features to keep the pipeline green.

## Data Layout
Uploads and training artifacts live under data/ (created automatically):
- data/datasets/{scene_id}/raw/ - raw images for the legacy scene workflow.
- data/projects/{project_id}/raw/ - raw images in the project workflow.
- data/projects/{project_id}/models/{model_id}/ - model.pt, config.yaml, and threshold.json.
- data/models/{model_id}/ - same artifacts for scene-based runs.
- data/preview/{model_id}/ - generated previews (input, heatmap, overlay) from inference.

## Training Workflows
### Scene-based (legacy)
```bash
# Upload all "normal" images
curl -X POST "http://localhost:8000/datasets/pump-01/upload" \
     -F "files=@/path/image1.jpg" -F "files=@/path/image2.jpg"

# Kick off training (returns job_id)
curl -X POST "http://localhost:8000/train" \
     -H "Content-Type: application/json" \
     -d '{"scene_id":"pump-01","img_size":256,"epochs":30,"lr":1e-4}'

# Poll progress
curl http://localhost:8000/jobs/{job_id}
```

### Project-based
```bash
# Create a project (name is also the confirmation string for deletions)
curl -X POST http://localhost:8000/projects \
     -H "Content-Type: application/json" \
     -d '{"name":"widget-line","description":"Gear inspection"}'

# Upload images scoped to the project
curl -X POST "http://localhost:8000/projects/{project_id}/upload" \
     -F "files=@/path/image1.jpg"

# Train and poll just like the scene endpoint
curl -X POST "http://localhost:8000/projects/{project_id}/train" \
     -H "Content-Type: application/json" \
     -d '{"img_size":256,"epochs":40,"lr":5e-5}'
```

## Inference & Downloads
- POST /models/{model_id}/test - upload images to score; response includes URLs for input, heatmap,
  overlay, and bounding boxes where the anomaly threshold is exceeded.
- GET /models/{model_id}/download - download model.pt for scene models.
- GET /projects/{project_id}/models/{model_id}/download - download project-specific models.

## GPU Notes
- Training automatically switches to CUDA when torch.cuda.is_available().
- Automatic mixed precision now uses the modern torch.amp APIs. No changes
  are required from the client side; ensure your PyTorch installation includes GPU support.

## Troubleshooting
- **Missing DINO weights** - you will see FileNotFoundError referencing
  dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth. Place the file at the repository root.
- **Torch installation issues** - install torch/torchvision via the official index URLs shown above to avoid
  slow source builds on Windows.
- **Preview images not updating** - the UI caches aggressively; reload with cache disabled or hit the /tmp static mount directly.

## Development Tips
- Jobs run in a bounded background thread pool (see `app/services/jobs.py`), so long-running training will not block the API.
- All datasets and model artifacts stay under data/ so that the Git tree remains clean.
- Use the bundled pytest suite as a starting point for further tests when extending the system.

## License

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for details.

---

## ภาษาไทย (Thai Notes)

- สามารถกำหนดค่าระบบผ่านไฟล์ `config.yaml` (ดูตัวอย่างใน `config.example.yaml`) หรือใช้ environment variables ที่ขึ้นต้นด้วย `APP_` เช่น `APP_DATA_DIR`.
- ระบบอัปโหลดตรวจสอบนามสกุลและจำกัดขนาดไฟล์ต่อไฟล์/รวมในแต่ละคำขอเพื่อป้องกันการใช้ทรัพยากรเกิน.
- เพิ่มคำสั่งทดสอบด้วย `pytest` แล้ว หากพัฒนา feature ใหม่ควรเขียน unit test เพิ่มทุกครั้ง.

## YOLO Object Detection Workflow

Projects can now be created in a **YOLO** training mode to handle annotated object-detection datasets. Upload a ZIP archive that contains `train/images`, `train/labels`, and optional `val/…` folders (standard YOLO format). The new `/ui/yolo.html` page lets you:

- Inspect dataset health and class names after upload.
- Configure Ultralytics YOLO variants, epochs, and thresholds.
- Launch training jobs that run in the existing background queue.
- Download trained weights and run detection tests from the browser.

The backend exposes helper endpoints under `/projects/{project_id}/yolo/*` for dataset management and training, and uses the `ultralytics` package under the hood.
