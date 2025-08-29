# Anomaly Detection Backend (MVP)

FastAPI backend that ingests **normal** images, trains a simple **autoencoder** using PyTorch,
tracks a background **job**, and lets you download the resulting `model.pt`.

## Endpoints
- `POST /datasets/{scene_id}/upload`  — upload multiple images (field name: `files`)
- `POST /train` — start training job, returns `{ job_id }`
- `GET /jobs/{job_id}` — poll job status `{ status, progress, model_id? }`
- `GET /models/{model_id}/download` — download `model.pt`

> ROI endpoint is **optional** and not used by default, but stubbed as `/scenes/{scene_id}/roi` to avoid 404 from the sample frontend.

## Quick start
```bash
python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Install torch per your OS: https://pytorch.org/get-started/locally/  (or the CPU index-url hint in requirements.txt)
uvicorn app.main:app --reload --port 8000
```

Place uploads will be stored under `data/datasets/{scene_id}/raw/`.

Training will produce `data/models/{model_id}/model.pt` + configs.
