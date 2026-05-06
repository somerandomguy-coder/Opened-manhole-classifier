---
title: Opened Manhole Classifier
sdk: docker
app_port: 7860
---

# Opened Manhole Classifier (Inference-Only)

This repository is now deployment-ready and inference-only.

It serves a fine-tuned EfficientNet-B0 model with:

- `POST /predict` JSON inference API
- `GET /` visual browser inference showcase UI
- Optional WSOL output (Grad-CAM bounding box)

## Label Mapping

Binary output mapping is locked as:

- positive class (`sigmoid > threshold`): `safe` (closed manhole)
- negative class: `dangerous` (open/damaged)

## Runtime Components

- `app/model.py`: model architecture + checkpoint loading (`b0_v2.pth`)
- `app/preprocess.py`: notebook-matching preprocessing (`224x224`, normalize mean/std `0.5`)
- `app/inference.py`: prediction logic + optional Grad-CAM WSOL bbox
- `app/main.py`: FastAPI app (`/`, `/health`, `/predict`)
- `app/static/`: frontend assets (`index.html`, `styles.css`, `app.js`)

Legacy notebooks are preserved at `notebooks/legacy/` and are excluded from runtime flow.

## Local Run

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start API server

#### Option A: Mock mode (fastest, no real inference)

Use this to verify UI/API behavior on any laptop:

```bash
USE_MOCK_PREDICTOR=1 uvicorn app.main:app --host 0.0.0.0 --port 7860
```

#### Option B: Real model on CPU

```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

Then open:

- UI: `http://localhost:7860/`
- Health: `http://localhost:7860/health`

If `/health` returns `"model_loaded": false`, the model file path is likely wrong.

## UI Demo Flow

1. Open `http://localhost:7860/`.
2. Upload an image file (`jpg`, `jpeg`, or `png`; `webp` can be selected in UI preview but API validation is JPG/PNG).
3. Adjust `Decision threshold` if needed.
4. Optionally enable `Show WSOL / Grad-CAM bbox` and tune `CAM threshold`.
5. Click `Predict` and wait for inference.
6. Review label, confidence, safe/danger probabilities, and threshold.
7. Open `Raw response` only for debugging details.

### 3. Example API calls

Class-only prediction:

```bash
curl -X POST http://localhost:7860/predict \
  -F "file=@/path/to/image.jpg" \
  -F "threshold=0.5" \
  -F "include_wsol=false" \
  -F "cam_threshold=0.05"
```

Prediction with WSOL:

```bash
curl -X POST http://localhost:7860/predict \
  -F "file=@/path/to/image.jpg" \
  -F "threshold=0.5" \
  -F "include_wsol=true" \
  -F "cam_threshold=0.05"
```

## API Contract

### `GET /health`

Returns model/service status.

### `POST /predict`

`multipart/form-data` fields:

- `file`: required image (`jpg`, `jpeg`, `png`)
- `threshold`: optional float in `[0, 1]` (default `0.5`)
- `include_wsol`: optional bool (default `false`)
- `cam_threshold`: optional float in `[0, 1]` (default `0.05`)

Response:

```json
{
  "label": "safe",
  "confidence": 0.91,
  "prob_safe": 0.91,
  "prob_dangerous": 0.09,
  "threshold": 0.5,
  "wsol": {
    "cam_threshold": 0.05,
    "bbox": {
      "pixel": {
        "x_min": 10,
        "y_min": 20,
        "x_max": 100,
        "y_max": 120,
        "width": 90,
        "height": 100
      },
      "normalized": {
        "x": 0.04,
        "y": 0.09,
        "width": 0.40,
        "height": 0.45
      }
    }
  }
}
```

When `include_wsol=false`, `wsol` is `null`.

When `include_wsol=true`, `wsol.bbox` may be `null` if no CAM region passes the selected CAM threshold.

Error responses use a structured format:

```json
{
  "error": {
    "code": "invalid_threshold",
    "message": "threshold must be between 0.0 and 1.0",
    "details": {
      "threshold": 1.2
    }
  }
}
```

## Hugging Face Spaces Deployment (CPU First)

This repo includes Docker Space front-matter and a `Dockerfile`.

1. Create a new Hugging Face Space with SDK set to **Docker**.
2. Push this repository to that Space.
3. Keep `b0_v2.pth` in repo root (or set `MODEL_PATH` env var).
4. Space will build and expose the app on port `7860`.

Free CPU hardware is enough for class prediction demos. WSOL is available but slower on CPU.

## Optional Railway Deployment

Railway can also run this Dockerized FastAPI app. For demo usage, Hugging Face Spaces CPU is usually the lowest-cost path.

## Environment Variables

- `MODEL_PATH` (default: `b0_v2.pth`)
- `MAX_IMAGE_BYTES` (default: `8388608`, i.e. 8 MB)
- `DISABLE_STARTUP_MODEL_LOAD` (default: `0`, useful in tests)
- `USE_MOCK_PREDICTOR` (default: `0`; set `1` for pseudo/local testing mode)

## Local Docker Run

Build:

```bash
docker build -t opened-manhole-classifier:local .
```

Run real model:

```bash
docker run --rm -p 7860:7860 opened-manhole-classifier:local
```

Run mock mode:

```bash
docker run --rm -p 7860:7860 -e USE_MOCK_PREDICTOR=1 opened-manhole-classifier:local
```

## Tests

Run:

```bash
pytest -q
```

Test coverage includes:

- preprocessing shape/range
- label mapping behavior
- CAM bbox fallback with no contour
- API health/predict/error behavior
- root route HTML frontend serving
- WSOL bbox-present and bbox-null API cases

## Manual UI Acceptance Checklist

1. Selecting an image immediately shows preview and file metadata.
2. Predict button disables and shows loading text during inference.
3. Non-WSOL prediction shows label, confidence, probabilities, threshold, and image.
4. WSOL prediction shows canvas overlay with bbox when bbox is returned.
5. WSOL with null bbox shows a clear threshold guidance message.
6. Raw JSON remains available only in collapsed `<details>`.
7. UI behavior is valid in both mock mode and real model mode.
8. Layout remains readable on mobile-width screens.

## Showcase References

- Suggested screenshots directory: `docs/screenshots/` (for upload preview, non-WSOL result, WSOL overlay, and mobile view captures).
- Model metrics reference location: `results/README.md` (add the latest exported evaluation summary used for demos).
- Deployment hardening note: dependency versions are pinned in `requirements.txt`; review and refresh pins before each release cut.
