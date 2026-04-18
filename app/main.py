from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image, UnidentifiedImageError

from app.schemas import ErrorResponse, HealthResponse, PredictResponse


MODEL_PATH = os.getenv("MODEL_PATH", "b0_v2.pth")
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(8 * 1024 * 1024)))
ALLOWED_MIME_TYPES = {"image/jpeg", "image/jpg", "image/png"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
USE_MOCK_PREDICTOR = os.getenv("USE_MOCK_PREDICTOR", "0") == "1"


class MockPredictor:
    """Pseudo predictor for local API/UI testing without loading model weights."""

    device = "cpu-mock"

    def predict(
        self,
        image: Image.Image,
        threshold: float = 0.5,
        include_wsol: bool = False,
        cam_threshold: float = 0.05,
    ) -> dict[str, Any]:
        # Simple heuristic so responses vary by input image while staying deterministic.
        grayscale = image.convert("L")
        mean_intensity = sum(grayscale.getdata()) / (grayscale.width * grayscale.height * 255.0)

        prob_safe = float(max(0.0, min(1.0, mean_intensity)))
        prob_dangerous = float(1.0 - prob_safe)
        is_safe = prob_safe > threshold
        label = "safe" if is_safe else "dangerous"
        confidence = prob_safe if is_safe else prob_dangerous

        wsol: dict[str, Any] | None = None
        if include_wsol:
            box_width = int(grayscale.width * 0.4)
            box_height = int(grayscale.height * 0.4)
            x_min = int((grayscale.width - box_width) / 2)
            y_min = int((grayscale.height - box_height) / 2)
            x_max = x_min + box_width
            y_max = y_min + box_height
            wsol = {
                "cam_threshold": float(cam_threshold),
                "bbox": {
                    "pixel": {
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max,
                        "width": box_width,
                        "height": box_height,
                    },
                    "normalized": {
                        "x": round(x_min / grayscale.width, 6),
                        "y": round(y_min / grayscale.height, 6),
                        "width": round(box_width / grayscale.width, 6),
                        "height": round(box_height / grayscale.height, 6),
                    },
                },
            }

        return {
            "label": label,
            "confidence": round(float(confidence), 6),
            "prob_safe": round(prob_safe, 6),
            "prob_dangerous": round(prob_dangerous, 6),
            "threshold": float(threshold),
            "wsol": wsol,
        }


def _error(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details,
        }
    }


def _raise_http(status_code: int, code: str, message: str, details: dict[str, Any] | None = None) -> None:
    raise HTTPException(status_code=status_code, detail=_error(code, message, details))


def _has_allowed_extension(filename: str | None) -> bool:
    if not filename:
        return False
    suffix = Path(filename).suffix.lower()
    return suffix in ALLOWED_EXTENSIONS


def _load_predictor(app: FastAPI) -> None:
    if USE_MOCK_PREDICTOR:
        app.state.predictor = MockPredictor()
        app.state.model_error = None
        return

    # Deferred import so mock mode can run without model stack initialization.
    from app.inference import Predictor

    app.state.predictor = Predictor(model_path=MODEL_PATH)
    app.state.model_error = None


def _ensure_predictor(app: FastAPI) -> Any:
    predictor = getattr(app.state, "predictor", None)
    if predictor is not None:
        return predictor

    try:
        _load_predictor(app)
        return app.state.predictor
    except Exception as exc:
        app.state.model_error = str(exc)
        _raise_http(
            status_code=503,
            code="model_not_available",
            message="Model is not available. Check model path and startup logs.",
            details={"model_path": MODEL_PATH, "reason": str(exc)},
        )


def _validate_thresholds(threshold: float, cam_threshold: float) -> None:
    if not (0.0 <= threshold <= 1.0):
        _raise_http(
            status_code=400,
            code="invalid_threshold",
            message="threshold must be between 0.0 and 1.0",
            details={"threshold": threshold},
        )

    if not (0.0 <= cam_threshold <= 1.0):
        _raise_http(
            status_code=400,
            code="invalid_cam_threshold",
            message="cam_threshold must be between 0.0 and 1.0",
            details={"cam_threshold": cam_threshold},
        )


async def _read_and_validate_image(file: UploadFile) -> Image.Image:
    mime_ok = file.content_type in ALLOWED_MIME_TYPES
    ext_ok = _has_allowed_extension(file.filename)

    if not (mime_ok or ext_ok):
        _raise_http(
            status_code=400,
            code="unsupported_file_type",
            message="Only JPG and PNG images are accepted.",
            details={
                "content_type": file.content_type,
                "filename": file.filename,
                "allowed_mime_types": sorted(ALLOWED_MIME_TYPES),
            },
        )

    content = await file.read()
    if not content:
        _raise_http(status_code=400, code="empty_file", message="Uploaded file is empty.")

    if len(content) > MAX_IMAGE_BYTES:
        _raise_http(
            status_code=400,
            code="file_too_large",
            message="Uploaded file exceeds size limit.",
            details={
                "max_bytes": MAX_IMAGE_BYTES,
                "actual_bytes": len(content),
            },
        )

    try:
        image = Image.open(BytesIO(content)).convert("RGB")
    except UnidentifiedImageError:
        _raise_http(
            status_code=400,
            code="invalid_image",
            message="Could not decode image. Please upload a valid JPG/PNG file.",
        )

    return image


app = FastAPI(
    title="Opened Manhole Classifier",
    version="1.0.0",
    description="Inference-only API with optional Grad-CAM WSOL output.",
)
app.state.predictor = None
app.state.model_error = None


@app.on_event("startup")
def startup_event() -> None:
    if os.getenv("DISABLE_STARTUP_MODEL_LOAD", "0") == "1":
        return

    try:
        _load_predictor(app)
    except Exception as exc:
        app.state.model_error = str(exc)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=_error(code="http_error", message=str(exc.detail)),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=_error(
            code="validation_error",
            message="Request validation failed.",
            details={"issues": exc.errors()},
        ),
    )


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    return """
<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Opened Manhole Classifier Demo</title>
    <style>
      body { font-family: sans-serif; margin: 2rem auto; max-width: 760px; line-height: 1.4; }
      .card { border: 1px solid #ddd; border-radius: 8px; padding: 1rem; }
      .row { margin-bottom: 0.8rem; }
      label { display: block; margin-bottom: 0.25rem; font-weight: 600; }
      input[type=\"number\"] { width: 120px; }
      button { padding: 0.6rem 1rem; cursor: pointer; }
      pre { background: #111; color: #f5f5f5; padding: 1rem; border-radius: 8px; overflow-x: auto; }
      .hint { color: #666; font-size: 0.9rem; }
    </style>
  </head>
  <body>
    <h1>Opened Manhole Classifier</h1>
    <p>Upload a JPG/PNG image for class prediction. Toggle WSOL to include Grad-CAM bounding box output.</p>
    <div class=\"card\">
      <form id=\"predict-form\">
        <div class=\"row\">
          <label for=\"image\">Image</label>
          <input id=\"image\" name=\"file\" type=\"file\" accept=\"image/jpeg,image/png\" required />
        </div>

        <div class=\"row\">
          <label for=\"threshold\">Threshold (0-1)</label>
          <input id=\"threshold\" name=\"threshold\" type=\"number\" min=\"0\" max=\"1\" step=\"0.01\" value=\"0.5\" />
        </div>

        <div class=\"row\">
          <label for=\"cam_threshold\">CAM Threshold (0-1)</label>
          <input id=\"cam_threshold\" name=\"cam_threshold\" type=\"number\" min=\"0\" max=\"1\" step=\"0.01\" value=\"0.05\" />
        </div>

        <div class=\"row\">
          <label>
            <input id=\"include_wsol\" name=\"include_wsol\" type=\"checkbox\" value=\"true\" />
            Include WSOL bounding box (Grad-CAM)
          </label>
          <div class=\"hint\">WSOL is slower on CPU.</div>
        </div>

        <button type=\"submit\">Predict</button>
      </form>
    </div>

    <h2>Response</h2>
    <pre id=\"output\">Submit an image to see prediction JSON.</pre>

    <script>
      const form = document.getElementById('predict-form');
      const output = document.getElementById('output');

      form.addEventListener('submit', async (event) => {
        event.preventDefault();
        const formData = new FormData(form);
        const include = document.getElementById('include_wsol').checked;
        if (!include) {
          formData.set('include_wsol', 'false');
        }

        output.textContent = 'Running inference...';

        try {
          const response = await fetch('/predict', {
            method: 'POST',
            body: formData,
          });

          const data = await response.json();
          output.textContent = JSON.stringify(data, null, 2);
        } catch (error) {
          output.textContent = JSON.stringify({ error: String(error) }, null, 2);
        }
      });
    </script>
  </body>
</html>
"""


@app.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": ErrorResponse}},
)
async def health() -> HealthResponse:
    predictor = getattr(app.state, "predictor", None)
    model_error = getattr(app.state, "model_error", None)

    return HealthResponse(
        status="ok" if predictor is not None else "degraded",
        model_loaded=predictor is not None,
        device=str(predictor.device) if predictor is not None else None,
        model_path=MODEL_PATH,
        model_error=model_error,
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def predict(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    include_wsol: bool = Form(False),
    cam_threshold: float = Form(0.05),
) -> PredictResponse:
    _validate_thresholds(threshold=threshold, cam_threshold=cam_threshold)
    image = await _read_and_validate_image(file)
    predictor = _ensure_predictor(app)

    try:
        result = predictor.predict(
            image=image,
            threshold=threshold,
            include_wsol=include_wsol,
            cam_threshold=cam_threshold,
        )
    except HTTPException:
        raise
    except Exception as exc:
        _raise_http(
            status_code=500,
            code="inference_failed",
            message="Inference failed unexpectedly.",
            details={"reason": str(exc)},
        )

    return PredictResponse(**result)
