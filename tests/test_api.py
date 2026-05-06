from __future__ import annotations

from io import BytesIO

import httpx
import pytest
from PIL import Image

from app.main import app


class StubPredictor:
    device = "cpu"

    def predict(
        self,
        image: Image.Image,
        threshold: float = 0.5,
        include_wsol: bool = False,
        cam_threshold: float = 0.05,
    ) -> dict:
        wsol = None
        if include_wsol:
            bbox = None if cam_threshold >= 0.95 else {
                "pixel": {
                    "x_min": 10,
                    "y_min": 20,
                    "x_max": 100,
                    "y_max": 120,
                    "width": 90,
                    "height": 100,
                },
                "normalized": {
                    "x": 0.044643,
                    "y": 0.089286,
                    "width": 0.401786,
                    "height": 0.446429,
                },
            }
            wsol = {
                "cam_threshold": cam_threshold,
                "bbox": bbox,
            }

        return {
            "label": "safe",
            "confidence": 0.91,
            "prob_safe": 0.91,
            "prob_dangerous": 0.09,
            "threshold": threshold,
            "wsol": wsol,
        }


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def client(monkeypatch: pytest.MonkeyPatch) -> httpx.AsyncClient:
    monkeypatch.setenv("DISABLE_STARTUP_MODEL_LOAD", "1")
    app.state.predictor = StubPredictor()
    app.state.model_error = None

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as test_client:
        yield test_client


def _make_image_bytes(image_format: str = "PNG") -> bytes:
    image = Image.new("RGB", (224, 224), color=(255, 20, 20))
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


@pytest.mark.anyio
async def test_health_returns_ok(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True


@pytest.mark.anyio
async def test_root_serves_frontend_html(client: httpx.AsyncClient) -> None:
    response = await client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Opened Manhole Classifier" in response.text
    assert "id=\"predictForm\"" in response.text


@pytest.mark.anyio
async def test_predict_class_only(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/predict",
        files={"file": ("sample.png", _make_image_bytes("PNG"), "image/png")},
        data={"threshold": "0.5", "include_wsol": "false", "cam_threshold": "0.05"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {
        "label",
        "confidence",
        "prob_safe",
        "prob_dangerous",
        "threshold",
        "wsol",
    }
    assert payload["label"] == "safe"
    assert payload["confidence"] == 0.91
    assert payload["prob_safe"] == 0.91
    assert payload["prob_dangerous"] == 0.09
    assert payload["threshold"] == 0.5
    assert payload["wsol"] is None


@pytest.mark.anyio
async def test_predict_with_wsol(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/predict",
        files={"file": ("sample.jpg", _make_image_bytes("JPEG"), "image/jpeg")},
        data={"threshold": "0.5", "include_wsol": "true", "cam_threshold": "0.1"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["wsol"] is not None
    assert payload["wsol"]["cam_threshold"] == 0.1
    assert payload["wsol"]["bbox"]["pixel"]["x_min"] == 10


@pytest.mark.anyio
async def test_predict_with_wsol_can_return_null_bbox(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/predict",
        files={"file": ("sample.jpg", _make_image_bytes("JPEG"), "image/jpeg")},
        data={"threshold": "0.5", "include_wsol": "true", "cam_threshold": "0.95"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["wsol"] is not None
    assert payload["wsol"]["bbox"] is None


@pytest.mark.anyio
async def test_predict_rejects_invalid_file_type(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/predict",
        files={"file": ("sample.txt", b"not an image", "text/plain")},
        data={"threshold": "0.5", "include_wsol": "false", "cam_threshold": "0.05"},
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["code"] == "unsupported_file_type"


@pytest.mark.anyio
async def test_predict_rejects_invalid_threshold(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/predict",
        files={"file": ("sample.png", _make_image_bytes("PNG"), "image/png")},
        data={"threshold": "1.2", "include_wsol": "false", "cam_threshold": "0.05"},
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["code"] == "invalid_threshold"


@pytest.mark.anyio
async def test_predict_rejects_invalid_cam_threshold(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/predict",
        files={"file": ("sample.png", _make_image_bytes("PNG"), "image/png")},
        data={"threshold": "0.5", "include_wsol": "true", "cam_threshold": "1.2"},
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["code"] == "invalid_cam_threshold"
