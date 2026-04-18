from __future__ import annotations

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
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
            wsol = {
                "cam_threshold": cam_threshold,
                "bbox": {
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
                },
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
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("DISABLE_STARTUP_MODEL_LOAD", "1")
    app.state.predictor = StubPredictor()
    app.state.model_error = None

    with TestClient(app) as test_client:
        yield test_client


def _make_image_bytes(image_format: str = "PNG") -> bytes:
    image = Image.new("RGB", (224, 224), color=(255, 20, 20))
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True


def test_predict_class_only(client: TestClient) -> None:
    response = client.post(
        "/predict",
        files={"file": ("sample.png", _make_image_bytes("PNG"), "image/png")},
        data={"threshold": "0.5", "include_wsol": "false", "cam_threshold": "0.05"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] == "safe"
    assert payload["wsol"] is None


def test_predict_with_wsol(client: TestClient) -> None:
    response = client.post(
        "/predict",
        files={"file": ("sample.jpg", _make_image_bytes("JPEG"), "image/jpeg")},
        data={"threshold": "0.5", "include_wsol": "true", "cam_threshold": "0.1"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["wsol"] is not None
    assert payload["wsol"]["bbox"]["pixel"]["x_min"] == 10


def test_predict_rejects_invalid_file_type(client: TestClient) -> None:
    response = client.post(
        "/predict",
        files={"file": ("sample.txt", b"not an image", "text/plain")},
        data={"threshold": "0.5", "include_wsol": "false", "cam_threshold": "0.05"},
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["code"] == "unsupported_file_type"


def test_predict_rejects_invalid_threshold(client: TestClient) -> None:
    response = client.post(
        "/predict",
        files={"file": ("sample.png", _make_image_bytes("PNG"), "image/png")},
        data={"threshold": "1.2", "include_wsol": "false", "cam_threshold": "0.05"},
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["code"] == "invalid_threshold"
