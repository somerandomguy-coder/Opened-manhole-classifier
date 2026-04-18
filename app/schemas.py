from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorPayload(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    error: ErrorPayload


class BBoxPixel(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    width: int
    height: int


class BBoxNormalized(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(ge=0.0, le=1.0)
    height: float = Field(ge=0.0, le=1.0)


class WSOLBBox(BaseModel):
    pixel: BBoxPixel
    normalized: BBoxNormalized


class WSOLResult(BaseModel):
    cam_threshold: float = Field(ge=0.0, le=1.0)
    bbox: WSOLBBox | None


class PredictResponse(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    prob_safe: float = Field(ge=0.0, le=1.0)
    prob_dangerous: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    wsol: WSOLResult | None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str | None
    model_path: str
    model_error: str | None = None
