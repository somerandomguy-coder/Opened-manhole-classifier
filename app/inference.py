from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

from app.model import MyModel, load_model
from app.preprocess import preprocess_image


@dataclass(frozen=True)
class ClassificationResult:
    label: str
    confidence: float
    prob_safe: float
    prob_dangerous: float
    threshold: float
    predicted_class: int


def classify_probability(prob_safe: float, threshold: float) -> ClassificationResult:
    """Map sigmoid output to public labels and confidence."""
    predicted_safe = prob_safe > threshold
    prob_dangerous = 1.0 - prob_safe

    if predicted_safe:
        label = "safe"
        confidence = prob_safe
        predicted_class = 1
    else:
        label = "dangerous"
        confidence = prob_dangerous
        predicted_class = 0

    return ClassificationResult(
        label=label,
        confidence=float(confidence),
        prob_safe=float(prob_safe),
        prob_dangerous=float(prob_dangerous),
        threshold=float(threshold),
        predicted_class=predicted_class,
    )


def cam_to_bounding_box(
    grayscale_cam: np.ndarray,
    threshold: float,
    original_width: int,
    original_height: int,
) -> dict[str, Any] | None:
    """Convert Grad-CAM heatmap into pixel + normalized bbox. Returns None if no contour."""
    cam_resized = cv2.resize(grayscale_cam, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
    binary_mask = (cam_resized > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    x_min, y_min, width, height = cv2.boundingRect(largest_contour)
    x_max = x_min + width
    y_max = y_min + height

    return {
        "pixel": {
            "x_min": int(x_min),
            "y_min": int(y_min),
            "x_max": int(x_max),
            "y_max": int(y_max),
            "width": int(width),
            "height": int(height),
        },
        "normalized": {
            "x": round(x_min / original_width, 6),
            "y": round(y_min / original_height, 6),
            "width": round(width / original_width, 6),
            "height": round(height / original_height, 6),
        },
    }


class Predictor:
    """Inference wrapper for class prediction + optional Grad-CAM WSOL."""

    def __init__(self, model_path: str | Path, device: str | None = None) -> None:
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: MyModel = load_model(model_path=model_path, device=self.device)
        self.sigmoid = nn.Sigmoid()
        self.target_layers = [self.model.pre_trained_model.features[-1]]
        self._cam_lock = Lock()

    def predict(
        self,
        image: Image.Image,
        threshold: float = 0.5,
        include_wsol: bool = False,
        cam_threshold: float = 0.05,
    ) -> dict[str, Any]:
        """Run prediction on a single PIL image."""
        rgb_image = image.convert("RGB")
        original_width, original_height = rgb_image.size
        input_tensor = preprocess_image(rgb_image).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor).squeeze()
            prob_safe = float(self.sigmoid(logits).item())

        classification = classify_probability(prob_safe=prob_safe, threshold=threshold)

        wsol: dict[str, Any] | None = None
        if include_wsol:
            targets = [BinaryClassifierOutputTarget(classification.predicted_class)]
            with self._cam_lock:
                with GradCAM(model=self.model, target_layers=self.target_layers) as cam:
                    grayscale_cams = cam(input_tensor=input_tensor, targets=targets)

            bbox = cam_to_bounding_box(
                grayscale_cam=grayscale_cams[0],
                threshold=cam_threshold,
                original_width=original_width,
                original_height=original_height,
            )

            wsol = {
                "cam_threshold": float(cam_threshold),
                "bbox": bbox,
            }

        return {
            "label": classification.label,
            "confidence": round(classification.confidence, 6),
            "prob_safe": round(classification.prob_safe, 6),
            "prob_dangerous": round(classification.prob_dangerous, 6),
            "threshold": float(classification.threshold),
            "wsol": wsol,
        }
