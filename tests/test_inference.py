from __future__ import annotations

import numpy as np
import pytest

from app.inference import cam_to_bounding_box, classify_probability


def test_classification_mapping_safe() -> None:
    result = classify_probability(prob_safe=0.9, threshold=0.5)

    assert result.label == "safe"
    assert result.confidence == pytest.approx(0.9)
    assert result.predicted_class == 1


def test_classification_mapping_dangerous() -> None:
    result = classify_probability(prob_safe=0.2, threshold=0.5)

    assert result.label == "dangerous"
    assert result.confidence == pytest.approx(0.8)
    assert result.predicted_class == 0


def test_cam_to_bounding_box_returns_none_when_no_contours() -> None:
    empty_cam = np.zeros((224, 224), dtype=np.float32)

    bbox = cam_to_bounding_box(
        grayscale_cam=empty_cam,
        threshold=0.5,
        original_width=224,
        original_height=224,
    )

    assert bbox is None
