from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from app.preprocess import preprocess_image


def test_preprocess_shape_and_range() -> None:
    image_array = np.full((320, 480, 3), 255, dtype=np.uint8)
    image = Image.fromarray(image_array, mode="RGB")

    tensor = preprocess_image(image)

    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32
    assert tensor.min().item() >= -1.01
    assert tensor.max().item() <= 1.01
