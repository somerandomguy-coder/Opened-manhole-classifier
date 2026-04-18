from __future__ import annotations

from typing import Tuple

import torch
from PIL import Image
from torchvision import transforms

from app.model import MODEL_INPUT_SIZE


NORMALIZE_MEAN: Tuple[float, float, float] = (0.5, 0.5, 0.5)
NORMALIZE_STD: Tuple[float, float, float] = (0.5, 0.5, 0.5)


def build_preprocess() -> transforms.Compose:
    """Build the notebook-matching preprocessing pipeline."""
    return transforms.Compose(
        [
            transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Convert PIL image to normalized batch tensor [1, 3, 224, 224]."""
    transform = build_preprocess()
    return transform(image).unsqueeze(0)
