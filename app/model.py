from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping

import torch
import torch.nn as nn
import torchvision.models as models


MODEL_INPUT_SIZE = 224


class SimpleClassifier(nn.Module):
    """Notebook-matching EfficientNet-B0 classifier head (1 logit)."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1280, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MyModel(nn.Module):
    """Notebook-matching inference model architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.pre_trained_model = models.efficientnet_b0(weights=None)
        self.pre_trained_model.classifier = SimpleClassifier()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pre_trained_model(x)


def _extract_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    """Support common checkpoint structures from notebook exports."""
    if isinstance(checkpoint, Mapping):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], Mapping):
            return checkpoint["state_dict"]
        if "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], Mapping):
            return checkpoint["model_state_dict"]
    if isinstance(checkpoint, Mapping):
        return checkpoint
    raise TypeError("Unsupported checkpoint format: expected a mapping")


def _strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Handle DataParallel checkpoints that prefix keys with 'module.'."""
    keys = list(state_dict.keys())
    if keys and all(k.startswith("module.") for k in keys):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return dict(state_dict)


def load_model(model_path: str | Path, device: torch.device) -> MyModel:
    """Load model weights from disk and return an eval-mode model on device."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    model = MyModel()
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state_dict = _strip_module_prefix(_extract_state_dict(checkpoint))
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()
    return model
