# model.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Model definition
# -----------------------------
class EmotionCNN(nn.Module):
    """
    Convolutional network for 48x48 grayscale facial emotion recognition.
    Input:  N x 1 x 48 x 48
    Output: N x num_classes (default: 7)
    """
    def __init__(self, num_classes: int = 7):
        super().__init__()
        # Block 1
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.25)

        # Block 2
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(0.25)

        # Block 3
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout(0.50)

        # For 48x48 -> 24x24 -> 12x12 -> 6x6
        self.fc1 = nn.Linear(512 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.drop1(x)

        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.drop2(x)

        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.drop3(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# -----------------------------
# Utilities
# -----------------------------
def _default_model_path() -> Path:
    """
    Resolve the path to the trained weights.
    The repo layout (per your screenshot) puts assets alongside model.py/app.py:
      .../happiness calander/
          app.py
          model.py
          assets/emotion_cnn.pth
    """
    base = Path(__file__).resolve().parent
    return base / "assets" / "emotion_cnn.pth"


def _clean_state_dict(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize various checkpoint formats to a bare state_dict compatible with EmotionCNN.
    Handles:
      - raw state_dict
      - {'state_dict': ...}
      - keys starting with 'module.' (from DataParallel)
    """
    # Pull nested state_dict if present
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    # Strip 'module.' prefixes
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    return state


def load_trained_model(
    weights_path: str | os.PathLike | None = None,
    num_classes: int = 7,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> EmotionCNN:
    """
    Create an EmotionCNN and load trained weights.

    Parameters
    ----------
    weights_path : path to .pth file. If None, uses assets/emotion_cnn.pth next to this file.
    num_classes  : number of output classes
    map_location : where to load the checkpoint (default 'cpu' for Streamlit)
    strict       : strict loading of state dict

    Returns
    -------
    model : EmotionCNN with weights loaded
    """
    path = Path(weights_path) if weights_path is not None else _default_model_path()
    if not path.exists():
        raise FileNotFoundError(f"❗ Model file not found at {path}")

    # Load checkpoint (supports both raw state_dict and full checkpoint dicts)
    checkpoint = torch.load(path, map_location=map_location)
    if isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
        # Likely a state_dict or a wrapped dict
        state = _clean_state_dict(checkpoint)
    else:
        # Fallback—if someone saved the whole nn.Module (not recommended)
        # we'll try to get its state_dict.
        try:
            state = checkpoint.state_dict()  # type: ignore[attr-defined]
        except Exception as e:
            raise RuntimeError(
                "Unsupported checkpoint format. Expected a state_dict or a dict containing 'state_dict'."
            ) from e

    model = EmotionCNN(num_classes=num_classes)
    model.load_state_dict(state, strict=strict)
    model.eval()
    return model


def get_device() -> torch.device:
    """Helper to pick CUDA if available else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


__all__ = ["EmotionCNN", "load_trained_model", "get_device"]
