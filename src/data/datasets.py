"""Torch datasets. Every item is a dict with at least "x" and "labels"."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.images import load_image, to_canonical
from src.data.labels import LABELS
from src.utils.paths import TRAIN_IMAGES_DIR

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
EYE_LABELS: list[str] = [f"eye_{label}" for label in LABELS]


class FundusDataset(Dataset):
    """Canonical (crop, pad, resize), ImageNet-normalized fundus images from ``load_eyes``.

    Item: {"x": (3, size, size) float, "labels": float, "patient_id": int, "eye": str}.
    label_cols defaults to the eye labels EYE_LABELS; pass LABELS for patient labels.
    """

    def __init__(
        self,
        eyes_df: pd.DataFrame,
        size: int = 224,
        label_cols: Sequence[str] | None = None,
        image_dir: Path = TRAIN_IMAGES_DIR,
    ) -> None:
        self.eyes = eyes_df.reset_index(drop=True)
        cols = EYE_LABELS if label_cols is None else list(label_cols)
        self.labels = torch.tensor(self.eyes[cols].to_numpy(), dtype=torch.float32)
        self.size = size
        self.image_dir = image_dir

    def __len__(self) -> int:
        return len(self.eyes)

    def __getitem__(self, i: int) -> dict:
        row = self.eyes.iloc[i]
        img = to_canonical(load_image(self.image_dir / row["image"]), self.size)
        x = torch.from_numpy(img).permute(2, 0, 1) / 255
        return {
            "x": (x - IMAGENET_MEAN) / IMAGENET_STD,
            "labels": self.labels[i],
            "patient_id": int(row["patient_id"]),
            "eye": row["eye"],
        }


class EmbeddingDataset(Dataset):
    """Precomputed feature vectors. Item: {"x": float, "labels": float}."""

    def __init__(self, x: np.ndarray, labels: np.ndarray) -> None:
        self.x = torch.tensor(x, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> dict:
        return {"x": self.x[i], "labels": self.labels[i]}
