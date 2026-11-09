"""Fundus image loading and background removal."""

from pathlib import Path

import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    """Read an image as an RGB uint8 array of shape (H, W, 3)."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def fundus_mask(img: np.ndarray, low: int = 10, high: int = 240) -> np.ndarray:
    """Boolean (H, W) mask of the fundus disc.

    A pixel is a fundus candidate if it is neither black background (max channel > low)
    nor white background (min channel < high). Only the largest connected region is kept
    and filled, so bright spots inside the disc (optic disc, glare) stay in the mask.
    """
    candidate = ((img.max(axis=2) > low) & (img.min(axis=2) < high)).astype(np.uint8)
    contours, _ = cv2.findContours(
        candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    mask = np.zeros_like(candidate)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, color=1, thickness=cv2.FILLED)
    return mask.astype(bool)


def crop_to_fundus(
    img: np.ndarray, low: int = 10, high: int = 240, fill_background: bool = True
) -> np.ndarray:
    """Crop to the bounding box of the fundus mask.

    If fill_background is True, pixels outside the mask become black, so black, white or any
    other background ends up the same. Returns the image unchanged if the mask is empty.
    """
    mask = fundus_mask(img, low, high)
    if not mask.any():
        return img
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    crop = img[y : y + h, x : x + w]
    if fill_background:
        crop = np.where(mask[y : y + h, x : x + w, None], crop, 0).astype(img.dtype)
    return crop
