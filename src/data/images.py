"""Fundus image loading, background removal and canonical resizing."""

from pathlib import Path

import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    """Read an image as an RGB uint8 array of shape (H, W, 3)."""
    # Decode bytes instead of passing a Unicode path to OpenCV (Windows).
    data = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
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


def pad_to_square(img: np.ndarray) -> np.ndarray:
    """Pad with black to a square, keeping the image centered."""
    h, w = img.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    return cv2.copyMakeBorder(
        img, top, side - h - top, left, side - w - left, cv2.BORDER_CONSTANT, value=0
    )


def to_canonical(img: np.ndarray, size: int) -> np.ndarray:
    """crop_to_fundus -> pad_to_square -> resize to (size, size) with INTER_AREA."""
    square = pad_to_square(crop_to_fundus(img))
    return cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)


def load_canonical(path: str | Path, size: int = 256) -> np.ndarray:
    """load_image -> to_canonical: an RGB (size, size, 3) image for display."""
    return to_canonical(load_image(path), size)
