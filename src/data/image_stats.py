"""Per-image statistics for EDA, cached to data/processed/image_stats.csv."""

import hashlib
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from src.data.images import fundus_mask, pad_to_square
from src.utils.paths import IMAGE_STATS_CSV

CANONICAL_SIZE = 512
N_CAMERAS = 6


def image_stats(path: Path) -> dict[str, Any]:
    """Metadata of the original, color stats inside the fundus mask, and scale-dependent
    stats (sharpness, optic disc position, 16x16 hash) on the 512 canonical image.
    """
    data = path.read_bytes()
    img = cv2.cvtColor(
        cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
    )
    h, w = img.shape[:2]
    mask = fundus_mask(img)
    k = max(4, min(h, w) // 40)
    corners = np.concatenate(
        [img[:k, :k], img[:k, -k:], img[-k:, :k], img[-k:, -k:]]
    ).reshape(-1, 3)
    x, y, bw, bh = cv2.boundingRect(mask.astype(np.uint8))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    pix, gpix = img[mask], gray[mask]
    crop = np.where(mask[..., None], img, 0)[y : y + bh, x : x + bw]
    canon = cv2.resize(
        pad_to_square(crop),
        (CANONICAL_SIZE, CANONICAL_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    g = cv2.cvtColor(canon, cv2.COLOR_RGB2GRAY)
    radius = cv2.distanceTransform(fundus_mask(canon).astype(np.uint8), cv2.DIST_L2, 5)
    inner = radius > 30
    bright = (g.astype(np.float32) + canon[..., 1]) / 2
    blob = cv2.GaussianBlur(bright, (0, 0), 8) - cv2.GaussianBlur(bright, (0, 0), 40)
    blob[radius < 0.3 * radius.max()] = -np.inf
    dy, dx = np.unravel_index(np.argmax(blob), blob.shape)
    p5, p50, p95 = np.percentile(gpix, [5, 50, 95])
    return {
        "image": path.name,
        "md5": hashlib.md5(data).hexdigest(),
        "file_kb": len(data) / 1024,
        "width": w,
        "height": h,
        "corner_mean": corners.mean(),
        "fundus_frac": mask.mean(),
        "fundus_w": bw,
        "fundus_h": bh,
        "fundus_cut": y == 0 or y + bh == h or x == 0 or x + bw == w,
        **{f"mean_{c}": v for c, v in zip("rgb", pix.mean(axis=0))},
        **{f"std_{c}": v for c, v in zip("rgb", pix.std(axis=0))},
        "gray_p5": p5,
        "gray_p50": p50,
        "gray_p95": p95,
        "contrast": gpix.std(),
        "sharpness": cv2.Laplacian(g, cv2.CV_64F)[inner].var(),
        "disc_x": dx / CANONICAL_SIZE,
        "disc_y": dy / CANONICAL_SIZE,
        "hash16": cv2.resize(g, (16, 16), interpolation=cv2.INTER_AREA).tobytes().hex(),
    }


def compute_image_stats(paths: Sequence[Path], workers: int = 8) -> pd.DataFrame:
    """image_stats over paths in threads, plus patient_id and eye from file names."""
    with ThreadPoolExecutor(workers) as pool:
        stats = pd.DataFrame(list(pool.map(image_stats, paths)))
    parsed = stats["image"].str.extract(r"^(\d+)_(left|right)\.jpg$")
    return stats.assign(patient_id=parsed[0].astype(int), eye=parsed[1])


def load_image_stats(path: Path = IMAGE_STATS_CSV) -> pd.DataFrame:
    """Read the stats csv and add size, aspect, fundus_d, background and camera.

    camera is the size string for the N_CAMERAS most common sizes of Training Images,
    "other" for the rest.
    """
    stats = pd.read_csv(path)
    stats["size"] = stats["width"].astype(str) + "x" + stats["height"].astype(str)
    stats["aspect"] = stats["width"] / stats["height"]
    stats["fundus_d"] = stats[["fundus_w", "fundus_h"]].max(axis=1)
    stats["background"] = np.where(stats["corner_mean"] > 128, "white", "black")
    train_sizes = stats.loc[stats["image_set"] == "train", "size"].value_counts()
    top = train_sizes.index[:N_CAMERAS]
    stats["camera"] = stats["size"].where(stats["size"].isin(top), "other")
    return stats
