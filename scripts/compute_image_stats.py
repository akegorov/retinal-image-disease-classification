"""Compute per-image statistics of Training and Testing Images for EDA.

Writes data/processed/image_stats.csv (one row per image, image_set = train / test);
--limit N takes the first N images of each folder and writes image_stats_limitN.csv.
About 4-5 minutes for 8000 images with 8 workers.

Usage: uv run python scripts/compute_image_stats.py [--limit 50] [--workers 8]
"""

import argparse
import time

import pandas as pd

from src.data.image_stats import compute_image_stats
from src.utils.paths import IMAGE_STATS_CSV, TEST_IMAGES_DIR, TRAIN_IMAGES_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-image statistics.")
    parser.add_argument("--limit", type=int, help="first N images of each folder")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    start = time.perf_counter()
    parts = []
    for image_set, folder in [("train", TRAIN_IMAGES_DIR), ("test", TEST_IMAGES_DIR)]:
        paths = sorted(folder.glob("*.jpg"))[: args.limit]
        stats = compute_image_stats(paths, args.workers)
        parts.append(stats.assign(image_set=image_set))
    stats = pd.concat(parts, ignore_index=True)
    stats = stats[
        [c for c in stats.columns if c not in ("patient_id", "eye")]
        + ["patient_id", "eye"]
    ]

    out = IMAGE_STATS_CSV
    if args.limit:
        out = out.with_name(f"{out.stem}_limit{args.limit}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(out, index=False)
    elapsed = time.perf_counter() - start
    print(f"{len(stats)} images in {elapsed:.1f} s -> {out}")


if __name__ == "__main__":
    main()
