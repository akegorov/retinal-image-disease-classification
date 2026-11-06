"""Plotting helpers shared by notebooks."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.data.images import crop_to_fundus, load_image
from src.data.labels import LABELS
from src.utils.paths import FIGURES_DIR, TRAIN_IMAGES_DIR


def show_images(
    images_or_paths: Sequence[np.ndarray | str | Path],
    titles: Sequence[str] | None = None,
    ncols: int = 4,
) -> Figure:
    """Show RGB arrays or image files in a grid and return the figure."""
    ncols = min(ncols, len(images_or_paths))
    nrows = -(-len(images_or_paths) // ncols)  # ceil division
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows), squeeze=False
    )
    for ax in axes.flat:
        ax.axis("off")
    for i, (ax, img) in enumerate(zip(axes.flat, images_or_paths)):
        ax.imshow(img if isinstance(img, np.ndarray) else load_image(img))
        if titles:
            ax.set_title(titles[i], fontsize=9)
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, name: str) -> Path:
    """Save a figure to reports/figures/{name}.png and return the path."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def show_patient(eyes: pd.DataFrame, patient_id: int) -> Figure:
    """Show both cropped eyes of a patient with their keywords and the patient labels.

    ``eyes`` is the one-row-per-eye table built in eda/intro.ipynb.
    """
    rows = eyes[eyes["patient_id"] == patient_id].sort_values("eye")
    if rows.empty:
        raise ValueError(f"No patient with ID {patient_id}")
    images = [crop_to_fundus(load_image(TRAIN_IMAGES_DIR / f)) for f in rows["image"]]
    titles = [f"{e}: {', '.join(kw)}" for e, kw in zip(rows["eye"], rows["keywords"])]
    labels = [label for label in LABELS if rows[label].iloc[0]]
    fig = show_images(images, titles, ncols=2)
    fig.suptitle(f"patient {patient_id}: {', '.join(labels)}", y=1.02)
    return fig
