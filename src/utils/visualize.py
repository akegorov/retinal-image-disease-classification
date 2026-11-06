"""Plotting helpers shared by notebooks."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from src.data.images import load_image
from src.utils.paths import FIGURES_DIR


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
