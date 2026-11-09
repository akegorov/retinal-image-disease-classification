"""ODIR-5K labels from diagnostic keywords.

The organizers' patient labels follow exactly from the per-eye keywords with this rule
(checked on all 3500 patients in eda/intro.ipynb, section 5).
"""

import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from src.utils.paths import ANNOTATIONS_XLSX

LABELS: list[str] = ["N", "D", "G", "C", "A", "H", "M", "O"]

QUALITY_KEYWORDS: frozenset[str] = frozenset(
    {
        "lens dust",
        "low image quality",
        "image offset",
        "no fundus image",
        "optic disk photographically invisible",
        "anterior segment image",
    }
)

_RULES: list[tuple[str, str]] = [
    ("^normal fundus$", "N"),
    ("hypertensive", "H"),
    ("myopi", "M"),
    ("glaucoma", "G"),
    ("cataract", "C"),
    ("age-related macular degeneration", "A"),
    ("(non ?proliferative|proliferative diabetic|diabetic) retinopathy", "D"),
]


def split_keywords(cell: object) -> list[str]:
    """Split a keywords cell on the full-width comma (U+FF0C) or ","; drop empties."""
    if not isinstance(cell, str):
        return []
    return [kw.strip() for kw in re.split("[，,]", cell) if kw.strip()]


def keyword_to_label(kw: str) -> str | None:
    """Class of one keyword, or None for image-quality keywords."""
    if kw in QUALITY_KEYWORDS:
        return None
    return next((label for pattern, label in _RULES if re.search(pattern, kw)), "O")


def _labels(diseases: set[str]) -> dict[str, int]:
    """Label dict from a set of disease classes; N = 1 only if there are none."""
    return {"N": int(not diseases)} | {
        label: int(label in diseases) for label in LABELS[1:]
    }


def eye_labels(keywords: str | Iterable[str]) -> dict[str, int]:
    """Eye labels from a raw keywords cell or a list of keywords."""
    if isinstance(keywords, str):
        keywords = split_keywords(keywords)
    return _labels({keyword_to_label(kw) for kw in keywords} - {None, "N"})


def patient_labels(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    """Union of the two eyes' diseases; N = 1 only if neither eye has one."""
    return _labels({label for label in LABELS[1:] if left[label] or right[label]})


def load_eyes(path: Path = ANNOTATIONS_XLSX) -> pd.DataFrame:
    """One row per eye from the ODIR-5K annotations, sorted by (patient_id, eye).

    Patient labels ``N..O`` come from ``data.xlsx``, eye labels ``eye_N..eye_O`` from
    the eye's keywords. Full image path: ``TRAIN_IMAGES_DIR / image``.
    """
    df = pd.read_excel(path)
    eyes = pd.concat(
        [
            pd.DataFrame(
                {
                    "patient_id": df["ID"],
                    "eye": eye,
                    "image": df[f"{side}-Fundus"],
                    "age": df["Patient Age"],
                    "sex": df["Patient Sex"],
                    "keywords": df[f"{side}-Diagnostic Keywords"].map(split_keywords),
                }
            ).join(df[LABELS])
            for eye, side in [("left", "Left"), ("right", "Right")]
        ],
        ignore_index=True,
    )
    eye_cols = pd.DataFrame(eyes["keywords"].map(eye_labels).tolist())[LABELS]
    eyes = eyes.join(eye_cols.add_prefix("eye_"))
    return eyes.sort_values(["patient_id", "eye"], ignore_index=True)
