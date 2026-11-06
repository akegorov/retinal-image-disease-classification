"""ODIR-5K labels from diagnostic keywords.

The organizers' patient labels follow exactly from the per-eye keywords with this rule
(checked on all 3500 patients in eda/intro.ipynb, section 5).
"""

import re
from collections.abc import Iterable

LABELS: list[str] = ["N", "D", "G", "C", "A", "H", "M", "O"]

# Image-quality remarks, not diagnoses.
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

# (regex, label), first match wins. H and M come before D: "hypertensive retinopathy"
# and "myopic retinopathy" also contain "retinopathy". No match -> O.
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
