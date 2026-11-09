"""Patient-level train / val / test splits, saved as data/splits/{name}.csv."""

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from src.data.labels import LABELS
from src.utils.paths import SPLITS_DIR


def _holdout(y: np.ndarray, size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices (rest, holdout) of one multilabel-stratified split of label matrix y."""
    splitter = MultilabelStratifiedShuffleSplit(1, test_size=size, random_state=seed)
    return next(splitter.split(y, y))


def make_patient_split(
    eyes: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15, seed: int = 42
) -> pd.DataFrame:
    """Split patients into train / val / test, stratified on patient labels N..O.

    Sizes are fractions of all patients. Returns (patient_id, split), one row per patient,
    so both eyes of a patient always share a split.
    """
    patients = eyes.drop_duplicates("patient_id").sort_values("patient_id")
    y = patients[LABELS].to_numpy()
    rest, test = _holdout(y, test_size, seed)
    _, val = _holdout(y[rest], val_size / (1 - test_size), seed)

    split = np.full(len(y), "train", dtype=object)
    split[rest[val]] = "val"
    split[test] = "test"
    return pd.DataFrame(
        {"patient_id": patients["patient_id"].to_numpy(), "split": split}
    )


def save_split(split: pd.DataFrame, name: str) -> None:
    """Save (patient_id, split) to data/splits/{name}.csv."""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    split.to_csv(SPLITS_DIR / f"{name}.csv", index=False)


def load_split(name: str) -> pd.DataFrame:
    """Load (patient_id, split) from data/splits/{name}.csv."""
    return pd.read_csv(SPLITS_DIR / f"{name}.csv")
