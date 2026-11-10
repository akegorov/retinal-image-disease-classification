"""Combine per-eye predictions into patient predictions."""

import numpy as np
import pandas as pd

from src.data.labels import LABELS


def eyes_to_patient(probs: np.ndarray, eyes_index: pd.DataFrame) -> pd.DataFrame:
    """Patient probabilities from per-eye ones: max over the eyes for D..O, min for N.

    Row i of ``eyes_index`` (with a patient_id column) belongs to probs[i].
    """
    eye_probs = pd.DataFrame(probs, columns=LABELS)
    eye_probs["patient_id"] = eyes_index["patient_id"].to_numpy()
    grouped = eye_probs.groupby("patient_id")
    patient = grouped.max()
    patient["N"] = grouped["N"].min()
    return patient
