"""Multilabel classification metrics shared by all models."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.labels import LABELS


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
    label_names: list[str] = LABELS,
) -> dict[str, float]:
    """Per-class and macro ROC-AUC, PR-AUC, F1, precision and recall.

    F1, precision and recall use ``y_prob >= thresholds`` (0.5 if None) and give 0
    where undefined. AUCs of a class with one value in y_true are NaN and left out of
    the macro mean.
    """
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    y_pred = y_prob >= (0.5 if thresholds is None else thresholds)

    per_class = {"roc_auc": [], "pr_auc": [], "f1": [], "precision": [], "recall": []}
    for t, p, pred in zip(y_true.T, y_prob.T, y_pred.T):
        has_both = 0 < t.sum() < len(t)
        per_class["roc_auc"].append(roc_auc_score(t, p) if has_both else np.nan)
        per_class["pr_auc"].append(
            average_precision_score(t, p) if has_both else np.nan
        )
        per_class["f1"].append(f1_score(t, pred, zero_division=0))
        per_class["precision"].append(precision_score(t, pred, zero_division=0))
        per_class["recall"].append(recall_score(t, pred, zero_division=0))

    metrics: dict[str, float] = {}
    for metric, values in per_class.items():
        metrics |= {
            f"{metric}_{name}": float(v) for name, v in zip(label_names, values)
        }
        metrics[f"{metric}_macro"] = float(np.nanmean(values))
    return metrics


def find_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """Per-class threshold that maximizes F1 (choose on val, apply to test).

    A class without positives keeps 0.5.
    """
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    thresholds = np.full(y_prob.shape[1], 0.5)
    for k in range(y_prob.shape[1]):
        if not y_true[:, k].any():
            continue
        precision, recall, candidates = precision_recall_curve(
            y_true[:, k], y_prob[:, k]
        )
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
        thresholds[k] = candidates[np.argmax(f1[:-1])]
    return thresholds
