"""The competition metric: macro-averaged AUC ROC over the 12 targets.

Always report per-label AUC alongside the mean. The mean hides which label is
costing you points, and the per-label view is how you decide what to work on next.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.constants import TARGETS


def per_label_auc(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return {label: AUC}. A label with only one class present yields NaN.

    y_true, y_pred: shape (n_samples, 12), column order must match TARGETS.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.shape[1] != len(TARGETS):
        raise ValueError(f"expected {len(TARGETS)} columns, got {y_true.shape[1]}")

    out: dict[str, float] = {}
    for i, name in enumerate(TARGETS):
        col = y_true[:, i]
        # AUC is undefined when only one class is present in the fold.
        if len(np.unique(col[~np.isnan(col)])) < 2:
            out[name] = float("nan")
        else:
            out[name] = float(roc_auc_score(col, y_pred[:, i]))
    return out


def macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """The competition score. NaN labels are skipped rather than counted as 0."""
    scores = per_label_auc(y_true, y_pred)
    vals = [v for v in scores.values() if not np.isnan(v)]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def score_report(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Per-label AUC plus positive counts, sorted worst first.

    Sorted worst first on purpose: the top of this table is your to-do list.
    """
    scores = per_label_auc(y_true, y_pred)
    y_true = np.asarray(y_true)
    rows = [
        {
            "label": name,
            "auc": scores[name],
            "n_pos": int(np.nansum(y_true[:, i])),
            "prevalence": float(np.nanmean(y_true[:, i])),
        }
        for i, name in enumerate(TARGETS)
    ]
    df = pd.DataFrame(rows).sort_values("auc", na_position="first").reset_index(drop=True)
    return df
