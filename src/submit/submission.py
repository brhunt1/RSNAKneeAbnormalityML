"""Building and validating the submission file.

Run `validate_submission` before every submit. A malformed file wastes one of your
limited daily submissions and, near the deadline, that hurts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import ID_COL, TARGETS


def make_submission(
    study_ids: list[str] | pd.Series,
    preds: np.ndarray,
    path: str | Path = "submission.csv",
) -> pd.DataFrame:
    """Assemble a submission from study ids and a (n, 12) prediction array."""
    preds = np.asarray(preds)
    if preds.shape[1] != len(TARGETS):
        raise ValueError(f"expected {len(TARGETS)} columns, got {preds.shape[1]}")
    if len(study_ids) != len(preds):
        raise ValueError("study_ids and preds length mismatch")

    sub = pd.DataFrame(preds, columns=TARGETS)
    sub.insert(0, ID_COL, list(study_ids))

    validate_submission(sub)
    sub.to_csv(path, index=False)
    return sub


def constant_submission(
    study_ids: list[str] | pd.Series,
    value: float = 0.5,
    path: str | Path = "submission.csv",
) -> pd.DataFrame:
    """The 0.5 baseline. Scores exactly 0.500. Use it to prove the pipeline works."""
    preds = np.full((len(study_ids), len(TARGETS)), value)
    return make_submission(study_ids, preds, path)


def prevalence_submission(
    study_ids: list[str] | pd.Series,
    train_df: pd.DataFrame,
    path: str | Path = "submission.csv",
) -> pd.DataFrame:
    """Predict each label's training prevalence for every study.

    Also scores 0.500, because AUC is rank based and every study gets the same value.
    Useful mainly as a sanity check that your prevalence numbers are sane.
    """
    prev = train_df[TARGETS].mean().values
    preds = np.tile(prev, (len(study_ids), 1))
    return make_submission(study_ids, preds, path)


def validate_submission(sub: pd.DataFrame) -> None:
    """Raise on anything Kaggle would reject or that would silently score badly."""
    expected = [ID_COL] + TARGETS
    if list(sub.columns) != expected:
        raise ValueError(f"column mismatch.\nexpected: {expected}\ngot:      {list(sub.columns)}")

    if sub[ID_COL].duplicated().any():
        dupes = sub.loc[sub[ID_COL].duplicated(), ID_COL].tolist()[:5]
        raise ValueError(f"duplicate study ids, e.g. {dupes}")

    vals = sub[TARGETS].values
    if np.isnan(vals).any():
        raise ValueError("submission contains NaN")
    if not np.isfinite(vals).all():
        raise ValueError("submission contains inf")
    if vals.min() < 0 or vals.max() > 1:
        raise ValueError(f"values outside [0, 1]: min={vals.min()}, max={vals.max()}")


def rank_average(preds: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Blend predictions by averaging ranks rather than raw values.

    Preferred over plain averaging for an AUC metric: it is invariant to how each
    model scales its outputs, so a badly scaled model cannot dominate the blend.
    """
    from scipy.stats import rankdata

    if weights is None:
        weights = [1.0] * len(preds)
    if len(weights) != len(preds):
        raise ValueError("weights and preds length mismatch")

    total = np.zeros_like(np.asarray(preds[0], dtype=float))
    for p, w in zip(preds, weights):
        p = np.asarray(p, dtype=float)
        ranked = np.apply_along_axis(rankdata, 0, p)
        ranked = ranked / ranked.shape[0]
        total += w * ranked

    return total / sum(weights)
