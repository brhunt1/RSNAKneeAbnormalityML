"""Cross validation splits.

Build these once, save to disk, and never change them again. Every experiment must
use the same folds or your comparisons are meaningless.

Multilabel stratification matters here because rare labels (Fracture, Synovitis)
would otherwise land unevenly across folds and their per-label AUC would be noise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import ID_COL, N_FOLDS, SEED, TARGETS


def make_folds(
    train_df: pd.DataFrame,
    n_folds: int = N_FOLDS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Return train_df with a `fold` column added.

    Only rows with complete labels are assigned a fold. Unlabeled rows get fold -1,
    which marks them as pseudo-label-only training data that never enters validation.
    """
    df = train_df.copy()
    labeled_mask = df[TARGETS].notna().all(axis=1)

    df["fold"] = -1
    labeled = df[labeled_mask].reset_index()

    y = labeled[TARGETS].astype(int).values
    X = np.zeros((len(labeled), 1))

    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

        splitter = MultilabelStratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=seed
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Install iterative-stratification: pip install iterative-stratification"
        ) from exc

    for fold, (_, val_idx) in enumerate(splitter.split(X, y)):
        original_idx = labeled.loc[val_idx, "index"].values
        df.loc[original_idx, "fold"] = fold

    return df


def fold_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Positive count per label per fold. Eyeball this before trusting the split.

    If a fold has 0 or 1 positives for a label, that fold's AUC for that label is
    meaningless and the split needs rethinking.
    """
    labeled = df[df["fold"] >= 0]
    return labeled.groupby("fold")[TARGETS].sum().astype(int)


def save_folds(df: pd.DataFrame, path: str | Path = "data/folds.csv") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[[ID_COL, "fold"]].to_csv(path, index=False)
    return path


def load_folds(path: str | Path = "data/folds.csv") -> pd.DataFrame:
    return pd.read_csv(path)
