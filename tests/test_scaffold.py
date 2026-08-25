"""Tests for the parts of the scaffold that do not need the competition data.

Run with: python -m pytest tests/ -q
"""

import numpy as np
import pandas as pd
import pytest

from src.constants import ID_COL, N_TARGETS, TARGETS
from src.eval.metric import macro_auc, per_label_auc, score_report
from src.submit.submission import (
    constant_submission,
    make_submission,
    rank_average,
    validate_submission,
)


def _fake(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.3, size=(n, N_TARGETS))
    # Predictions correlated with truth so AUC lands above 0.5.
    p = y * 0.6 + rng.random((n, N_TARGETS)) * 0.4
    ids = [f"study_{i}" for i in range(n)]
    return ids, y, p


def test_targets_shape():
    assert N_TARGETS == 12
    assert len(set(TARGETS)) == 12


def test_perfect_and_inverted_auc():
    y = np.tile(np.array([0, 1]), (50, 1)).reshape(-1, 1).repeat(N_TARGETS, axis=1)
    assert macro_auc(y, y.astype(float)) == pytest.approx(1.0)
    assert macro_auc(y, 1.0 - y.astype(float)) == pytest.approx(0.0)


def test_constant_prediction_scores_half():
    _, y, _ = _fake()
    const = np.full_like(y, 0.5, dtype=float)
    assert macro_auc(y, const) == pytest.approx(0.5)


def test_single_class_label_is_nan_not_zero():
    """A label with no positives must not silently drag the mean toward zero."""
    _, y, p = _fake()
    y[:, 0] = 0  # no positives for ACL
    scores = per_label_auc(y, p)
    assert np.isnan(scores["ACL"])
    assert not np.isnan(macro_auc(y, p))  # other 11 labels still score


def test_score_report_sorted_worst_first():
    _, y, p = _fake()
    df = score_report(y, p)
    assert list(df.columns) == ["label", "auc", "n_pos", "prevalence"]
    aucs = df["auc"].dropna().values
    assert (np.diff(aucs) >= -1e-9).all()


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        macro_auc(np.zeros((10, 12)), np.zeros((10, 11)))


def test_submission_roundtrip(tmp_path):
    ids, _, p = _fake(n=50)
    path = tmp_path / "sub.csv"
    sub = make_submission(ids, p, path)
    assert list(sub.columns) == [ID_COL] + TARGETS
    reloaded = pd.read_csv(path)
    assert len(reloaded) == 50
    validate_submission(reloaded)


def test_constant_submission_is_valid(tmp_path):
    ids, _, _ = _fake(n=10)
    sub = constant_submission(ids, 0.5, tmp_path / "sub.csv")
    assert (sub[TARGETS].values == 0.5).all()


def test_validate_catches_bad_submissions():
    ids, _, p = _fake(n=20)
    good = pd.DataFrame(p, columns=TARGETS)
    good.insert(0, ID_COL, ids)

    bad_range = good.copy()
    bad_range.loc[0, "ACL"] = 1.5
    with pytest.raises(ValueError, match="outside"):
        validate_submission(bad_range)

    bad_nan = good.copy()
    bad_nan.loc[0, "ACL"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_submission(bad_nan)

    bad_dupe = pd.concat([good, good.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_submission(bad_dupe)

    bad_cols = good.rename(columns={"ACL": "acl"})
    with pytest.raises(ValueError, match="column mismatch"):
        validate_submission(bad_cols)


def test_rank_average_preserves_auc_under_rescaling():
    """The point of rank averaging: a badly scaled model does not dominate."""
    _, y, p = _fake(n=300)
    scaled = p * 1000.0  # same ranking, wildly different scale
    blended = rank_average([p, scaled])
    assert macro_auc(y, blended) == pytest.approx(macro_auc(y, p), abs=1e-6)


def test_rank_average_output_in_unit_range():
    _, _, p1 = _fake(n=100, seed=1)
    _, _, p2 = _fake(n=100, seed=2)
    out = rank_average([p1, p2])
    assert out.shape == p1.shape
    assert out.min() >= 0.0 and out.max() <= 1.0
