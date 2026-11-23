"""The numbers every verdict rests on: AUC and the paired bootstrap interval."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

import fastci


@pytest.fixture
def scores():
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, 1500)

    # Rounded on purpose: random forests produce many tied probabilities.
    good = np.round(rng.random(1500) * 0.6 + y * 0.3, 2)
    worse = np.round(rng.random(1500) * 0.6 + y * 0.2, 2)
    stars = rng.integers(0, 1200, 1500)
    return y, good, worse, stars


def test_auc_matches_scikit_learn_with_ties(scores):
    y, good, _, _ = scores
    assert fastci.auc(y, good) == pytest.approx(roc_auc_score(y, good), abs=1e-12)


def test_auc_of_a_constant_prediction_is_chance():
    y = np.array([0, 1, 0, 1, 1, 0])
    assert fastci.auc(y, np.zeros(6)) == 0.5


def test_auc_does_not_depend_on_row_order(scores):
    # The bug the first code review found: ties ranked by row order changed the score.
    y, good, _, _ = scores
    order = np.random.default_rng(1).permutation(len(y))
    assert fastci.auc(y[order], good[order]) == pytest.approx(fastci.auc(y, good), abs=1e-12)


def test_a_clearly_better_model_has_an_interval_above_zero(scores):
    y, good, worse, stars = scores
    lo, hi = fastci.paired_ci(y, good, worse, groups=stars, n=1000)
    assert 0 < lo < hi


def test_identical_models_have_an_interval_around_zero(scores):
    y, good, _, stars = scores
    lo, hi = fastci.paired_ci(y, good, good, groups=stars, n=500)
    assert lo <= 0 <= hi


def test_resampling_whole_stars_widens_the_interval():
    # Planets of one star move together. Treating them as independent makes the interval
    # too narrow, which is how noise would slip past the NOISE and OVERFIT gates.
    rng = np.random.default_rng(3)
    stars = np.repeat(np.arange(200), 5)
    y = np.repeat(rng.integers(0, 2, 200), 5)
    star_effect = np.repeat(rng.normal(0, 0.3, 200), 5)
    a = y * 0.3 + star_effect + rng.normal(0, 0.05, 1000)
    b = y * 0.3 + rng.normal(0, 0.05, 1000)

    lo_obj, hi_obj = fastci.paired_ci(y, a, b, n=800)
    lo_star, hi_star = fastci.paired_ci(y, a, b, groups=stars, n=800)
    assert (hi_star - lo_star) > (hi_obj - lo_obj)


def test_native_and_numpy_agree(scores, monkeypatch):
    if not fastci.NATIVE:
        pytest.skip("C++ library not built (make -C exo/native)")
    y, good, _, _ = scores
    native = fastci.auc(y, good)
    monkeypatch.setattr(fastci, "NATIVE", False)
    assert fastci.auc(y, good) == pytest.approx(native, abs=1e-12)
