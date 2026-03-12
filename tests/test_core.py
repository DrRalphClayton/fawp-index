"""tests/test_core.py — Core FAWP index and estimator tests."""
import numpy as np
from fawp_index import FAWPAlphaIndex, FAWPResult
from fawp_index.core.estimators import mi_from_arrays, conservative_null_floor


def make_signals(n=300, rng=None):
    rng = rng or np.random.default_rng(42)
    pred   = rng.normal(0, 1, n)
    future = pred + rng.normal(0, 0.5, n)
    action = pred * 0.8 + rng.normal(0, 0.3, n)
    obs    = action + rng.normal(0, 0.2, n)
    return pred, future, action, obs


class TestFAWPAlphaIndex:
    def test_returns_result(self):
        pred, future, action, obs = make_signals()
        result = FAWPAlphaIndex().compute(pred, future, action, obs)
        assert isinstance(result, FAWPResult)

    def test_mi_values_non_negative(self):
        pred, future, action, obs = make_signals()
        result = FAWPAlphaIndex().compute(pred, future, action, obs)
        assert (result.pred_mi_corrected >= 0).all()
        assert (result.steer_mi_corrected >= 0).all()

    def test_alpha_index_is_array(self):
        pred, future, action, obs = make_signals()
        result = FAWPAlphaIndex().compute(pred, future, action, obs)
        assert hasattr(result.alpha_index, '__len__')
        assert len(result.alpha_index) > 0

    def test_in_fawp_is_bool_array(self):
        pred, future, action, obs = make_signals()
        result = FAWPAlphaIndex().compute(pred, future, action, obs)
        assert result.in_fawp.dtype == bool

    def test_peak_alpha_positive(self):
        pred, future, action, obs = make_signals()
        result = FAWPAlphaIndex().compute(pred, future, action, obs)
        assert result.peak_alpha >= 0

    def test_correlated_signals_higher_mi(self):
        rng = np.random.default_rng(0)
        n = 300
        pred = rng.normal(0, 1, n)
        future_hi = pred + rng.normal(0, 0.1, n)
        action_hi = pred * 0.9 + rng.normal(0, 0.1, n)
        obs_hi    = action_hi + rng.normal(0, 0.1, n)
        result_hi = FAWPAlphaIndex().compute(pred, future_hi, action_hi, obs_hi)

        future_lo = rng.normal(0, 1, n)
        action_lo = rng.normal(0, 1, n)
        obs_lo    = rng.normal(0, 1, n)
        result_lo = FAWPAlphaIndex().compute(pred, future_lo, action_lo, obs_lo)

        assert result_hi.pred_mi_corrected.mean() > result_lo.pred_mi_corrected.mean()


class TestEstimators:
    def test_mi_from_arrays_positive(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, 500)
        y = x + rng.normal(0, 0.3, 500)
        assert mi_from_arrays(x, y) > 0

    def test_mi_from_arrays_independent(self):
        rng = np.random.default_rng(2)
        x = rng.normal(0, 1, 500)
        y = rng.normal(0, 1, 500)
        assert mi_from_arrays(x, y) < 0.5

    def test_conservative_null_floor_non_negative(self):
        rng = np.random.default_rng(3)
        x = rng.normal(0, 1, 200)
        y = rng.normal(0, 1, 200)
        assert conservative_null_floor(x, y) >= 0
