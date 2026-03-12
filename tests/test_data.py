"""tests/test_data.py — Bundled data paths and DataFrame API tests."""
import numpy as np
import pandas as pd
import pytest
from fawp_index.data import (
    E1_HORIZONS_SWEEP, E2_CONVERGENCE,
    E3_CURVES_LINEAR, E3_CURVES_QUADRATIC, E3_CURVES_SATURATING, E3_HORIZONS_SUMMARY,
    E4_ROBUSTNESS, E5_CONTROL_CLIFF,
    E6_SURFACE_BITS_MEAN, E6_SURFACE_BITS_STD,
    E6_SURFACE_DROPOUT_MEAN, E6_SURFACE_DROPOUT_STD,
    E7_QUANTUM,
    E8_DATA, E8_CONFIRM_FULL, E8_SIGNIFICANCE,
)


class TestDataPaths:
    @pytest.mark.parametrize("path,name", [
        (E1_HORIZONS_SWEEP,        "E1"),
        (E2_CONVERGENCE,           "E2"),
        (E3_CURVES_LINEAR,         "E3 linear"),
        (E3_CURVES_QUADRATIC,      "E3 quadratic"),
        (E3_CURVES_SATURATING,     "E3 saturating"),
        (E3_HORIZONS_SUMMARY,      "E3 summary"),
        (E4_ROBUSTNESS,            "E4"),
        (E5_CONTROL_CLIFF,         "E5"),
        (E6_SURFACE_BITS_MEAN,     "E6 bits mean"),
        (E6_SURFACE_BITS_STD,      "E6 bits std"),
        (E6_SURFACE_DROPOUT_MEAN,  "E6 dropout mean"),
        (E6_SURFACE_DROPOUT_STD,   "E6 dropout std"),
        (E7_QUANTUM,               "E7"),
        (E8_DATA,                  "E8 data"),
        (E8_CONFIRM_FULL,          "E8 confirm full"),
        (E8_SIGNIFICANCE,          "E8 significance"),
    ])
    def test_csv_exists_and_loads(self, path, name):
        assert path.exists(), f"{name} CSV not found: {path}"
        df = pd.read_csv(path)
        assert len(df) > 0, f"{name} CSV is empty"

    def test_e1_has_required_columns(self):
        df = pd.read_csv(E1_HORIZONS_SWEEP)
        for col in ['P', 'alpha', 'epsilon', 'tau_h_theory', 'status']:
            assert col in df.columns

    def test_e5_has_required_columns(self):
        df = pd.read_csv(E5_CONTROL_CLIFF)
        for col in ['delay_steps', 'mean_cost', 'failure_rate', 'mi_bits']:
            assert col in df.columns

    def test_e7_chsh_within_tsirelson(self):
        df = pd.read_csv(E7_QUANTUM)
        assert (df['CHSH'] <= 2 * np.sqrt(2) + 1e-10).all()

    def test_e7_no_signaling_near_zero(self):
        df = pd.read_csv(E7_QUANTUM)
        assert (df['NS_Dev'] < 1e-10).all()


class TestDataFrameAPI:
    def make_df(self, n=300, rng=None):
        rng = rng or np.random.default_rng(42)
        pred = rng.normal(0, 1, n)
        return pd.DataFrame({
            'pred':    pred,
            'future':  pred + rng.normal(0, 0.5, n),
            'action':  pred * 0.8 + rng.normal(0, 0.3, n),
            'obs':     pred * 0.6 + rng.normal(0, 0.4, n),
        })

    def test_fawp_from_dataframe(self):
        from fawp_index import fawp_from_dataframe
        df = self.make_df()
        result = fawp_from_dataframe(df, pred_col='pred', action_col='action')
        assert result is not None

    def test_fawp_rolling_adds_columns(self):
        from fawp_index import fawp_rolling
        df = self.make_df(n=400)
        out = fawp_rolling(df, pred_col='pred', action_col='action',
                           future_col='future', window=100)
        for col in ['fawp_pred_mi', 'fawp_steer_mi', 'fawp_gap', 'fawp_in_regime']:
            assert col in out.columns
