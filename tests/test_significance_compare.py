"""Tests for fawp_index.significance and fawp_index.compare."""

import json
import pytest
import numpy as np

from fawp_index import ODWDetector, FAWPAlphaIndexV2
from fawp_index.significance import fawp_significance, FAWPSignificance, SignificanceResult
from fawp_index.compare import compare_fawp, ComparisonResult
from fawp_index.benchmarks import clean_control, delayed_collapse


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def odw():
    return ODWDetector.from_e9_2_data()

@pytest.fixture(scope="module")
def alpha():
    return FAWPAlphaIndexV2.from_e9_2_data()

@pytest.fixture(scope="module")
def sig_seed(odw):
    return fawp_significance(odw, n_bootstrap=80, seed=42)


# ── SignificanceResult structure ──────────────────────────────────────────────

class TestSignificanceResult:
    def test_returns_significance_result(self, sig_seed):
        assert isinstance(sig_seed, SignificanceResult)

    def test_method_is_seed_bootstrap(self, sig_seed):
        assert sig_seed.method == "seed_bootstrap"

    def test_p_value_fawp_in_range(self, sig_seed):
        assert 0.0 <= sig_seed.p_value_fawp <= 1.0

    def test_p_value_null_in_range(self, sig_seed):
        assert 0.0 <= sig_seed.p_value_null <= 1.0

    def test_p_value_fawp_high_for_real_signal(self, sig_seed):
        # E9.2 is a real FAWP signal — should be detected most of the time
        assert sig_seed.p_value_fawp > 0.5

    def test_p_value_null_low(self, sig_seed):
        # False positive rate should be low
        assert sig_seed.p_value_null < 0.5

    def test_ci_tau_h_is_tuple(self, sig_seed):
        lo, hi = sig_seed.ci_tau_h
        assert lo is not None and hi is not None
        assert lo <= hi

    def test_ci_odw_start_is_tuple(self, sig_seed):
        lo, hi = sig_seed.ci_odw_start
        assert lo is not None and hi is not None
        assert lo <= hi

    def test_ci_peak_gap_is_tuple(self, sig_seed):
        lo, hi = sig_seed.ci_peak_gap
        assert lo is not None and hi is not None
        assert lo <= hi
        assert lo >= 0.0

    def test_bootstrap_samples_length(self, sig_seed):
        assert len(sig_seed.tau_h_samples) == sig_seed.n_bootstrap
        assert len(sig_seed.peak_gap_samples) == sig_seed.n_bootstrap

    def test_significant_property(self, sig_seed):
        # Should be significant (real FAWP signal)
        assert isinstance(sig_seed.significant, bool)

    def test_confidence_pct(self, sig_seed):
        assert sig_seed.confidence_pct == 95

    def test_summary_str(self, sig_seed):
        s = sig_seed.summary()
        assert "FAWP" in s
        assert "bootstrap" in s.lower()
        assert "p_value" in s.lower() or "P(FAWP" in s

    def test_pred_p_values_none_for_seed_method(self, sig_seed):
        # seed_bootstrap doesn't compute per-tau p-values
        assert sig_seed.pred_p_values is None

    def test_to_dict(self, sig_seed):
        d = sig_seed.to_dict()
        assert d["method"] == "seed_bootstrap"
        assert "p_value_fawp" in d
        assert "confidence_intervals" in d
        assert "tau_h" in d["confidence_intervals"]  # key is "tau_h", not "ci"

    def test_to_json(self, sig_seed, tmp_path):
        p = tmp_path / "sig.json"
        sig_seed.to_json(p)
        data = json.loads(p.read_text())
        assert data["method"] == "seed_bootstrap"
        assert 0.0 <= data["p_value_fawp"] <= 1.0

    def test_to_html(self, sig_seed, tmp_path):
        p = tmp_path / "sig.html"
        sig_seed.to_html(p)
        text = p.read_text()
        assert "<!DOCTYPE html>" in text
        assert "Significance" in text
        assert "data:image/png;base64," in text
        assert text.strip().endswith("</html>")


# ── mi_permutation method ──────────────────────────────────────────────────

class TestMIPermutation:
    def test_from_mi_curves(self, odw):
        from fawp_index.data import E9_2_AGGREGATE_CURVES
        import pandas as pd
        df = pd.read_csv(E9_2_AGGREGATE_CURVES).sort_values("tau")

        sig = fawp_significance(
            odw,
            tau       = df["tau"].values,
            pred_raw  = df["pred_strat_corr"].values,
            steer_raw = df["steer_u_corr"].values,
            fail_rate = df["fail_rate"].values,
            n_bootstrap=60, n_null=40, seed=42,
        )
        assert sig.method == "mi_permutation"
        assert sig.pred_p_values is not None
        assert len(sig.pred_p_values) == len(df)
        assert all(0.0 <= p <= 1.0 for p in sig.pred_p_values)

    def test_from_mi_curves_requires_tau(self, odw):
        with pytest.raises((ValueError, TypeError)):
            fawp_significance(
                odw,
                pred_raw=np.ones(10),
                steer_raw=np.ones(10),
                # missing tau and fail_rate
            )


# ── fawp_significance convenience function ───────────────────────────────────

class TestConvenienceFunction:
    def test_default_is_seed_bootstrap(self, odw):
        sig = fawp_significance(odw, n_bootstrap=50, seed=0)
        assert sig.method == "seed_bootstrap"

    def test_top_level_import(self):
        from fawp_index import fawp_significance as fs, FAWPSignificance, SignificanceResult
        assert callable(fs)
        assert callable(FAWPSignificance)


# ── ComparisonResult structure ────────────────────────────────────────────────

class TestCompare:
    def test_returns_comparison_result(self, odw):
        odw2 = ODWDetector.from_e9_2_data(steering="xi")
        cmp = compare_fawp(odw, odw2, label_a="u", label_b="xi")
        assert isinstance(cmp, ComparisonResult)

    def test_labels(self, odw):
        odw2 = ODWDetector.from_e9_2_data(steering="xi")
        cmp = compare_fawp(odw, odw2, label_a="u-steer", label_b="xi-steer")
        assert cmp.label_a == "u-steer"
        assert cmp.label_b == "xi-steer"

    def test_result_type_odw(self, odw):
        odw2 = ODWDetector.from_e9_2_data(steering="xi")
        cmp = compare_fawp(odw, odw2)
        assert cmp.result_type == "ODW"

    def test_rows_not_empty(self, odw):
        odw2 = ODWDetector.from_e9_2_data(steering="xi")
        cmp = compare_fawp(odw, odw2)
        assert len(cmp.rows) > 0

    def test_winner_is_valid(self, odw):
        odw2 = ODWDetector.from_e9_2_data(steering="xi")
        cmp = compare_fawp(odw, odw2)
        assert cmp.winner_overall in ("A", "B", "tie")

    def test_scores_sum_to_at_most_n_rows(self, odw):
        odw2 = ODWDetector.from_e9_2_data(steering="xi")
        cmp = compare_fawp(odw, odw2)
        assert cmp.score_a + cmp.score_b <= len(cmp.rows)

    def test_summary_str(self, odw):
        odw2 = ODWDetector.from_e9_2_data(steering="xi")
        cmp = compare_fawp(odw, odw2, label_a="u", label_b="xi")
        s = cmp.summary()
        assert "u" in s and "xi" in s
        assert "winner" in s.lower()

    def test_to_dict(self, odw):
        odw2 = ODWDetector.from_e9_2_data(steering="xi")
        cmp = compare_fawp(odw, odw2)
        d = cmp.to_dict()
        assert "rows" in d
        assert "winner_overall" in d
        assert len(d["rows"]) == len(cmp.rows)

    def test_to_json(self, odw, tmp_path):
        odw2 = ODWDetector.from_e9_2_data(steering="xi")
        cmp = compare_fawp(odw, odw2, label_a="u", label_b="xi")
        p = tmp_path / "cmp.json"
        cmp.to_json(p)
        data = json.loads(p.read_text())
        assert data["label_a"] == "u"
        assert data["label_b"] == "xi"
        assert data["winner_overall"] in ("A", "B", "tie")

    def test_to_html(self, odw, tmp_path):
        odw2 = ODWDetector.from_e9_2_data(steering="xi")
        cmp = compare_fawp(odw, odw2, label_a="u", label_b="xi")
        p = tmp_path / "cmp.html"
        cmp.to_html(p)
        text = p.read_text()
        assert "<!DOCTYPE html>" in text
        assert "u" in text and "xi" in text
        assert "data:image/png;base64," in text
        assert text.strip().endswith("</html>")

    def test_alpha2_comparison(self, alpha):
        alpha2 = FAWPAlphaIndexV2.from_e9_2_data(steering="xi")
        cmp = compare_fawp(alpha, alpha2, label_a="u", label_b="xi")
        assert cmp.result_type == "AlphaV2"
        assert len(cmp.rows) > 0

    def test_benchmark_comparison(self):
        r1 = clean_control()
        r2 = delayed_collapse()
        cmp = compare_fawp(r1, r2, label_a="clean", label_b="delayed")
        assert cmp.result_type == "ODW"
        assert len(cmp.rows) > 0

    def test_type_mismatch_raises(self, odw, alpha):
        with pytest.raises(TypeError):
            compare_fawp(odw, alpha)

    def test_top_level_import(self):
        from fawp_index import compare_fawp as cf, ComparisonResult
        assert callable(cf)
