"""Tests for fawp_index.benchmarks — eight canonical cases."""

import json
import pytest

from fawp_index.benchmarks import (
    run_all, BenchmarkSuite, BenchmarkFailure,
    clean_control, prediction_only, control_only,
    noisy_false_positive, delayed_collapse,
    gradual_fade, multi_regime, spiky_false_positive,
)


# ── Individual cases ─────────────────────────────────────────────────────────

class TestCases:
    def test_clean_control_passes(self):
        r = clean_control()
        assert r.passed
        assert r.odw_result.fawp_found
        assert r.expected_fawp is True
        r.verify()  # should not raise

    def test_prediction_only_passes(self):
        r = prediction_only()
        assert r.passed
        assert not r.odw_result.fawp_found
        assert r.expected_fawp is False
        r.verify()

    def test_control_only_passes(self):
        r = control_only()
        assert r.passed
        assert not r.odw_result.fawp_found
        r.verify()

    def test_noisy_false_positive_passes(self):
        r = noisy_false_positive()
        assert r.passed
        assert not r.odw_result.fawp_found
        r.verify()

    def test_delayed_collapse_passes(self):
        r = delayed_collapse()
        assert r.passed
        assert r.odw_result.fawp_found
        r.verify()

    def test_verify_raises_on_wrong_result(self, monkeypatch):
        r = clean_control()
        # Flip the expected outcome so verify() should raise
        monkeypatch.setattr(r, "expected_fawp", False)
        with pytest.raises(BenchmarkFailure):
            r.verify()

    def test_result_has_required_fields(self):
        r = clean_control()
        assert r.name
        assert r.description
        assert r.tau is not None
        assert r.pred_mi is not None
        assert r.steer_mi is not None
        assert r.fail_rate is not None
        assert r.odw_result is not None
        assert r.verdict

    def test_to_dict(self):
        r = clean_control()
        d = r.to_dict()
        assert d["name"] == "clean_control"
        assert d["expected_fawp"] is True
        assert d["detected_fawp"] is True
        assert d["passed"] is True
        assert "odw" in d
        assert "tau_h_plus" in d["odw"]

    def test_no_sim_result_by_default(self):
        r = clean_control()
        assert r.sim_result is None


    def test_gradual_fade_passes(self):
        r = gradual_fade()
        assert r.passed
        assert r.odw_result.fawp_found
        assert r.expected_fawp is True
        r.verify()

    def test_multi_regime_passes(self):
        r = multi_regime()
        assert r.passed
        assert r.odw_result.fawp_found
        assert r.expected_fawp is True
        r.verify()

    def test_spiky_false_positive_passes(self):
        r = spiky_false_positive()
        assert r.passed
        assert not r.odw_result.fawp_found
        assert r.expected_fawp is False
        r.verify()


# ── BenchmarkSuite ───────────────────────────────────────────────────────────

class TestSuite:
    def test_run_all_returns_suite(self):
        suite = run_all()
        assert isinstance(suite, BenchmarkSuite)
        assert len(suite.results) == 8

    def test_all_pass(self):
        suite = run_all()
        assert suite.n_passed == 8
        assert suite.n_failed == 0

    def test_verify_all(self):
        suite = run_all()
        suite.verify_all()  # should not raise

    def test_summary_str(self):
        suite = run_all()
        s = suite.summary()
        assert "clean_control" in s
        assert "PASS" in s
        assert "8" in s  # 8 total cases

    def test_to_json(self, tmp_path):
        suite = run_all()
        p = tmp_path / "bench.json"
        suite.to_json(p)
        data = json.loads(p.read_text())
        assert data["n_passed"] == 8
        assert data["n_failed"] == 0
        assert len(data["cases"]) == 8
        assert data["fawp_index_version"] == "0.20.0"

    def test_to_html(self, tmp_path):
        suite = run_all()
        p = tmp_path / "bench.html"
        suite.to_html(p)
        text = p.read_text()
        assert "<!DOCTYPE html>" in text
        assert "Benchmark Suite" in text
        assert "clean_control" in text
        assert "PASS" in text
        assert text.strip().endswith("</html>")

    def test_html_has_charts(self, tmp_path):
        suite = run_all()
        p = tmp_path / "bench_charts.html"
        suite.to_html(p)
        text = p.read_text()
        assert "data:image/png;base64," in text

    def test_n_passed_property(self):
        suite = run_all()
        assert suite.n_passed + suite.n_failed == len(suite.results)


# ── Top-level API ────────────────────────────────────────────────────────────

class TestTopLevelAPI:
    def test_imported_from_package(self):
        from fawp_index import (  # noqa: F811
            run_benchmarks, BenchmarkSuite,
            BenchmarkFailure, clean_control, prediction_only,
            control_only, noisy_false_positive, delayed_collapse,
        )
        assert callable(run_benchmarks)
        assert callable(clean_control)

    def test_run_benchmarks_alias(self):
        from fawp_index import run_benchmarks  # noqa: F811
        suite = run_benchmarks()
        assert suite.n_passed == 8
