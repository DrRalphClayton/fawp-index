"""Tests for fawp_index.exports — to_json, to_markdown, to_html."""

import json
import tempfile
from pathlib import Path

import pytest

from fawp_index import ODWDetector, FAWPAlphaIndexV2


@pytest.fixture(scope="module")
def odw():
    return ODWDetector.from_e9_2_data()


@pytest.fixture(scope="module")
def alpha():
    return FAWPAlphaIndexV2.from_e9_2_data()


# ── ODWResult exports ────────────────────────────────────────────────────────

class TestODWExports:
    def test_to_dict_keys(self, odw):
        d = odw.to_dict()
        assert d["result_type"] == "ODWResult"
        assert "results" in d
        assert "diagnosis" in d
        assert "meta" in d
        r = d["results"]
        assert "fawp_found" in r
        assert "tau_h_plus" in r
        assert "odw_start" in r
        assert "peak_gap_bits" in r

    def test_to_dict_values(self, odw):
        r = odw.to_dict()["results"]
        assert r["fawp_found"] is True
        assert r["tau_h_plus"] == 31
        assert r["tau_f"] == 36
        assert r["odw_start"] == 31
        assert r["odw_end"] == 33

    def test_to_json(self, odw, tmp_path):
        p = tmp_path / "odw.json"
        result = odw.to_json(p)
        assert result == p
        assert p.exists()
        data = json.loads(p.read_text())
        assert data["result_type"] == "ODWResult"
        assert data["results"]["fawp_found"] is True
        assert data["meta"]["fawp_index_version"] == "0.7.0"

    def test_to_json_valid_json(self, odw, tmp_path):
        p = tmp_path / "odw_valid.json"
        odw.to_json(p)
        # Must be valid JSON — no NaN or Infinity
        text = p.read_text()
        parsed = json.loads(text)
        assert isinstance(parsed, dict)

    def test_to_markdown(self, odw, tmp_path):
        p = tmp_path / "odw.md"
        result = odw.to_markdown(p)
        assert result == p
        text = p.read_text()
        assert "# FAWP Analysis" in text
        assert "FAWP DETECTED" in text
        assert "tau_h" in text
        assert "tau_f" in text
        assert "fawp-index" in text

    def test_to_html(self, odw, tmp_path):
        p = tmp_path / "odw.html"
        result = odw.to_html(p)
        assert result == p
        text = p.read_text()
        assert "<!DOCTYPE html>" in text
        assert "FAWP DETECTED" in text
        assert "tau" in text
        assert "0E2550" in text  # brand colour

    def test_to_html_is_complete(self, odw, tmp_path):
        p = tmp_path / "odw_complete.html"
        odw.to_html(p)
        text = p.read_text()
        assert text.strip().endswith("</html>")


# ── AlphaV2Result exports ────────────────────────────────────────────────────

class TestAlphaV2Exports:
    def test_to_dict_keys(self, alpha):
        d = alpha.to_dict()
        assert d["result_type"] == "AlphaV2Result"
        r = d["results"]
        assert "fawp_detected" in r
        assert "peak_alpha2" in r
        assert "params" in r
        assert "curves" in d
        assert "alpha2" in d["curves"]
        assert "tau" in d["curves"]

    def test_to_json(self, alpha, tmp_path):
        p = tmp_path / "alpha.json"
        alpha.to_json(p)
        data = json.loads(p.read_text())
        assert data["result_type"] == "AlphaV2Result"
        assert isinstance(data["curves"]["alpha2"], list)
        assert len(data["curves"]["tau"]) == len(data["curves"]["alpha2"])

    def test_to_json_no_curves(self, alpha, tmp_path):
        p = tmp_path / "alpha_summary.json"
        alpha.to_json(p, include_curves=False)
        data = json.loads(p.read_text())
        assert "curves" not in data
        assert "results" in data

    def test_to_markdown(self, alpha, tmp_path):
        p = tmp_path / "alpha.md"
        alpha.to_markdown(p)
        text = p.read_text()
        assert "Alpha Index v2.1" in text
        assert "peak" in text.lower()
        assert "fawp-index" in text

    def test_to_html(self, alpha, tmp_path):
        p = tmp_path / "alpha.html"
        alpha.to_html(p)
        text = p.read_text()
        assert "<!DOCTYPE html>" in text
        assert "Alpha" in text
        assert text.strip().endswith("</html>")

    def test_to_html_has_chart(self, alpha, tmp_path):
        """HTML should contain an embedded base64 chart."""
        p = tmp_path / "alpha_chart.html"
        alpha.to_html(p)
        text = p.read_text()
        assert "data:image/png;base64," in text


# ── Report ───────────────────────────────────────────────────────────────────

class TestReport:
    def test_odw_report(self, odw, tmp_path):
        from fawp_index.report import generate_report
        p = tmp_path / "odw_report.pdf"
        result = generate_report(odw, p, title="Test ODW Report")
        assert result == p
        assert p.exists()
        assert p.stat().st_size > 20_000

    def test_alpha_report(self, alpha, tmp_path):
        from fawp_index.report import generate_report
        p = tmp_path / "alpha_report.pdf"
        generate_report(alpha, p, title="Test Alpha Report")
        assert p.exists()
        assert p.stat().st_size > 20_000

    def test_combined_report(self, odw, alpha, tmp_path):
        from fawp_index.report import generate_report
        p = tmp_path / "combined.pdf"
        generate_report({"odw": odw, "alpha": alpha}, p, title="Combined")
        assert p.exists()
        assert p.stat().st_size > 30_000

    def test_lab_mode(self, odw, tmp_path):
        from fawp_index.report import FAWPReport
        p = tmp_path / "lab.pdf"
        FAWPReport(mode="lab").build(odw, p)
        assert p.exists()

    def test_invalid_mode(self):
        from fawp_index.report import FAWPReport
        with pytest.raises(ValueError):
            FAWPReport(mode="banana")

    def test_no_figures(self, odw, tmp_path):
        from fawp_index.report import generate_report
        p = tmp_path / "nofig.pdf"
        generate_report(odw, p, include_figures=False)
        assert p.exists()

    def test_no_methods(self, odw, tmp_path):
        from fawp_index.report import generate_report
        p = tmp_path / "nomethods.pdf"
        generate_report(odw, p, include_methods=False)
        assert p.exists()
