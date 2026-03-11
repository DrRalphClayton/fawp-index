"""
fawp_index.data — Bundled experimental data from Clayton (2026)

These CSVs are the actual output files from Experiment E8, as reported in:
  "Forecasting Without Power: Agency Horizons and the Leverage Gap"
  DOI: https://doi.org/10.5281/zenodo.18663547

Files:
  e8_data.csv               — original E8 leverage gap sweep (delay 0-78, step 2)
  e8_confirm_final_full.csv — full confirmation run (delay 0-80), stratified MI + shuffle controls
  e8_confirm_significance.csv — significance analysis with confidence intervals
"""

import pathlib

DATA_DIR = pathlib.Path(__file__).parent

def get_path(filename: str) -> pathlib.Path:
    """Return the absolute path to a bundled data file."""
    p = DATA_DIR / filename
    if not p.exists():
        raise FileNotFoundError(f"Bundled data file not found: {filename}")
    return p

E8_DATA = DATA_DIR / "e8_data.csv"
E8_CONFIRM_FULL = DATA_DIR / "e8_confirm_final_full.csv"
E8_SIGNIFICANCE = DATA_DIR / "e8_confirm_significance.csv"
