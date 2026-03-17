# Contributing to fawp-index

Thank you for your interest in contributing! This is an open-science project — contributions of all kinds are welcome.

## Types of contributions

- **Bug reports** — open a [GitHub issue](https://github.com/DrRalphClayton/fawp-index/issues)
- **New benchmarks** — add cases to `fawp_index/benchmarks.py`
- **Weather variables** — extend `fawp_index/weather.py`
- **Empirical validation** — new SPHERE-series data welcome
- **Documentation** — improve examples, docstrings, or the docs/ folder

## Setup

```bash
git clone https://github.com/DrRalphClayton/fawp-index
cd fawp-index
pip install -e ".[dev,test,plot]"
```

## Running tests

```bash
pytest tests/ -v --ignore=tests/test_weather.py --ignore=tests/test_finance.py
```

## Code style

We use [ruff](https://docs.astral.sh/ruff/) for linting:

```bash
ruff check fawp_index/
```

## Key conventions

- All calibration constants go in `fawp_index/constants.py` with a source reference (paper + equation)
- New SPHERE results → add `E{N}_*` constants with clear docstring
- No breaking changes to `fawp_from_series()` or `fawp_from_open_meteo()` signatures without a major version bump
- Scientific claims must reference a SPHERE paper DOI

## Pull requests

1. Fork → branch → PR against `main`
2. Include a test or example demonstrating the change
3. Update `CHANGELOG.md` with a brief entry

## Questions?

Open an issue or email via the GitHub profile.
