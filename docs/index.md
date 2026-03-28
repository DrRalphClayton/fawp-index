# fawp-index

**FAWP Alpha Index** — Information-Control Exclusion Principle detector.

[![PyPI](https://img.shields.io/pypi/v/fawp-index)](https://pypi.org/project/fawp-index/)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.18673949-blue)](https://doi.org/10.5281/zenodo.18673949)
[![Live Demo](https://img.shields.io/badge/🔴%20Live%20Demo-fawp--scanner.info-gold)](https://fawp-scanner.info)

## What is FAWP?

FAWP detects the **Information-Control Exclusion Principle** — the regime where forecast
skill persists after the window to act on that forecast has already closed.

## Quick install

```bash
pip install fawp-index
```

## Quick start

```python
import fawp_index as fi
result = fi.scan(your_dataframe)
print(result.summary())
```

See [Getting started](quickstart.md) for a full walkthrough.

## Live scanners

Three scanners are available at [fawp-scanner.info](https://fawp-scanner.info):

- **Finance** — equities, crypto, ETFs
- **Weather** — ERA5 reanalysis for any location
- **Seismic** — USGS earthquake catalogs

## Papers

- E1–E7: [doi:10.5281/zenodo.18663547](https://doi.org/10.5281/zenodo.18663547)
- E8 / SPHERE-16: [doi:10.5281/zenodo.18673949](https://doi.org/10.5281/zenodo.18673949)
- E9 suite: [doi:10.5281/zenodo.19065421](https://doi.org/10.5281/zenodo.19065421)
