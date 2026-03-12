# Changelog — fawp-index

All notable changes to this project are documented here.
Versions follow [Semantic Versioning](https://semver.org/).

---

## [0.6.0] — 2026-03-12

### Added

**`fawp_index.core.alpha_v2` — Upgraded FAWP Alpha Index v2.1**

Full implementation of the α₂(τ) formula from Clayton (2026),
*"Future Access Without Presence: The Information–Control Exclusion
Principle in Unstable Dynamics"* (doi:10.5281/zenodo.18673949).

- `FAWPAlphaIndexV2` — computes α₂(τ) from null-corrected MI curves
- `AlphaV2Result` — result object with `.summary()` and `.plot()`
- Null-corrected MI input (shuffle + shift, β=0.99 recommended)
- Robust stability window `S_m(τ) = min_{k=0..m} Ĩ_pred(τ-k)` (default m=5)
- Log-slope resonance `R_log(τ)` — scale-invariant, no finite-difference noise
- Hard gate `g(τ)`: τ≥1, `S_m > η`, `Ĩ_steer ≤ ε` (default η=ε=1e-4 bits)
- `FAWPAlphaIndexV2.from_e9_2_data(steering='u'|'xi')` — one-line demo from bundled data
- Calibration anchors: β=0.99, m=5, η=ε≈10⁻⁴ bits (from record-chain validation)

**`fawp_index.detection.odw` — Operational Detection Window detector**

Clean port of the E9 detection methodology.

- `ODWDetector` — finds τ⁺ₕ, τf, ODW from null-corrected MI + failure-rate curves
- `ODWResult` — result with `.summary()`, all key quantities
- `ODWDetector.from_e9_2_data(steering='u'|'xi')` — reproduces E9.2 results directly
- Persistence filter (m-of-n rule) carried over from E9.2 script
- Default `epsilon=0.01` reproduces E9.2 paper numbers; use `1e-4` for strict mode

**Bundled E9.2 data**

- `fawp_index.data.E9_2_AGGREGATE_CURVES` — aggregate τ-wise curves (20 seeds, 400 trials/τ)
- `fawp_index.data.E9_2_SEED_CURVES` — per-seed curves
- `fawp_index.data.E9_2_SUMMARY_JSON` — summary JSON with full config + aggregate results
- `examples/e9_2_u_vs_xi.py` — the generation script

Key E9.2 numbers bundled: τ⁺ₕ=31 (u and ξ), τf=36, ODW=[31,33],
peak leverage gap=1.55 bits at τ=34, peak prediction=2.20 bits at τ=9.

### New exports

```python
from fawp_index import FAWPAlphaIndexV2, AlphaV2Result
from fawp_index import ODWDetector, ODWResult
from fawp_index.data import E9_2_AGGREGATE_CURVES, E9_2_SEED_CURVES, E9_2_SUMMARY_JSON
```

---

## [0.5.1] — 2026-03-12

### Added
- `fawp_index.explain` — plain-English diagnosis module
  - `explain()`, `explain_fawp()`, `explain_oats()`, `explain_control_cliff()`
  - Severity indicators (✅/⚠️/🔴/🚨), key numbers, suggested actions
- `notebooks/oats_demo.ipynb` — Colab-ready 9-section walkthrough of E1-E7

### Fixed
- PyPI project links now live (version bump from 0.5.0 triggers URL activation)

---

## [0.5.0] — 2026-03-11

### Added
- Full E1-E7 experimental suite bundled in `fawp_index.data`
- `fawp_index.oats` — OATS (Operational Agency Test Suite) module
- `fawp_index.capacity` — agency capacity analysis
- `ControlCliff` / `ControlCliffResult` in `fawp_index.simulate`
- Repository cleanup: examples/, notebooks/, tests/ structure

---

## [0.4.0] — 2026-03-11

### Added
- `notebooks/fawp_demo.ipynb` — comprehensive Jupyter walkthrough
- `MultivariateFAWP`, `MultivariateFAWPResult`
- `FAWPSimulator` with full simulation engine
- README overhaul with quick-start, API reference, citation block

---

## [0.3.0] — 2026-03-11

### Added
- Quant finance suite (`fawp_index.finance`)
- Data science APIs: `fawp_from_dataframe`, `fawp_rolling`
- `FAWPTransformer` (scikit-learn compatible)
- `FAWPFeatureImportance`

---

## [0.2.0] — 2026-03-11

### Added
- Visualization module (`fawp_index.viz`)
- CLI (`fawp-index` command)
- Real data feeds
- Bundled E8 data (`fawp_index.data.E8_DATA`)

---

## [0.1.0] — 2026-03-11

### Added
- Core alpha index (`FAWPAlphaIndex`, `FAWPResult`)
- CSV loader
- Live stream detector (`FAWPStreamDetector`)
- MI estimators (`mi_from_arrays`, `null_corrected_mi`)
