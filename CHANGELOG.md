# Changelog

All notable changes to fawp-index are documented here.

---

## [0.15.0] — 2026-03-14

### Highlights
- Dashboard UX overhaul — severity pills, sparklines, ODW bars, leaderboard, explain cards
- Plotly interactive figures via `pip install "fawp-index[plotly]"`
- Slack alerts with Block Kit messages
- Alert message templates — customise per alert type
- 8 benchmark cases (was 5) — gradual fade, multi-regime, spiky false-positive
- Pre-commit hooks + expanded ruff config
- GitHub issue templates (bug / detection / feature)
- Attribution functions — `attribute_gap()`, `attribute_windows()`, `attribution_report()`
- Calibration constants centralised in `fawp_index.constants`

### Migration from 0.12.x
- `AlertEngine` state file format migrated automatically (old bool → new dict).
  No manual action needed.
- `run_all()` now returns 8 cases instead of 5. Update any hardcoded `== 5` assertions.
- `explain()` dispatcher now handles `AssetResult` and `dict` — previously only
  handled `FAWPResult`, `OATSResult`, `ControlCliffResult`, `SimulationResult`.
- `FAWPSimulator` default `n_trials` changed 100 → 400 to match E8 flagship.
  Pass `n_trials=100` explicitly if you need the old fast behaviour.

### Migration from 0.11.x
- Dashboard `app.py` fully rewritten (946 lines). Copy from latest zip if you
  have a local deployment.
- `fawp_index.constants` is new — all calibration anchors now importable directly:
  `from fawp_index.constants import FLAGSHIP_A, EPSILON_STEERING_RAW, ...`


### Added

- **`fawp_index.viz.plotly_plots`** — interactive Plotly figures (install with `pip install "fawp-index[plotly]"`):
  `plot_mi_curves()`, `plot_regime_score()`, `plot_leverage_gap_bar()`, `plot_heatmap()`, `plot_leaderboard()` — all dark-themed, full hover tooltips
- **3 new benchmark cases** in `fawp_index.benchmarks`:
  - `gradual_fade()` — case 6, FAWP expected, linear steering decay (tests slow-crowding pattern)
  - `multi_regime()` — case 7, FAWP expected, two episodes + recovery (non-monotone trajectory)
  - `spiky_false_positive()` — case 8, FAWP NOT expected, transient dips from news/volatility events
  - All 8 cases now run via `run_all()`
- **`_SlackBackend`** in `alerts.py` — rich Slack Block Kit messages with severity colour, 4 key fields, paper link; wire with `engine.add_slack(webhook_url)`
- **Alert templates** — `_DEFAULT_TEMPLATES` dict + `engine.set_template(alert_type, template)`. Template fields: `{ticker}`, `{timeframe}`, `{score}`, `{gap}`, `{odw}`, `{severity}`, `{alert_type}`, `{timestamp}`, `{version}`
- **`pyproject.toml`** — `plotly = ["plotly>=5.0"]` optional extra added; included in `all`
- **GitHub Release workflow** — `publish.yml` now creates a GitHub Release with CHANGELOG notes and `.whl`/`.tar.gz` attachments on every version tag

---

## [0.12.0] — 2026-03-14

### Added

- **Dashboard full UI upgrade** (`dashboard/app.py` rewritten, 946 lines):
  - Severity pills (FAWP / HIGH / WATCH / CLEAR) with pulsing crimson dot for active regimes
  - Sparkline 6-bar score trend per asset with ▲/▼/— arrow
  - ODW proportional bar showing window position within τ range
  - Score colour-coding by severity tier
  - Filter bar: All / FAWP only / Watching / Rising (client-side, no re-scan)
  - Inline "Why flagged?" expanders using `explain_asset()` for FAWP and HIGH assets
  - Mini leaderboard 2×2 grid in Scanner tab via `Leaderboard.from_watchlist()`
  - Scan metadata line (timestamp, duration, ε, window, τmax)
  - Download leaderboard button; leaderboard export (HTML/CSV/JSON) in Export tab
  - Heatmap DRY refactor (`_heatmap()` helper)
- **`dashboard/.streamlit/config.toml`** — dark theme + `server.headless=true` + `gatherUsageStats=false`
- **`dashboard/README.md`** — Streamlit Cloud deploy instructions
- **`README.md`** full rewrite — live demo link, PyPI downloads badge, CI badge, Python 3.9–3.12 badge, benchmarks quickstart with output, `explain_asset` example, leaderboard, saved watchlists, calibration constants section
- **`docs/examples.md`** — 10 runnable end-to-end examples covering all major use cases

---

## [0.11.0] — 2026-03-14

### Added

- **`fawp_index.leaderboard`** — `Leaderboard.from_watchlist(result)` → 4 ranked categories: `top_fawp`, `rising_risk`, `collapsing_control`, `strongest_odw`. HTML/CSV/JSON export.
- **`fawp_index.watchlist_store`** — persistent named watchlists at `~/.fawp/watchlists.json` (or `$FAWP_STORE`). `store.create()`, `store.scan()`, `store.list()`, `store.show()`, `store.delete()`
- **`fawp_index.watchlist_cli`** — `fawp-watchlist` CLI: `create / scan / list / show / delete` subcommands; `scan` supports `--leaderboard`, `--explain`, `--out`
- **`explain_asset()`** added to `fawp_index.explain` — plain-English "Why flagged?" card with score (0–100), coupling tiers, gap assessment, bulleted reasons, recommendation
- **`AlertEngine` upgraded**: `cooldown_hours`, `min_consecutive_windows`, `score_change_threshold`, `min_severity`
- **`AlertSeverity` enum** (LOW/MEDIUM/HIGH/CRITICAL) added to all `FAWPAlert` objects
- Alert state file auto-migrates from old boolean format
- **`fawp-watchlist`** entry point added to `pyproject.toml`
- **`fawp-scan`** upgraded: `--leaderboard`, `--leaderboard-out`, `--explain` flags

---

## [0.10.0] — 2026-03-14

### Added

- **`fawp_index.constants`** — single source of truth for all paper-derived calibration anchors, imported by 7 core modules:
  `EPSILON_STEERING_RAW=0.01`, `EPSILON_STEERING_CORRECTED=1e-4`, `BETA_NULL_QUANTILE=0.99`,
  `PERSISTENCE_WINDOW_M=5`, `PERSISTENCE_RULE_M=3`, `PERSISTENCE_RULE_N=4` (E9.1 confirmed),
  `FLAGSHIP_A=1.02`, `FLAGSHIP_K=0.8`, `TAU_PLUS_H_E9=31`, `TAU_F_E9=36`, `ODW_START_E9=31`,
  `ODW_END_E9=33`, `PEAK_GAP_BITS_E9=1.55`
- All 7 core modules now import from constants: `alpha_index.py`, `alpha_v2.py`, `estimators.py`, `odw.py`, `market.py`, `simulate.py`, `benchmarks.py`
- **`simulate.py`** `n_trials` default: 100 → 400 (E8 flagship)
- **`PRED_AT_CLIFF`** typo fixed: 1.1010 → 1.0110 (digits were transposed)

### Fixed

- Lint cleanup: 111 → ~17 ruff issues
  - Removed unused imports across 18 files (`field`, `warnings`, `np`, `timedelta`, `scan_fawp_market`, typing imports)
  - Removed unused local `last_alert_time` in `alerts.py`
  - Semicolons expanded, fake f-strings fixed, lambda→def in `features.py`
- `cli.py` — 22 broken `{{...}}` f-strings fixed; `main()` parsed args twice → once
- `dashboard/app.py` footer — was a plain string not an f-string; version never printed
- `__init__.py` — 13 missing names added to `__all__`

---

## [0.9.0] — 2026-03-13

### Added

**`fawp_index.market` — Rolling FAWP Market Scanner**

`FAWPMarketScanner` / `scan_fawp_market()` — rolling-window FAWP detection
on financial price (and optionally volume) DataFrames.

Financial interpretation: pred channel = `I(return_t; return_{t+Δ})` (forecastability),
steer channel = `I(signed_flow_t; return_{t+τ})` (market-impact effectiveness).
A FAWP window means you can still forecast direction but your orders no longer move price.

Usage::

    from fawp_index.market import scan_fawp_market

    df = pd.read_csv("SPY.csv", parse_dates=["Date"], index_col="Date")

    # Fast (no null correction):
    scan = scan_fawp_market(df, ticker="SPY")

    # Rigorous:
    scan = scan_fawp_market(df, ticker="SPY", n_null=50)

    print(scan.summary())
    scan.plot(prices=df["Close"])
    scan.to_html("spy_fawp.html")
    scan.to_csv("spy_fawp.csv")
    scan.to_json("spy_fawp.json")

Key classes: `MarketScanConfig`, `MarketWindowResult`, `MarketScanSeries`

`MarketScanSeries` attributes: `windows`, `dates`, `regime_scores`, `fawp_flags`,
`fawp_fraction`, `.latest`, `.peak`, `.fawp_windows`

`MarketWindowResult` attributes: `date`, `fawp_found`, `regime_score`, `odw_result`,
`pred_mi`, `steer_mi`, `pred_mi_raw`, `steer_mi_raw`

Supports: price-only (no volume), custom pred/steer columns, `date_col` param,
null correction (`n_null` > 0), log or simple returns, full HTML/JSON/CSV export.

Also fixed: removed dead `FAWPStreamDetector` stub (was broken since v0.6.0).
Replaced `stream/live.py` and `viz/plots.py` with thin delegation shims.

# Changelog — fawp-index

All notable changes to this project are documented here.
Versions follow [Semantic Versioning](https://semver.org/).

---

## [0.8.0] — 2026-03-12

### Added

**`fawp_index.significance` — Significance testing**

Three entry points depending on available data:

- `fawp_significance(odw)` — one-call convenience, auto-selects method
- `FAWPSignificance().from_seed_curves(odw)` — bootstrap from bundled E9.2 20-seed CSV (fastest, no extra data needed)
- `FAWPSignificance().from_mi_curves(odw, tau, pred_raw, steer_raw, fail_rate)` — permutation test from pre-computed raw MI arrays
- `FAWPSignificance().from_arrays(odw, tau, pred_pairs, steer_pairs, fail_rate)` — full shuffle+shift null from raw (x,y) paired arrays

Output (`SignificanceResult`): `p_value_fawp`, `p_value_null`, `significant`, `ci_tau_h`, `ci_odw_start`, `ci_odw_end`, `ci_peak_gap`, `ci_peak_gap_tau`, per-tau p-values, bootstrap sample arrays, `.summary()`, `.to_json()`, `.to_html()` (with embedded histograms), `.plot()`

E9.2 verified: p_fawp=1.000, p_null=0.145, significant=True, ci_tau_h=[31,31], ci_peak_gap=[1.538, 1.562]

**`fawp_index.compare` — Side-by-side comparison**

`compare_fawp(result_a, result_b, label_a, label_b)` — auto-detects ODWResult, AlphaV2Result, or BenchmarkResult.
Output (`ComparisonResult`): per-field comparison table, winner per field, overall winner, scores, `.summary()`, `.to_json()`, `.to_html()` (with embedded bar chart), `.plot()`, `.to_pdf()`

Usage::

    from fawp_index import ODWDetector, fawp_significance, compare_fawp

    odw = ODWDetector.from_e9_2_data()

    # Significance
    sig = fawp_significance(odw)
    print(sig.summary())
    sig.to_html("significance.html")

    # Comparison
    odw_xi = ODWDetector.from_e9_2_data(steering='xi')
    cmp = compare_fawp(odw, odw_xi, label_a='u', label_b='xi')
    print(cmp.summary())
    cmp.to_html("comparison.html")

## [0.7.0] — 2026-03-12

### Added

**`fawp_index.benchmarks` — Synthetic Benchmark Suite**

Five canonical ground-truth cases covering the full detection landscape:

- `clean_control()` — textbook FAWP: steering collapses, prediction survives (FAWP expected)
- `prediction_only()` — predictable system with no steering channel (FAWP NOT expected)
- `control_only()` — active controller but no predictive horizon (FAWP NOT expected)
- `noisy_false_positive()` — noisy stable system designed to trap detectors (FAWP NOT expected)
- `delayed_collapse()` — fast-collapsing unstable system, narrow ODW (FAWP expected)

All five run in < 1 second (analytic curves, no simulation).  Pass ``simulate=True``
to run real FAWPSimulator for each case instead.

Usage::

    from fawp_index.benchmarks import run_all

    suite = run_all()
    print(suite.summary())          # pass/fail table
    suite.verify_all()              # raises BenchmarkFailure if any case fails
    suite.to_html("bench.html")     # self-contained HTML report with charts
    suite.to_json("bench.json")     # JSON export

    # One case at a time
    from fawp_index import clean_control, delayed_collapse
    clean_control().verify()
    delayed_collapse().plot()

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

