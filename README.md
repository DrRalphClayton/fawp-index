# fawp-index

**FAWP Alpha Index v2.1** — Python implementation of the Information-Control Exclusion Principle detector.

Based on research by **Ralph Clayton** (2026):
- [Agency Horizon paper](https://doi.org/10.5281/zenodo.18663547)
- [FAWP confirmation suite (E8)](https://doi.org/10.5281/zenodo.18673949)

---

## What is FAWP?

**Future Access Without Presence (FAWP)** is the condition where:
- Predictive coupling **persists** — you can still forecast the future
- Steering coupling has **collapsed** — you can no longer influence it

This is the *Information-Control Exclusion Principle*: in unstable regimes, prediction and control are conjugate variables. High predictive certainty is not a measure of mastery — it is a leading indicator of control failure.

---

## Installation

```bash
pip install fawp-index
```

Or from source:
```bash
git clone https://github.com/ralphclayton/fawp-index
cd fawp-index
pip install -e .
```

---

## Quick Start

### From a CSV file

```python
from fawp_index import FAWPAlphaIndex
from fawp_index.io.csv_loader import load_csv_simple

# Load data (only needs state + action columns)
data = load_csv_simple(
    "market_data.csv",
    state_col="price",
    action_col="trade_size",
    delta_pred=20,
)

# Run FAWP Alpha Index
detector = FAWPAlphaIndex(eta=1e-4, epsilon=1e-4, m_persist=5)
result = detector.compute(
    pred_series=data.pred_series,
    future_series=data.future_series,
    action_series=data.action_series,
    obs_series=data.obs_series,
)

print(result.summary())
```

### Live data stream

```python
from fawp_index import FAWPStreamDetector

def alert(result):
    print(f"⚠️  FAWP REGIME DETECTED at tau={result.peak_tau}, alpha={result.peak_alpha:.4f}")

detector = FAWPStreamDetector(
    window=500,
    delta_pred=20,
    on_fawp=alert,
)

# Feed data points as they arrive
for state, action in live_feed:
    detector.update(state=state, action=action)
```

---

## Output

`FAWPResult` contains:

| Field | Description |
|---|---|
| `tau` | Delay grid |
| `alpha_index` | FAWP Alpha Index v2.1 at each tau |
| `in_fawp` | Boolean array — True where FAWP regime detected |
| `tau_h` | Empirical agency horizon |
| `peak_alpha` | Maximum alpha index value |
| `peak_tau` | Delay at peak alpha |
| `pred_mi_corrected` | Null-corrected predictive MI |
| `steer_mi_corrected` | Null-corrected steering MI |

---

## Applications

- **Financial systems** — detect when forecast signal persists after execution leverage collapses
- **Weather prediction** — flag when forecast certainty arrives after intervention window closes
- **Seismic monitoring** — quantify predictive coupling vs zero steering (no earthquake off-switch)
- **Control systems** — early warning of impending control failure via resonance spike

---

## Citation

```bibtex
@misc{clayton2026fawp,
  author = {Clayton, Ralph},
  title  = {Future Access Without Presence (FAWP)},
  year   = {2026},
  doi    = {10.5281/zenodo.18673949},
}
```

---

## License

MIT © Ralph Clayton 2026
