# fawp-index

[![PyPI version](https://badge.fury.io/py/fawp-index.svg)](https://badge.fury.io/py/fawp-index)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18673949.svg)](https://doi.org/10.5281/zenodo.18673949)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DrRalphClayton/fawp-index/blob/main/notebooks/fawp_demo.ipynb)

---

## Your model still predicts. But you've already lost control.

**fawp-index** detects the moment when a system crosses into the *Information-Control Exclusion Principle* regime — where **predictive information persists** but **the ability to act on it has collapsed**.

This happens more than you think:

| Domain | What predicts | What collapses |
|--------|--------------|----------------|
| 📈 Quant finance | Factor alpha signal | Market execution edge (crowding) |
| 🌀 Dynamical systems | State forecasts | Stabilizing control authority |
| 🌊 Weather / climate | Forecast skill | Intervention window |
| 🌍 Seismic | Precursor signal | Stress release control |
| 🤖 ML systems | Model predictions | Ability to retrain / intervene |

This is not a failure of prediction. It is a structural decoupling — and it has a precise information-theoretic signature.

---

## Install

```bash
pip install fawp-index                  # core
pip install fawp-index[plot]            # + matplotlib figures
pip install fawp-index[finance]         # + Yahoo Finance loader
pip install fawp-index[all]             # everything
```

---

## 60-second quickstart

```python
import numpy as np
from fawp_index import FAWPAlphaIndex

pred    = np.random.randn(5000)
future  = pred[20:] + np.random.randn(4980) * 0.3
action  = np.random.randn(4980) * 0.001   # near zero = FAWP
obs     = np.random.randn(4980) * 0.1

result = FAWPAlphaIndex().compute(pred[:4980], future, action, obs)
print(result.summary())
result.plot()   # requires: pip install fawp-index[plot]
```

```
==================================================
FAWP Alpha Index v2.1 — Results Summary
==================================================
Agency Horizon (tau_h):  4
Peak Alpha Index:        2.2326
Peak Alpha at tau:       9
FAWP regime detected:   YES
FAWP tau range:          [4, 5, 6, 7, 8, 9, 10, 11, 12]
==================================================
```

---

## Works natively with DataFrames

```python
import pandas as pd
from fawp_index import fawp_from_dataframe

df = pd.read_csv("my_data.csv")

result = fawp_from_dataframe(
    df,
    pred_col   = "factor_score",
    action_col = "trade_size",
    future_col = "forward_return",
)
print(result.summary())
```

### Rolling regime detection

```python
from fawp_index import fawp_rolling

df_annotated = fawp_rolling(df, pred_col="returns", action_col="volume")
df_annotated[df_annotated["fawp_in_regime"]]
```

Adds columns: `fawp_pred_mi`, `fawp_steer_mi`, `fawp_gap`, `fawp_in_regime`

---

## Sklearn compatible

```python
from fawp_index.sklearn_api import FAWPTransformer
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("fawp", FAWPTransformer(pred_col=0, action_col=1, delta=20)),
])
pipe.fit(X)
score = pipe.score(X)   # peak FAWP alpha index
```

---

## Feature importance

```python
from fawp_index.features import FAWPFeatureImportance

fi = FAWPFeatureImportance(action_col="trade_size", delta=21)
result = fi.fit(df, feature_cols=["momentum", "value", "quality", "carry"])
print(result.summary())
result.plot()
```

```
Rank     Feature    Alpha  Pred MI  Steer MI   FAWP
   1    momentum   1.8823   2.1034    0.0003  ✓
   2       carry   0.4211   0.5102    0.0041  ✓
   3       value   0.0000   0.0823    0.1204
   4     quality   0.0000   0.0412    0.2891
2/4 features in FAWP regime
```

---

## Quant finance

```python
from fawp_index.quant import (
    FAWPRegimeDetector,     # rolling market breakdown flag
    MomentumDecayDetector,  # crowded trade detection
    RiskParityWarning,      # vol-targeting failure warning
    EventStudyFAWP,         # pre/post announcement analysis
)

detector = FAWPRegimeDetector(window=252, step=21)
result = detector.detect(returns, volumes)
print(result.summary())
```

---

## Command line

```bash
fawp-index mydata.csv --state price --action trade_size --plot
```

---

## Reproduce published figures

The real E8 experimental data is bundled with the package:

```bash
python examples/reproduce_e8.py --save
```

Exactly reproduces figures from:
> *"Forecasting Without Power: Agency Horizons and the Leverage Gap"*
> Ralph Clayton (2026) · [doi:10.5281/zenodo.18663547](https://doi.org/10.5281/zenodo.18663547)

---

## The mathematics

The **FAWP Alpha Index v2.1**:

```
α₂(τ) = I[τ≥1] · g(τ) · (Sₘ(τ) − Ĩ_steer(τ)) · (1 + κ · R_log(τ))
```

- `g(τ)` — gate: fires when pred MI > η AND steer MI ≤ ε
- `Sₘ(τ)` — windowed-min corrected predictive MI (persistence)
- `Ĩ_steer(τ)` — null-corrected steering MI: I(action_t ; obs_{t+τ+1})
- `R_log(τ)` — log-slope resonance amplifier near the horizon

The **agency horizon τ_h** is where steering MI first falls below ε.
Near τ_h, predictive MI does not fall — it surges. This resonance ridge is the empirical signature of the Information-Control Exclusion Principle.

---

## What this is not

- ❌ Not a forecasting model
- ❌ Not a trading signal
- ✅ A diagnostic: tells you *when* your model is in an irrecoverable information regime

---

## Citation

```bibtex
@software{clayton2026fawpindex,
  author    = {Ralph Clayton},
  title     = {fawp-index: FAWP Alpha Index v2.1},
  year      = {2026},
  url       = {https://github.com/DrRalphClayton/fawp-index}
}

@article{clayton2026agency,
  author = {Ralph Clayton},
  title  = {Forecasting Without Power: Agency Horizons and the Leverage Gap},
  year   = {2026},
  doi    = {10.5281/zenodo.18663547}
}
```

---

## Links

- 📦 **PyPI:** [pypi.org/project/fawp-index](https://pypi.org/project/fawp-index/)
- 📂 **GitHub:** [github.com/DrRalphClayton/fawp-index](https://github.com/DrRalphClayton/fawp-index)
- 📄 **Paper (E1-E7):** [doi:10.5281/zenodo.18663547](https://doi.org/10.5281/zenodo.18663547)
- 📄 **Paper (E8):** [doi:10.5281/zenodo.18673949](https://doi.org/10.5281/zenodo.18673949)
- 📗 **Book:** [*Forecasting Without Power*](https://www.amazon.com/dp/B0GS1ZVNM7/) — Ralph Clayton (2026)

---

*MIT License · Ralph Clayton · 2026*
