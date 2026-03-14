# fawp_index.significance — Significance Testing

Test whether a detected FAWP regime is statistically significant using
bootstrap or permutation methods.

## Quick start

```python
from fawp_index import ODWDetector, fawp_significance

odw = ODWDetector.from_e9_2_data()
sig = fawp_significance(odw)

print(sig.summary())
# p_fawp=1.000  p_null=0.145  significant=YES
# ci_tau_h=[31,31]  ci_peak_gap=[1.538, 1.562]

sig.to_html("significance.html")
sig.plot()
```

## Three methods

### 1. Seed bootstrap (default — fastest)
```python
sig = fawp_significance(odw, method="seed_bootstrap", n_bootstrap=200)
# Resamples the bundled E9.2 20-seed CSV with replacement.
# No raw data required. Best for quick checks.
# significant = (p_value_fawp >= 1 - alpha)
```

### 2. MI-curve permutation
```python
sig = FAWPSignificance().from_mi_curves(
    odw, tau, pred_raw, steer_raw, fail_rate
)
# Permutes pre-computed raw MI arrays.
# significant = (p_value_null <= alpha)
```

### 3. Full array permutation
```python
sig = FAWPSignificance().from_arrays(
    odw, tau, pred_pairs, steer_pairs, fail_rate
)
# Shuffle + shift null from raw (x, y) paired arrays.
# Most rigorous. Requires raw data.
```

## SignificanceResult fields

| Field | Description |
|-------|-------------|
| `p_value_fawp` | Fraction of bootstrap reps that also find FAWP |
| `p_value_null` | Fraction of null reps that exceed observed stat |
| `significant` | Boolean: regime is significant at `alpha` level |
| `ci_tau_h` | Bootstrap CI for agency horizon |
| `ci_odw_start` | Bootstrap CI for ODW start |
| `ci_odw_end` | Bootstrap CI for ODW end |
| `ci_peak_gap` | Bootstrap CI for peak leverage gap (bits) |
| `ci_peak_gap_tau` | Bootstrap CI for tau of peak gap |

## Exports

```python
sig.to_html("sig.html")    # HTML report with embedded histograms
sig.to_json("sig.json")    # Full JSON export
sig.summary()              # Human-readable text summary
sig.plot()                 # Bootstrap distribution plots
```

## Verified E9.2 results (bundled data)

```
p_fawp=1.000   p_null=0.145   significant=YES
ci_tau_h       = [31, 31]
ci_odw         = [31, 33]
ci_peak_gap    = [1.538, 1.562] bits
ci_peak_gap_τ  = [34, 34]
```
