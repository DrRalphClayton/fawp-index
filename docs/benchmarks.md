# fawp_index.benchmarks — Benchmark Suite

Five canonical ground-truth cases that verify the detector behaves correctly
across the full detection landscape.

## Quick start

```python
from fawp_index import run_benchmarks

suite = run_benchmarks()
print(suite.summary())       # pass/fail table
suite.verify_all()           # raises BenchmarkFailure if any case fails
suite.to_html("bench.html")
suite.to_json("bench.json")
```

## CLI

```bash
fawp-index benchmarks
fawp-index benchmarks --verify          # exit 1 if any fail
fawp-index benchmarks --out bench.html
```

## The five cases

| Case | FAWP expected? | Description |
|------|:--------------:|-------------|
| `clean_control` | ✅ Yes | Textbook FAWP: steering collapses, prediction survives |
| `prediction_only` | ❌ No | Predictable system, no steering channel |
| `control_only` | ❌ No | Active controller, no predictive horizon |
| `noisy_false_positive` | ❌ No | Noisy stable system designed to trap detectors |
| `delayed_collapse` | ✅ Yes | Fast-collapsing unstable system, narrow ODW |

All five run in < 1 second (analytic curves, no simulation).

## Run individual cases

```python
from fawp_index import clean_control, delayed_collapse

case = clean_control()
case.verify()          # raises if wrong result
case.plot()            # leverage gap plot

case2 = delayed_collapse()
print(case2.result.summary())
```

## Simulate instead of analytic curves

```python
from fawp_index.benchmarks import clean_control
case = clean_control(simulate=True)   # runs FAWPSimulator
```

## BenchmarkResult fields

| Field | Description |
|-------|-------------|
| `name` | Case name |
| `passed` | Boolean |
| `expected_fawp` | Expected detection outcome |
| `actual_fawp` | Actual detection outcome |
| `result` | Full `ODWResult` |
| `notes` | Human-readable description |
