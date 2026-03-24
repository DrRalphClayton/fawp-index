"""
fawp-index: Financial Markets Example
=====================================
Detects FAWP regimes in a simulated trading system.

The scenario: a momentum trader using delayed price signals.
As market microstructure noise increases with latency (slippage,
execution delay, feedback lag), steering coupling collapses while
predictive coupling from price momentum persists.

This is the "you can see the crash coming but can't stop it" regime
described in Clayton (2026) Forecasting Without Power.
"""
import sys, os as _os
# Allow running from repo root OR from examples/ directory
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd

from fawp_index import FAWPAlphaIndex
from fawp_index.io.csv_loader import load_csv


def generate_financial_data(
    n_steps: int = 2000,
    n_trials: int = 300,
    drift: float = 0.001,
    volatility: float = 0.02,
    momentum: float = 0.15,
    execution_delay: int = 4,
    slippage_rate: float = 0.001,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate a momentum-driven price process with delayed execution.

    Price follows: P_{t+1} = P_t * exp(drift + momentum*signal_t + noise)
    Trader observes delayed signal, executes with slippage.
    Observation noise grows with delay (microstructure friction).
    """
    rng = np.random.default_rng(seed)
    rows = []

    for trial in range(n_trials):
        price = 100.0
        signal_buf = [0.0] * (execution_delay + 2)
        prices = [price]

        # Generate price path
        for t in range(n_steps):
            signal = momentum * (prices[-1] / prices[max(0, len(prices)-5)] - 1)
            noise = rng.normal(0, volatility)
            price = price * np.exp(drift + signal + noise)
            price = np.clip(price, 1, 10000)
            prices.append(price)
            signal_buf.append(signal)

        # Build aligned arrays
        delta_pred = 20
        for t in range(execution_delay + 1, n_steps - delta_pred):
            # D_t: current log-return (predictor)
            d_t = np.log(prices[t] / prices[t - 1]) if prices[t-1] > 0 else 0.0

            # X_{t+delta}: future log-return (target)
            x_future = np.log(prices[t + delta_pred] / prices[t + delta_pred - 1])

            # A_t: trading action based on delayed signal
            delayed_signal = signal_buf[t - execution_delay]
            slippage = rng.normal(0, slippage_rate * (1 + execution_delay))
            action = np.clip(delayed_signal + slippage, -0.5, 0.5)

            # O_{t+tau+1}: delayed price observation with execution noise
            obs_noise = rng.normal(0, slippage_rate * execution_delay)
            obs = np.log(prices[t + 1] / prices[t]) + obs_noise

            rows.append({
                "log_return": d_t,
                "future_return": x_future,
                "trade_signal": action,
                "obs_return": obs,
                "trial": trial,
                "step": t,
            })

    return pd.DataFrame(rows)


# ── Run Finance Example ───────────────────────────────────────────────────────
print("=" * 60)
print("FAWP Alpha Index — Financial Markets Example")
print("=" * 60)
print("\nScenario: Momentum trader with execution delay")
print("  - Price momentum creates predictive signal")
print("  - Execution delay + slippage erodes steering coupling")
print("  - FAWP = you can see the move but can't catch it\n")

print("Generating simulated market data...")
df = generate_financial_data(n_steps=800, n_trials=200, seed=42)
csv_path = "finance_fawp_output.csv"
df.to_csv(csv_path, index=False)
print(f"Generated {len(df):,} rows across {df['trial'].nunique()} trials")

# Load and run
data = load_csv(
    csv_path,
    pred_col="log_return",
    future_col="future_return",
    action_col="trade_signal",
    obs_col="obs_return",
    delta_pred=20,
)

detector = FAWPAlphaIndex(
    eta=1e-4,
    epsilon=1e-4,
    m_persist=3,
    kappa=1.5,
    n_null=100,
    beta=0.99,
)

result = detector.compute(
    pred_series=data.pred_series,
    future_series=data.future_series,
    action_series=data.action_series,
    obs_series=data.obs_series,
    tau_grid=list(range(1, 16)),
    verbose=True,
)

print("\n" + result.summary())

# Interpret results
if result.in_fawp.any():
    print("\n📊 FINANCIAL INTERPRETATION:")
    print(f"   Predictive signal persists at tau={result.peak_tau}")
    print(f"   But execution leverage has collapsed")
    print(f"   Alpha Index = {result.peak_alpha:.4f}")
    print(f"   → Classic 'can see the move, can't catch it' regime")
    print(f"   → Signals are real but no longer actionable at these delays")
else:
    print("\n📊 No FAWP regime detected — steering coupling still intact")
    print("   Raw pred MI:", np.round(result.pred_mi_raw[:8], 3))
    print("   Raw steer MI:", np.round(result.steer_mi_raw[:8], 3))


# ── Live stream: real-time FAWP alert for a trading feed ─────────────────────
print("\n--- Live Trading Feed Simulation ---")
print("Feeding price data tick by tick...\n")

alerts = []

def trading_alert(r):
    alerts.append({
        "step": stream.step,
        "tau_h": r.tau_h,
        "peak_alpha": r.peak_alpha,
        "peak_tau": r.peak_tau,
    })
    if len(alerts) <= 2:
        print(f"  🚨 FAWP ALERT step={stream.step}: "
              f"alpha={r.peak_alpha:.4f} tau={r.peak_tau} "
              f"→ Execution leverage below detectability")

# Rolling-window FAWP scan (replaces removed FAWPStreamDetector)
# FAWPAlphaIndex.compute() takes: pred, future, action, obs, tau_grid, delta_pred
trial_df = df[df["trial"] == 0].reset_index(drop=True)
window    = 600
step      = 100
delta     = 20          # forecast horizon — must match data generation
tau_grid  = list(range(1, 16))
n_steps   = len(trial_df)
roll_results = []

print("\nRolling-window FAWP scan:")
for start in range(0, n_steps - window, step):
    chunk = trial_df.iloc[start : start + window].reset_index(drop=True)
    n     = len(chunk) - delta   # usable rows after aligning future

    idx = FAWPAlphaIndex(n_null=20)
    res = idx.compute(
        pred_series   = chunk["log_return"].values[:n],
        future_series = chunk["future_return"].values[:n],
        action_series = chunk["trade_signal"].values[:n],
        obs_series    = chunk["obs_return"].values[:n],
        tau_grid      = tau_grid,
        delta_pred    = delta,
    )
    if res.fawp_found:
        trading_alert(res)
    roll_results.append(res.fawp_found)

n_fawp = sum(roll_results)
print(f"\nRolling scan complete: {len(roll_results)} windows, {n_fawp} FAWP windows, {len(alerts)} alerts fired")
print("✅ Finance example complete\n")
