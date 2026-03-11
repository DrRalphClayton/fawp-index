"""
quant_finance_notebook.py
=========================
Quant finance and data science examples for fawp-index.
Run as a script or convert to Jupyter: jupyter nbconvert --to notebook quant_finance_notebook.py

Sections:
  1. DataFrame API               — pass df directly, get results back
  2. Regime Detection            — rolling FAWP flags on synthetic equity data
  3. Momentum Decay Scanner      — signal survives, execution edge gone
  4. Risk Parity Warning         — FAWP before vol spikes
  5. Feature Importance          — rank predictors by FAWP score
  6. Sklearn Pipeline            — FAWP inside a sklearn workflow

Ralph Clayton (2026) — doi:10.5281/zenodo.18673949
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

print("fawp-index: Quant Finance & Data Science Examples")
print("=" * 55)

# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA — realistic equity-style series
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(42)
n = 3000
dates = pd.date_range('2018-01-01', periods=n, freq='B')

# Price process: mean-reverting + trending regimes
returns = np.random.normal(0.0003, 0.015, n)
returns[1000:1500] += 0.002   # trending regime
returns[1500:2000] -= 0.001   # drawdown
prices = 100 * np.exp(np.cumsum(returns))

# Volume: correlated with abs returns, spikes during stress
volume = np.abs(returns) * 1e7 + np.random.exponential(5e5, n)
volume[1500:2000] *= 2.5      # volume spike during stress

# Factor signal: momentum score (rolling 60d return z-score)
momentum = pd.Series(returns).rolling(60).mean() / (pd.Series(returns).rolling(60).std() + 1e-10)
momentum = momentum.fillna(0).values

# Forward returns (20-day)
fwd_returns = pd.Series(returns).shift(-20).fillna(0).values

df = pd.DataFrame({
    'date': dates,
    'price': prices,
    'returns': returns,
    'volume': volume,
    'momentum': momentum,
    'fwd_return_20d': fwd_returns,
    'volatility': pd.Series(returns).rolling(20).std().fillna(0.015).values,
    'volume_change': pd.Series(volume).pct_change().fillna(0).values,
    'price_acceleration': pd.Series(returns).diff().fillna(0).values,
}, index=dates)

print(f"\nSynthetic dataset: {n} trading days ({dates[0].date()} → {dates[-1].date()})")
print(f"Columns: {list(df.columns)}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATAFRAME API
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 55)
print("1. DataFrame API")
print("─" * 55)

from fawp_index.sklearn_api import FAWPDataFrame

fawp_df = FAWPDataFrame(n_null=50, tau_grid=list(range(1, 12)))
result = fawp_df.compute(
    df,
    pred_col='momentum',
    future_col='fwd_return_20d',
    action_col='volume_change',
)

print(result)
print("\nPer-tau breakdown:")
print(result.summary_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 2. REGIME DETECTION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 55)
print("2. Rolling Regime Detection")
print("─" * 55)

from fawp_index.finance.quant import FAWPRegimeDetector

detector = FAWPRegimeDetector(
    window=252,
    step=21,
    tau_grid=list(range(1, 8)),
    delta_pred=20,
    n_null=50,
)
regime = detector.detect(
    prices=df['price'],
    volume=df['volume'],
)
print(regime.summary())
regime_df = regime.to_dataframe()
print(f"\nFirst 5 windows:\n{regime_df.head().to_string(index=False)}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. MOMENTUM DECAY SCANNER
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 55)
print("3. Momentum Decay Scanner")
print("─" * 55)

from fawp_index.finance.quant import MomentumDecayScanner

scanner = MomentumDecayScanner(
    tau_grid=list(range(1, 12)),
    delta_pred=20,
    n_null=50,
)
decay = scanner.scan(
    signal=pd.Series(momentum),
    future_returns=pd.Series(fwd_returns),
    trade_size=pd.Series(volume / volume.max()),
)
print(decay.summary())

# ─────────────────────────────────────────────────────────────────────────────
# 4. RISK PARITY WARNING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 55)
print("4. Risk Parity Warning")
print("─" * 55)

from fawp_index.finance.quant import RiskParityWarning

warner = RiskParityWarning(
    window=60,
    step=10,
    vol_threshold=1.5,
    tau_grid=list(range(1, 6)),
    n_null=50,
)
warning = warner.warn(
    returns=df['returns'],
    volume=df['volume'],
)
print(warning.summary())

# ─────────────────────────────────────────────────────────────────────────────
# 5. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 55)
print("5. Feature Importance")
print("─" * 55)

from fawp_index.sklearn_api import FAWPFeatureSelector

selector = FAWPFeatureSelector(
    target_col='fwd_return_20d',
    action_col='volume_change',
    tau_grid=list(range(1, 8)),
    delta_pred=20,
    n_null=50,
    top_k=5,
)
importance = selector.fit(df)
print(importance.summary())
print(f"\nTop 5 selected features: {selector.selected_features_}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. SKLEARN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 55)
print("6. Sklearn-style Interface")
print("─" * 55)

from fawp_index.sklearn_api import FAWPTransformer

# Build feature matrix: pred, future, action, obs
X = df[['momentum', 'fwd_return_20d', 'volume_change', 'returns']].values

transformer = FAWPTransformer(
    pred_col=0,
    future_col=1,
    action_col=2,
    obs_col=3,
    tau_grid=list(range(1, 8)),
    n_null=50,
    output='all',
)

scores = transformer.fit_transform(X)
peak_score = transformer.score(X)
print(f"Transformer output shape: {scores.shape}")
print(f"Columns: tau | pred_mi | steer_mi | alpha_index")
print(f"Peak FAWP score: {peak_score:.4f}")
print(f"FAWP detected: {transformer.result_.in_fawp.any()}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT — Combined dashboard
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating dashboard plot...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "fawp-index: Quant Finance Dashboard\nClayton (2026) doi:10.5281/zenodo.18673949",
    fontsize=12, fontweight='bold'
)

# Panel 1: Leverage gap (DataFrame API result)
ax = axes[0, 0]
tau = result.summary_df['tau'].values
ax.plot(tau, result.summary_df['steer_mi_raw'], 'b--', lw=2,
        label='Steering MI (execution edge)')
ax.plot(tau, result.summary_df['pred_mi_raw'], color='darkorange', lw=2,
        label='Predictive MI (alpha signal)')
ax.fill_between(tau, result.summary_df['steer_mi_raw'],
                result.summary_df['pred_mi_raw'],
                where=(result.summary_df['pred_mi_raw'] > result.summary_df['steer_mi_raw']),
                alpha=0.2, color='orange', label='Leverage gap')
ax.set_title('Momentum FAWP Leverage Gap', fontsize=10)
ax.set_xlabel('τ (delay)')
ax.set_ylabel('MI (bits)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 2: Rolling regime
ax = axes[0, 1]
if len(regime_df) > 0:
    ax.fill_between(range(len(regime_df)),
                    regime_df['alpha_score'],
                    alpha=0.5,
                    color=['darkorange' if f else 'steelblue'
                           for f in regime_df['in_fawp']],
                    step='mid')
    ax.plot(regime_df['alpha_score'], 'k-', lw=1.5, label='FAWP Alpha')
    for ch in regime.regime_changes:
        ax.axvline(ch, color='red', lw=1, alpha=0.5, linestyle='--')
ax.set_title(f'Rolling Regime Detection\n({regime.fawp_fraction:.1%} FAWP windows)', fontsize=10)
ax.set_xlabel('Window')
ax.set_ylabel('Peak Alpha')
ax.grid(True, alpha=0.3)

# Panel 3: Momentum decay
ax = axes[1, 0]
ax.plot(decay.tau, decay.execution_mi, 'b--', lw=2, label='Execution MI')
ax.plot(decay.tau, decay.signal_mi, color='green', lw=2, label='Signal MI')
ax.fill_between(decay.tau, decay.execution_mi, decay.signal_mi,
                where=(decay.signal_mi > decay.execution_mi),
                alpha=0.2, color='green', label='Alpha survives')
if decay.decay_point is not None:
    ax.axvline(decay.decay_point, color='red', lw=1.5, linestyle='--',
               label=f'Execution collapse τ={decay.decay_point}')
ax.set_title('Momentum Decay Scan', fontsize=10)
ax.set_xlabel('τ (delay)')
ax.set_ylabel('MI (bits)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 4: Feature importance
ax = axes[1, 1]
imp_df = importance.to_dataframe().head(5)
colors = ['darkorange' if f else 'steelblue' for f in imp_df['fawp_detected']]
ax.barh(imp_df['feature'][::-1], imp_df['fawp_score'][::-1],
        color=colors[::-1], edgecolor='white')
ax.set_title('Feature FAWP Importance (top 5)', fontsize=10)
ax.set_xlabel('FAWP Alpha Score')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('/tmp/quant_dashboard.png', dpi=150, bbox_inches='tight')
print("Saved: /tmp/quant_dashboard.png")

print("\n" + "=" * 55)
print("All quant finance examples completed successfully!")
print("=" * 55)
