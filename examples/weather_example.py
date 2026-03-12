"""
fawp-index: Weather Forecasting Example
========================================
Detects FAWP regimes in atmospheric forecast systems.

The scenario: a hurricane track forecast system.
Forecast skill (predictive coupling) increases as the storm
organizes and becomes more predictable. But intervention windows
(evacuation logistics, storm surge barriers) close on fixed timelines.

FAWP appears when forecast confidence arrives AFTER the
practical intervention window has closed — the defining
tragedy of modern disaster management.

Clayton (2026): "The map gets brighter, but the steering does not
improve at the same rate."
"""

import numpy as np
import pandas as pd

from fawp_index import FAWPAlphaIndex
from fawp_index.io.csv_loader import load_csv


def generate_weather_data(
    n_storms: int = 300,
    n_timesteps: int = 120,   # hourly steps, 5-day storm
    forecast_skill_buildup: int = 24,  # hours until forecast becomes reliable
    intervention_window: int = 18,     # hours before landfall for effective action
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate hurricane track forecasting with intervention timing.

    - Storm track follows a noisy AR process with slow drift
    - Forecast model improves as storm organizes (skill builds up over time)
    - Intervention (evacuation) must happen >18h before landfall
    - FAWP: forecast gets good AFTER intervention window closes
    """
    rng = np.random.default_rng(seed)
    rows = []
    delta_pred = 24  # 24-hour forecast horizon

    for storm_id in range(n_storms):
        # Storm track: position (lat/lon proxy as scalar)
        intensity = rng.uniform(0.5, 1.5)  # storm intensity multiplier
        track = [rng.normal(0, 1)]
        forecast_errors = []

        for t in range(n_timesteps + delta_pred + intervention_window + 5):
            drift = intensity * 0.02
            noise = rng.normal(0, 0.3 * intensity)
            track.append(track[-1] + drift + noise)

        # Forecast skill: poor early, improves as storm organizes
        for t in range(n_timesteps - delta_pred):
            skill_factor = min(1.0, t / forecast_skill_buildup)

            # D_t: current track position (monitoring stream)
            d_t = track[t]

            # X_{t+delta}: future track position (forecast target)
            x_future = track[t + delta_pred]

            # Forecast model action: weighted combination of current + trend
            trend = track[t] - track[max(0, t-3)]
            forecast_action = skill_factor * (d_t + delta_pred * trend * 0.1)
            obs_noise = rng.normal(0, 0.5 * (1 - skill_factor * 0.8))
            action = forecast_action + obs_noise

            # Intervention coupling: ability to change outcomes
            # Decays sharply as we approach landfall (fixed deadline)
            time_to_landfall = n_timesteps - t
            # Intervention is only effective if > intervention_window hours out
            intervention_possible = time_to_landfall > intervention_window
            intervention_coupling = (
                rng.normal(0.8, 0.1) if intervention_possible
                else rng.normal(0.0, 0.05)  # near-zero after window closes
            )

            # O_{t+tau+1}: delayed observation of track response to intervention
            obs = track[t + 1] * intervention_coupling + rng.normal(0, 0.2)

            rows.append({
                "track_position": d_t,
                "future_track": x_future,
                "forecast_action": action,
                "intervention_obs": obs,
                "skill_factor": skill_factor,
                "time_to_landfall": time_to_landfall,
                "intervention_open": int(intervention_possible),
                "storm_id": storm_id,
                "hour": t,
            })

    return pd.DataFrame(rows)


# ── Run Weather Example ───────────────────────────────────────────────────────
print("=" * 60)
print("FAWP Alpha Index — Weather Forecasting Example")
print("=" * 60)
print("\nScenario: Hurricane track forecast vs intervention window")
print("  - Forecast skill builds up over first 24 hours")
print("  - Evacuation window closes 18h before landfall")
print("  - FAWP = forecast gets good AFTER you can still evacuate\n")

print("Generating storm track simulation...")
df = generate_weather_data(n_storms=250, n_timesteps=100, seed=42)
csv_path = "weather_fawp_output.csv"
df.to_csv(csv_path, index=False)
print(f"Generated {len(df):,} rows across {df['storm_id'].nunique()} simulated storms")

data = load_csv(
    csv_path,
    pred_col="track_position",
    future_col="future_track",
    action_col="forecast_action",
    obs_col="intervention_obs",
    delta_pred=24,
)

detector = FAWPAlphaIndex(
    eta=1e-4,
    epsilon=1e-4,
    m_persist=3,
    kappa=1.0,
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

print("\n🌀 WEATHER INTERPRETATION:")
if result.in_fawp.any():
    print(f"   FAWP detected — forecast skill persists at tau={result.peak_tau}h")
    print(f"   But intervention coupling has collapsed")
    print(f"   Alpha Index = {result.peak_alpha:.4f}")
    print(f"   → Track forecast is informative but intervention window is closed")
    print(f"   → Classic disaster management FAWP: late clarity, closed window")
else:
    print("   Pred MI range:", np.round(result.pred_mi_raw, 3))
    print("   Steer MI range:", np.round(result.steer_mi_raw, 3))
    print("   Leverage gap:", np.round(result.pred_mi_raw - result.steer_mi_raw, 3))

print("\n✅ Weather example complete\n")
