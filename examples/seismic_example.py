"""
fawp-index: Seismic Monitoring Example
=======================================
Detects FAWP regimes in earthquake precursor systems.

The scenario: seismic precursor monitoring before a major event.
Ground deformation, radon emissions, and microseismic activity
can carry predictive information about an impending earthquake.

But the intervention coupling is essentially ZERO.
There is no earthquake off-switch.

This is FAWP in its most extreme and haunting form:
maximum predictive coupling, zero steering coupling.
The Alpha Index stays permanently elevated in the FAWP regime.

Clayton (2026): "The signals were there, but the control channels
were weak. Seeing that leverage was dangerous did not let you make
millions of other people unborrow money."
"""

import numpy as np
import pandas as pd

from fawp_index import FAWPAlphaIndex
from fawp_index.io.csv_loader import load_csv


def generate_seismic_data(
    n_sequences: int = 300,
    n_timesteps: int = 200,
    precursor_window: int = 48,   # hours of precursor signal before event
    intervention_coupling: float = 0.0,  # no earthquake off-switch
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate seismic precursor monitoring.

    - Ground strain accumulates toward failure (predictable pattern)
    - Radon / microseismic activity correlates with future rupture
    - Intervention coupling is effectively zero (no steering possible)
    - FAWP: prediction is real, steering is impossible by physics
    """
    rng = np.random.default_rng(seed)
    rows = []
    delta_pred = 24  # 24h forecast horizon

    for seq_id in range(n_sequences):
        # Stress accumulation: slow ramp + noise
        stress_rate = rng.uniform(0.005, 0.02)
        background_noise = rng.uniform(0.1, 0.3)

        stress = [rng.uniform(0.2, 0.5)]
        radon = []
        microseismic = []

        for t in range(n_timesteps + delta_pred + 5):
            # Stress accumulates
            stress.append(stress[-1] + stress_rate + rng.normal(0, 0.01))
            stress[-1] = np.clip(stress[-1], 0, 2.0)

            # Radon anomaly: rises as stress approaches failure threshold
            radon_signal = 0.3 * stress[-1] ** 2 + rng.normal(0, background_noise)
            radon.append(radon_signal)

            # Microseismic rate: increases before rupture
            mseis = 0.5 * stress[-1] + rng.normal(0, background_noise * 0.5)
            microseismic.append(mseis)

        for t in range(n_timesteps - delta_pred):
            # D_t: composite precursor signal (radon + microseismic)
            d_t = 0.6 * radon[t] + 0.4 * microseismic[t]

            # X_{t+delta}: future stress level (earthquake proxy)
            x_future = stress[t + delta_pred]

            # "Action": any attempted intervention (fracking stop, dam release)
            # In reality these have near-zero coupling to major earthquakes
            action = rng.normal(0, 0.01)  # tiny, random, ineffective

            # O_{t+tau+1}: observed stress response to "intervention"
            # Coupling is near-zero — physics prevents steering
            obs = stress[t + 1] * intervention_coupling + rng.normal(0, background_noise)

            rows.append({
                "precursor_signal": d_t,
                "future_stress": x_future,
                "intervention_action": action,
                "stress_response": obs,
                "stress_level": stress[t],
                "radon": radon[t],
                "microseismic": microseismic[t],
                "seq_id": seq_id,
                "hour": t,
            })

    return pd.DataFrame(rows)


# ── Run Seismic Example ───────────────────────────────────────────────────────
print("=" * 60)
print("FAWP Alpha Index — Seismic Monitoring Example")
print("=" * 60)
print("\nScenario: Earthquake precursor monitoring")
print("  - Radon + microseismic signal correlates with future rupture")
print("  - Intervention coupling = ~0 (no earthquake off-switch)")
print("  - FAWP = prediction is real, steering is physically impossible\n")
print("  This is FAWP in its most extreme form.")
print("  The haunting case: you can see it. You cannot stop it.\n")

print("Generating seismic sequence data...")
df = generate_seismic_data(n_sequences=250, n_timesteps=180, seed=42)
csv_path = "seismic_fawp_output.csv"
df.to_csv(csv_path, index=False)
print(f"Generated {len(df):,} rows across {df['seq_id'].nunique()} sequences")

data = load_csv(
    csv_path,
    pred_col="precursor_signal",
    future_col="future_stress",
    action_col="intervention_action",
    obs_col="stress_response",
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

print("\n🌍 SEISMIC INTERPRETATION:")
if result.in_fawp.any():
    fawp_taus = list(result.tau[result.in_fawp])
    print(f"   FAWP detected across tau={fawp_taus}")
    print(f"   Peak Alpha Index = {result.peak_alpha:.4f} at tau={result.peak_tau}")
    print(f"   Predictive MI = {result.pred_mi_raw[result.tau == result.peak_tau][0]:.4f} bits")
    print(f"   Steering MI  = {result.steer_mi_raw[result.tau == result.peak_tau][0]:.4f} bits")
    print(f"\n   → Precursor signals carry REAL information about future rupture")
    print(f"   → But intervention coupling is physically near-zero")
    print(f"   → Maximum FAWP: prediction persists, steering is impossible")
    print(f"   → The Alpha Index quantifies exactly HOW trapped you are")
else:
    print("   Checking raw MI...")
    print("   Pred MI:", np.round(result.pred_mi_raw[:8], 3))
    print("   Steer MI:", np.round(result.steer_mi_raw[:8], 3))

# Show the leverage gap numbers directly
print(f"\n   Leverage gap (pred - steer) across tau 1-8:")
for i in range(min(8, len(result.tau))):
    tau = result.tau[i]
    gap = result.pred_mi_raw[i] - result.steer_mi_raw[i]
    flag = " ← FAWP" if result.in_fawp[i] else ""
    print(f"   tau={tau:2d}: pred={result.pred_mi_raw[i]:.4f}  "
          f"steer={result.steer_mi_raw[i]:.4f}  gap={gap:.4f}{flag}")

print("\n✅ Seismic example complete\n")
