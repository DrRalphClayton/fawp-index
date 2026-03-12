"""
fawp_index.capacity — Agency Capacity Surfaces (E6)

Two surfaces mapping how hardware constraints degrade the Agency Horizon:

  Surface A — Quantization cliff:
    Agency requires a minimum bit-depth to exist.
    1-bit agents have near-zero horizon regardless of latency.
    8-bit ≈ 32-bit float for this task.

  Surface B — Erasure slope:
    Network packet loss is thermodynamically equivalent to adding latency.
    50% packet loss cuts the effective horizon roughly in half.

Ralph Clayton (2026) — doi:10.5281/zenodo.18663547
"""
from .surfaces import CapacitySurface, CapacityResult
__all__ = ["CapacitySurface", "CapacityResult"]
