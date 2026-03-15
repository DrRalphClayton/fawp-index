"""
fawp_index.core.estimators — MI estimators and null controls.

Core math for the FAWP detector.  The inner MI calculation
(Pearson → Gaussian MI) is JIT-compiled with Numba when available,
falling back to pure NumPy silently if Numba is not installed.

Install Numba for a significant speedup on large watchlist scans::

    pip install "fawp-index[fast]"   # includes numba
    pip install numba                # or directly

Speedup is most visible when null permutations are enabled
(n_null > 0), because the JIT loop replaces a Python list
comprehension that runs 200 times per tau value.

Ralph Clayton (2026) · doi:10.5281/zenodo.18663547
"""

import numpy as np
from fawp_index.constants import BETA_NULL_QUANTILE

# ── Numba JIT setup ───────────────────────────────────────────────────────────
# We try to import numba at module load time.  If it's not installed,
# _HAS_NUMBA stays False and every JIT function falls back to a plain
# NumPy implementation with identical results and public API.

_HAS_NUMBA = False
try:
    import numba as _numba
    _HAS_NUMBA = True
except ImportError:
    pass


def has_numba() -> bool:
    """Return True if Numba is installed and JIT is active."""
    return _HAS_NUMBA


# ── JIT-compiled inner kernels ────────────────────────────────────────────────

if _HAS_NUMBA:
    @_numba.njit(cache=True, fastmath=True)
    def _pearsonr_jit(x: np.ndarray, y: np.ndarray) -> float:
        """Pearson correlation — JIT compiled, no NaN handling (caller filters)."""
        n   = x.shape[0]
        if n < 2:
            return 0.0
        mx  = 0.0
        my  = 0.0
        for i in range(n):
            mx += x[i]
            my += y[i]
        mx /= n
        my /= n
        num = 0.0
        dx2 = 0.0
        dy2 = 0.0
        for i in range(n):
            dx  = x[i] - mx
            dy  = y[i] - my
            num += dx * dy
            dx2 += dx * dx
            dy2 += dy * dy
        denom = (dx2 * dy2) ** 0.5
        if denom < 1e-14:
            return 0.0
        r = num / denom
        if r >  0.99999: r =  0.99999
        if r < -0.99999: r = -0.99999
        return r

    @_numba.njit(cache=True, fastmath=True)
    def _mi_from_rho_jit(rho: float) -> float:
        """Gaussian MI from Pearson ρ — JIT compiled."""
        if rho > 0.99999:  rho =  0.99999
        if rho < -0.99999: rho = -0.99999
        return (-0.5 * np.log(1.0 - rho * rho)) / 0.6931471805599453  # log(2)

    @_numba.njit(cache=True, fastmath=True)
    def _mi_jit(x: np.ndarray, y: np.ndarray) -> float:
        """MI from two pre-filtered float64 arrays — JIT compiled."""
        rho = _pearsonr_jit(x, y)
        return _mi_from_rho_jit(rho)

    @_numba.njit(cache=True, fastmath=True, parallel=False)
    def _null_shuffle_loop_jit(
        x: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,  # shape (n_null, n) — pre-shuffled indices
    ) -> np.ndarray:
        """
        Vectorised null shuffle MI loop — JIT compiled.

        Parameters
        ----------
        x, y   : 1-D float64 arrays (already filtered for finiteness)
        indices: 2-D int array, shape (n_null, n) — caller generates shuffles
        """
        n_null = indices.shape[0]
        out    = np.empty(n_null)
        for i in range(n_null):
            y_shuf = y[indices[i]]
            out[i] = _mi_jit(x, y_shuf)
        return out

    @_numba.njit(cache=True, fastmath=True)
    def _null_shift_loop_jit(
        x:      np.ndarray,
        y:      np.ndarray,
        shifts: np.ndarray,   # 1-D int array of shift amounts
    ) -> np.ndarray:
        """
        Vectorised null shift MI loop — JIT compiled.
        """
        n_null = shifts.shape[0]
        n      = y.shape[0]
        out    = np.empty(n_null)
        for i in range(n_null):
            k      = int(shifts[i]) % n
            y_roll = np.empty(n)
            for j in range(n):
                y_roll[j] = y[(j - k) % n]
            out[i] = _mi_jit(x, y_roll)
        return out

else:
    # ── Pure NumPy fallbacks (identical results, no Numba required) ───────────
    def _pearsonr_jit(x, y):             # noqa: F811
        n = len(x)
        if n < 2:
            return 0.0
        r = float(np.corrcoef(x, y)[0, 1])
        return float(np.clip(r, -0.99999, 0.99999))

    def _mi_from_rho_jit(rho):           # noqa: F811
        rho = float(np.clip(rho, -0.99999, 0.99999))
        return float((-0.5 * np.log(1.0 - rho ** 2)) / np.log(2.0))

    def _mi_jit(x, y):                  # noqa: F811
        return _mi_from_rho_jit(_pearsonr_jit(x, y))

    def _null_shuffle_loop_jit(x, y, indices):   # noqa: F811
        return np.array([_mi_jit(x, y[idx]) for idx in indices])

    def _null_shift_loop_jit(x, y, shifts):      # noqa: F811
        n = len(y)
        return np.array([
            _mi_jit(x, np.roll(y, int(s))) for s in shifts
        ])


# ── Public API ────────────────────────────────────────────────────────────────
# All functions below have identical signatures and results regardless
# of whether Numba is installed.

def mi_from_rho(rho: float) -> float:
    """
    Gaussian channel MI from a Pearson correlation coefficient.

    I(X;Y) = -½ log₂(1 − ρ²)

    Parameters
    ----------
    rho : float   Pearson ρ in (−1, 1)

    Returns
    -------
    float   MI in bits (≥ 0)
    """
    if not np.isfinite(rho):
        return 0.0
    return _mi_from_rho_jit(float(rho))


def mi_from_arrays(x: np.ndarray, y: np.ndarray, min_n: int = 30) -> float:
    """
    Estimate MI between two arrays via the Gaussian identity.

    Filters NaN/Inf, checks minimum sample size, then computes
    Pearson ρ and returns I = −½ log₂(1 − ρ²).

    Parameters
    ----------
    x, y  : array-like
    min_n : int   Minimum valid samples required (default 30)

    Returns
    -------
    float   MI in bits
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < min_n:
        return 0.0
    return _mi_jit(x, y)


def null_shuffle(
    x, y,
    n_null: int   = 200,
    beta:   float = BETA_NULL_QUANTILE,
    rng=None,
    min_n:  int   = 30,
):
    """
    Stratified-shuffle null MI distribution.

    Permutes y independently for each of n_null trials
    and returns (mean, β-quantile) of the null MI values.
    Uses the JIT loop when Numba is available.

    Returns
    -------
    (mean_null, quantile_null) : (float, float)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < min_n:
        return 0.0, 0.0
    n = len(y)
    # Pre-generate all shuffled index arrays at once (one RNG call)
    indices = np.array([rng.permutation(n) for _ in range(n_null)], dtype=np.intp)
    null_mis = _null_shuffle_loop_jit(x, y, indices)
    return float(np.mean(null_mis)), float(np.quantile(null_mis, beta))


def null_shift(
    x, y,
    n_null: int   = 200,
    beta:   float = BETA_NULL_QUANTILE,
    rng=None,
    min_n:  int   = 30,
):
    """
    Autocorrelation-preserving shift null MI distribution.

    Circularly shifts y by a random amount for each of n_null trials
    and returns (mean, β-quantile) of the null MI values.
    Uses the JIT loop when Numba is available.

    Returns
    -------
    (mean_null, quantile_null) : (float, float)
    """
    if rng is None:
        rng = np.random.default_rng(43)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < min_n:
        return 0.0, 0.0
    n = len(y)
    shifts = rng.integers(1, max(2, n), size=n_null).astype(np.intp)
    null_mis = _null_shift_loop_jit(x, y, shifts)
    return float(np.mean(null_mis)), float(np.quantile(null_mis, beta))


def conservative_null_floor(
    x, y,
    n_null: int   = 200,
    beta:   float = BETA_NULL_QUANTILE,
    rng=None,
    min_n:  int   = 30,
) -> float:
    """
    Conservative null floor: max(β-quantile shuffle, β-quantile shift).

    Used by null_corrected_mi to subtract the noise floor from raw MI.

    Returns
    -------
    float   null floor in bits
    """
    _, q_shuf  = null_shuffle(x, y, n_null=n_null, beta=beta, rng=rng, min_n=min_n)
    _, q_shift = null_shift(x, y,   n_null=n_null, beta=beta, rng=rng, min_n=min_n)
    return max(q_shuf, q_shift)


def null_corrected_mi(
    x, y,
    n_null: int   = 200,
    beta:   float = BETA_NULL_QUANTILE,
    rng=None,
    min_n:  int   = 30,
) -> tuple:
    """
    Null-corrected MI: max(0, raw_MI − null_floor).

    Returns
    -------
    (corrected_mi, null_floor) : (float, float)
    """
    raw   = mi_from_arrays(x, y, min_n=min_n)
    floor = conservative_null_floor(
        x, y, n_null=n_null, beta=beta, rng=rng, min_n=min_n
    )
    return max(0.0, raw - floor), floor
