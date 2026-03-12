"""
fawp_index.capacity.surfaces — Agency Capacity Surfaces

Loads and plots the E6 capacity surface data.

Ralph Clayton (2026) — doi:10.5281/zenodo.18663547
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class CapacityResult:
    """Result from CapacitySurface."""
    surface_a_mean: object    # pd.DataFrame (tau × bit_depth)
    surface_a_std: object
    surface_b_mean: object    # pd.DataFrame (tau × dropout_rate)
    surface_b_std: object
    tau_labels: list          # latency column names
    bit_depths: list          # row labels for surface A
    dropout_rates: list       # row labels for surface B

    def summary(self) -> str:
        lines = [
            "=" * 55,
            "E6: Agency Capacity Surfaces",
            "=" * 55,
            f"Surface A (quantization): {len(self.bit_depths)} bit depths × {len(self.tau_labels)} tau points",
            f"  Bit depths: {self.bit_depths[:5]}{'...' if len(self.bit_depths) > 5 else ''}",
            f"Surface B (erasure):      {len(self.dropout_rates)} dropout rates × {len(self.tau_labels)} tau points",
            f"  Dropout rates: {[f'{r:.0%}' if isinstance(r, float) else str(r) for r in self.dropout_rates[:5]]}{'...' if len(self.dropout_rates) > 5 else ''}",
            "=" * 55,
            "Key findings:",
            "  Surface A: Agency collapses below 3 bits (quantization cliff)",
            "  Surface B: 50% packet loss ≈ halves the effective horizon",
            "=" * 55,
        ]
        return "\n".join(lines)

    def plot(self, surface: str = 'both', **kwargs):
        """
        Plot capacity surfaces as heatmaps.

        Parameters
        ----------
        surface : str
            'A', 'B', or 'both' (default).
        """
        return _plot_surfaces(self, surface=surface, **kwargs)


class CapacitySurface:
    """
    Load and visualise E6 Agency Capacity Surfaces.

    Surface A: MI vs (latency τ, bit depth k)
    Surface B: MI vs (latency τ, dropout probability p)

    Example
    -------
        from fawp_index.capacity import CapacitySurface

        # Load published E6 data
        result = CapacitySurface.from_e6_data()
        print(result.summary())
        result.plot()

        # Plot just surface A (quantization cliff)
        result.plot(surface='A', save_path='quantization_cliff.png')
    """

    @classmethod
    def from_e6_data(cls) -> CapacityResult:
        """
        Load the published E6 surface data.

        Returns
        -------
        CapacityResult
        """
        import pandas as pd
        from fawp_index.data import (
            E6_SURFACE_BITS_MEAN, E6_SURFACE_BITS_STD,
            E6_SURFACE_DROPOUT_MEAN, E6_SURFACE_DROPOUT_STD,
        )

        # Surface A: rows = bit depths, columns = tau values
        sa_mean = pd.read_csv(E6_SURFACE_BITS_MEAN, index_col=0)
        sa_std  = pd.read_csv(E6_SURFACE_BITS_STD, index_col=0)

        # Surface B: rows = dropout rates, columns = tau values
        sb_mean = pd.read_csv(E6_SURFACE_DROPOUT_MEAN, index_col=0)
        sb_std  = pd.read_csv(E6_SURFACE_DROPOUT_STD, index_col=0)

        # Parse bit depths from index (e.g. "bits_1" → 1)
        def parse_bits(idx):
            bits = []
            for s in idx:
                try:
                    bits.append(int(str(s).replace('bits_', '')))
                except ValueError:
                    bits.append(str(s))
            return bits

        # Parse dropout rates from index (e.g. "dropout_0.0" → 0.0)
        def parse_dropout(idx):
            rates = []
            for s in idx:
                try:
                    rates.append(float(str(s).replace('dropout_', '').replace('drop_', '')))
                except ValueError:
                    rates.append(str(s))
            return rates

        bit_depths = parse_bits(sa_mean.index)
        dropout_rates = parse_dropout(sb_mean.index) if len(sb_mean.index) > 0 else []

        # Parse tau values from column names (e.g. "tau_0" → 0)
        def parse_tau(cols):
            taus = []
            for c in cols:
                try:
                    taus.append(int(str(c).replace('tau_', '')))
                except ValueError:
                    taus.append(str(c))
            return taus

        tau_labels = parse_tau(sa_mean.columns)

        return CapacityResult(
            surface_a_mean=sa_mean,
            surface_a_std=sa_std,
            surface_b_mean=sb_mean,
            surface_b_std=sb_std,
            tau_labels=tau_labels,
            bit_depths=bit_depths,
            dropout_rates=dropout_rates,
        )


def _plot_surfaces(result: CapacityResult, surface: str = 'both',
                   save_path=None, show=True):
    """Plot E6 capacity surfaces as heatmaps."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        raise ImportError("pip install matplotlib")

    do_a = surface in ('A', 'both')
    do_b = surface in ('B', 'both')
    n_panels = int(do_a) + int(do_b)

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    panel = 0

    if do_a:
        ax = axes[panel]
        data = result.surface_a_mean.values.astype(float)
        tau_vals = np.array(result.tau_labels[:data.shape[1]])
        bit_vals = np.array(result.bit_depths[:data.shape[0]])

        im = ax.imshow(data, aspect='auto', origin='lower',
                       cmap='YlOrRd_r', norm=mcolors.LogNorm(
                           vmin=max(data.max() * 0.001, 1e-6),
                           vmax=data.max()
                       ))
        plt.colorbar(im, ax=ax, label='Mutual Information (bits)')

        # Label every other tick
        n_tau = len(tau_vals)
        tau_ticks = list(range(0, n_tau, max(1, n_tau // 8)))
        ax.set_xticks(tau_ticks)
        ax.set_xticklabels([f"{tau_vals[i]:,.0f}" for i in tau_ticks], rotation=45, fontsize=7)
        ax.set_yticks(range(len(bit_vals)))
        ax.set_yticklabels([f"{b}-bit" for b in bit_vals], fontsize=8)

        ax.set_xlabel("Latency τ", fontsize=10)
        ax.set_ylabel("Bit depth", fontsize=10)
        ax.set_title("E6 Surface A: Quantization Cliff\n"
                     "Agency collapses below 3-bit depth", fontsize=10)
        panel += 1

    if do_b:
        ax = axes[panel]
        data = result.surface_b_mean.values.astype(float)

        # Clip negatives and zeros for log norm
        data_plot = np.clip(data, 1e-9, None)

        im = ax.imshow(data_plot, aspect='auto', origin='lower',
                       cmap='YlOrRd_r',
                       norm=mcolors.LogNorm(vmin=data_plot.min(), vmax=data_plot.max()))
        plt.colorbar(im, ax=ax, label='Mutual Information (bits)')

        tau_vals = np.array(result.tau_labels[:data.shape[1]])
        dr_vals = result.dropout_rates[:data.shape[0]]

        n_tau = len(tau_vals)
        tau_ticks = list(range(0, n_tau, max(1, n_tau // 8)))
        ax.set_xticks(tau_ticks)
        ax.set_xticklabels([f"{tau_vals[i]:,.0f}" for i in tau_ticks], rotation=45, fontsize=7)
        ax.set_yticks(range(len(dr_vals)))
        ax.set_yticklabels([f"{r:.0%}" if isinstance(r, float) else str(r)
                            for r in dr_vals], fontsize=8)

        ax.set_xlabel("Latency τ", fontsize=10)
        ax.set_ylabel("Dropout probability", fontsize=10)
        ax.set_title("E6 Surface B: Erasure Slope\n"
                     "50% packet loss ≈ halves the horizon", fontsize=10)

    fig.suptitle("E6: Agency Capacity Surfaces\n"
                 "Clayton (2026) doi:10.5281/zenodo.18663547",
                 fontsize=9, y=1.01)
    fig.text(0.99, 0.01, 'fawp-index | Clayton (2026)',
             ha='right', va='bottom', fontsize=7, color='gray', style='italic')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig, axes
