"""tests/test_oats.py — AgencyHorizon, ControlCliff, CapacitySurface tests."""
import numpy as np
from fawp_index.oats import AgencyHorizon, DistributionalRobustness
from fawp_index.capacity import CapacitySurface
from fawp_index.simulate import ControlCliff


class TestAgencyHorizon:
    def test_tau_h_finite(self):
        ah = AgencyHorizon(P=1.0, sigma0_sq=0.01, alpha=0.001, epsilon=0.1)
        tau_h = ah.tau_h_analytic()
        assert np.isfinite(tau_h)
        assert tau_h > 0

    def test_tau_h_increases_with_P(self):
        ah = AgencyHorizon(sigma0_sq=0.01, alpha=0.001, epsilon=0.1)
        t1 = ah.tau_h_analytic(P=0.1)
        t2 = ah.tau_h_analytic(P=1.0)
        t3 = ah.tau_h_analytic(P=10.0)
        assert t1 < t2 < t3

    def test_tau_h_decreases_with_alpha(self):
        ah = AgencyHorizon(P=1.0, sigma0_sq=0.01, epsilon=0.1)
        t1 = ah.tau_h_analytic(alpha=0.0001)
        t2 = ah.tau_h_analytic(alpha=0.001)
        t3 = ah.tau_h_analytic(alpha=0.01)
        assert t1 > t2 > t3

    def test_mi_at_horizon_equals_epsilon(self):
        ah = AgencyHorizon(P=1.0, sigma0_sq=0.01, alpha=0.001, epsilon=0.1)
        result = ah.compute()
        assert abs(result.mi_at_tau_h - 0.1) < 1e-6

    def test_compute_returns_result(self):
        from fawp_index.oats.model import OATSResult
        ah = AgencyHorizon()
        result = ah.compute()
        assert isinstance(result, OATSResult)
        assert len(result.tau_grid) == 500
        assert len(result.mi) == 500

    def test_sweep_returns_dataframe(self):
        ah = AgencyHorizon()
        sweep = ah.sweep(
            P_values=[0.1, 1.0],
            alpha_values=[0.001],
            epsilon_values=[0.1],
        )
        df = sweep.dataframe()
        assert len(df) == 2
        assert 'tau_h_theory' in df.columns

    def test_compare_e1_loads(self):
        ah = AgencyHorizon()
        sweep = ah.compare_e1()
        df = sweep.dataframe()
        assert len(df) > 0
        assert 'tau_h_theory' in df.columns
        assert 'tau_h_mc' in df.columns


class TestControlCliff:
    def test_from_e5_data(self):
        from fawp_index.simulate import ControlCliffResult
        result = ControlCliff.from_e5_data()
        assert isinstance(result, ControlCliffResult)
        assert result.cliff_delay is not None
        assert result.cliff_mi is not None
        assert result.failure_rate.max() >= 0.5

    def test_cliff_delay_positive(self):
        result = ControlCliff.from_e5_data()
        assert result.cliff_delay > 0

    def test_mi_monotonically_decreasing(self):
        result = ControlCliff.from_e5_data()
        # MI should generally decrease with delay
        first_half = result.mi_bits[:len(result.mi_bits)//2].mean()
        second_half = result.mi_bits[len(result.mi_bits)//2:].mean()
        assert first_half > second_half


class TestCapacitySurface:
    def test_from_e6_data(self):
        from fawp_index.capacity.surfaces import CapacityResult
        result = CapacitySurface.from_e6_data()
        assert isinstance(result, CapacityResult)

    def test_surface_a_has_bit_depths(self):
        result = CapacitySurface.from_e6_data()
        assert len(result.bit_depths) > 0
        assert 1 in result.bit_depths   # 1-bit should be present

    def test_surface_b_has_dropout_rates(self):
        result = CapacitySurface.from_e6_data()
        assert len(result.dropout_rates) > 0

    def test_1bit_lower_mi_than_8bit(self):
        result = CapacitySurface.from_e6_data()
        df = result.surface_a_mean
        # 1-bit row should have lower MI than 8-bit row overall
        idx_1 = result.bit_depths.index(1) if 1 in result.bit_depths else None
        idx_8 = result.bit_depths.index(8) if 8 in result.bit_depths else None
        if idx_1 is not None and idx_8 is not None:
            mean_1 = df.iloc[idx_1].mean()
            mean_8 = df.iloc[idx_8].mean()
            assert mean_1 < mean_8


class TestDistributionalRobustness:
    def test_from_e4_data(self):
        from fawp_index.oats.robustness import RobustnessResult
        result = DistributionalRobustness.from_e4_data()
        assert isinstance(result, RobustnessResult)
        assert len(result.scenarios) == 3

    def test_all_scenarios_present(self):
        result = DistributionalRobustness.from_e4_data()
        names = list(result.scenarios.keys())
        assert any('Baseline' in n for n in names)
        assert any('Uniform' in n or 'Bounded' in n for n in names)
        assert any('Student' in n or 'Heavy' in n for n in names)
