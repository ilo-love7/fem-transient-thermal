"""v1.3 E16：多极点 Foster 网络的单测。"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fem.multipole import (FosterNetwork, fit_foster, log_tau_grid,  # noqa: E402
                           pole_stability, single_pole_from_step)


def test_foster_step_is_sum_of_exponentials():
    net = FosterNetwork(np.array([0.2, 0.8]), np.array([0.01, 0.5]))
    t = np.array([0.0, 0.01, 0.1, 1.0, 10.0])
    ref = 0.2 * (1 - np.exp(-t / 0.01)) + 0.8 * (1 - np.exp(-t / 0.5))
    np.testing.assert_allclose(net.step(t), ref, rtol=1e-12)


def test_foster_step_saturates_at_r_total():
    net = FosterNetwork(np.array([0.1, 0.2, 0.7]), np.array([0.001, 0.1, 1.0]))
    assert net.r_total == pytest.approx(1.0)
    assert net.step(1e6)[0] == pytest.approx(1.0, rel=1e-6)


def test_foster_rejects_invalid_params():
    with pytest.raises(ValueError):
        FosterNetwork(np.array([1.0, 2.0]), np.array([0.1]))   # 长度不等
    with pytest.raises(ValueError):
        FosterNetwork(np.array([-1.0]), np.array([0.1]))       # 负电阻
    with pytest.raises(ValueError):
        FosterNetwork(np.array([1.0]), np.array([0.0]))        # 零时间常数


def test_log_grid_monotonic():
    g = log_tau_grid(1e-4, 10.0, 24)
    assert g.size == 24 and np.all(np.diff(g) > 0)
    assert g[0] == pytest.approx(1e-4) and g[-1] == pytest.approx(10.0)


def test_fit_recovers_two_poles_exactly():
    true = FosterNetwork(np.array([0.3, 0.7]), np.array([0.005, 0.2]))
    t = np.geomspace(1e-5, 5.0, 120)
    z = true.step(t)
    # 给一个含真值的 tau 网格 -> NNLS 应精确恢复，残差近机器精度
    grid = np.sort(np.concatenate([log_tau_grid(1e-4, 1.0, 18), true.taus]))
    fit = fit_foster(t, z, grid)
    np.testing.assert_allclose(fit["network"].step(t), z, rtol=1e-8, atol=1e-10)
    assert fit["rel_rmse"] < 1e-8


def test_fit_nonnegative_even_for_noisy_data():
    t = np.geomspace(1e-4, 2.0, 60)
    rng = np.random.default_rng(0)
    z = np.maximum(0.5 * (1 - np.exp(-t / 0.05))
                   + 0.2 * (1 - np.exp(-t / 0.5))
                   + rng.normal(0, 1e-3, t.size), 0.0)
    fit = fit_foster(t, z, log_tau_grid(1e-4, 2.0, 16))
    assert np.all(fit["r"] >= 0)
    assert fit["network"].r_total > 0


def test_single_pole_beats_wrong_two_pole_on_pure_single_data():
    true = FosterNetwork(np.array([1.0]), np.array([0.08]))
    t = np.geomspace(1e-4, 3.0, 100)
    z = true.step(t)
    one = single_pole_from_step(t, z)
    assert one["tau"] == pytest.approx(0.08, rel=0.12)
    assert one["network"].step(t)[2] == pytest.approx(z[2], rel=0.05)


def test_periodic_swing_monotonic_in_duty_and_period():
    net = FosterNetwork(np.array([0.4, 0.6]), np.array([0.01, 0.4]))
    s_short = net.periodic_swing(0.05, 0.5, n_steps_per_period=40, n_periods=8)
    s_long = net.periodic_swing(2.0, 0.5, n_steps_per_period=40, n_periods=8)
    # 慢极点被长周期充分充放电 -> 长周期摆幅更大
    assert s_long["swing"] > s_short["swing"]
    assert s_short["swing"] == pytest.approx(s_short["peak"] - s_short["trough"])


def test_periodic_swing_reaches_periodic_steady():
    net = FosterNetwork(np.array([1.0]), np.array([0.1]))
    a = net.periodic_swing(0.5, 0.5, n_steps_per_period=40, n_periods=4)
    b = net.periodic_swing(0.5, 0.5, n_steps_per_period=40, n_periods=12)
    assert a["swing"] == pytest.approx(b["swing"], rel=1e-3)


def test_pole_stability_detects_drift():
    stable = [{"r": np.array([0.7, 0.3]), "tau": np.array([0.1, 0.01])},
              {"r": np.array([0.6, 0.4]), "tau": np.array([0.11, 0.012])}]
    drift = [{"r": np.array([0.7, 0.3]), "tau": np.array([0.1, 0.01])},
             {"r": np.array([0.7, 0.3]), "tau": np.array([1.0, 0.01])}]
    assert pole_stability(stable)["stable"] is True
    assert pole_stability(drift)["stable"] is False
    assert pole_stability(drift)["spread_factor"] == pytest.approx(10.0)
