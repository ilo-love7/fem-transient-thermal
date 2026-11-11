"""v1.2 深化 E 回归测试：两阶段动态扩散网络、静态极限、tau 标定。"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fem import default_case
from fem.axisym import axisym_from_case
from fem.cauer import CauerNetwork
from fem.solver import periodic_steady_metrics
from fem.spreading import add_static_spreading
from fem.two_stage import (
    TwoStageNetwork, build_two_stage, diffusion_pole_closed_form,
    diffusion_time_constant, fit_tau_from_swing, two_stage_swing,
)


@pytest.fixture(scope="module")
def ladder():
    c = default_case().with_params(nx=8, ny_per_layer=2, pulse_period=0.2,
                                   duty=0.5, dt=0.005, t_end=1.6)
    return c


def base_network(case, n=16):
    return CauerNetwork(case, tuple([n] * case.n_layers))


# ------------------------------------------------------------------ 纯函数
def test_diffusion_time_constant_and_validation():
    assert diffusion_time_constant(3.5, 2e-3) == pytest.approx(7e-3)
    with pytest.raises(ValueError):
        diffusion_time_constant(-1.0, 1.0)
    with pytest.raises(ValueError):
        diffusion_time_constant(1.0, -1.0)


def test_pole_closed_form_static_limit():
    """tau -> 0 时扩散温差退化为 P*R_sp（静态修正摆幅）。"""
    p, r_sp, period, duty = 23.56, 3.6, 0.02, 0.5
    fast = diffusion_pole_closed_form(p, r_sp, 1e-9, period, duty)
    assert fast["swing"] == pytest.approx(p * r_sp, rel=1e-4)
    slow = diffusion_pole_closed_form(p, r_sp, 10.0, period, duty)
    assert slow["swing"] < 0.05 * fast["swing"]  # 慢极点被滤平


# ------------------------------------------------------------ 状态空间构造
def test_steady_state_delta_equals_power_times_r_sp(ladder):
    p = ladder.power_per_length
    r_sp, tau = 3.6, 5e-3
    net = build_two_stage(ladder, r_sp, tau, n=16)
    z = net.steady(p)
    assert z[0] == pytest.approx(p * r_sp, rel=1e-10)
    ref = base_network(ladder, 16).steady(p)
    assert np.allclose(z[1:], ref, atol=1e-9)
    assert net.junction_index == 0 and net.top_index == 1


def test_junction_is_top_plus_delta(ladder):
    net = build_two_stage(ladder, 3.6, 5e-3, n=16)
    p = ladder.power_per_length
    z = net.steady(p)
    assert np.isclose(z[0] + z[1], z[0] + z[net.top_index])


def test_small_tau_reduces_to_static_spreading(ladder):
    """tau << dt 时动态修正严格回到 v1.1 的静态串联 R_sp。"""
    r_sp = 3.6
    raw = base_network(ladder, 16).run(dt=ladder.dt, t_end=ladder.t_end)
    static = add_static_spreading(raw, r_sp)
    net = build_two_stage(ladder, r_sp, 1e-8, n=16)
    dyn = net.run(ladder, dt=ladder.dt, t_end=ladder.t_end)
    # t=0 是两者的**定义差异**：静态模型瞬间跳到 P*R_sp，动态极点需充电
    assert abs(dyn.t_junction[0] - static.t_junction[0]) == pytest.approx(
        ladder.power_per_length * r_sp, rel=1e-6)
    assert np.max(np.abs(dyn.t_junction[1:] - static.t_junction[1:])) < 1e-3


def test_swing_decreases_monotonically_with_tau(ladder):
    swings = [two_stage_swing(ladder, 3.6, tau, 0.05, 0.5, n=16,
                              n_steps_per_period=40, n_periods=8)["swing"]
              for tau in (1e-4, 1e-3, 5e-3, 2e-2, 1e-1)]
    assert all(a > b for a, b in zip(swings[:-1], swings[1:]))


def test_fit_tau_recovers_injected_value(ladder):
    tau_true = 4e-3
    target = two_stage_swing(ladder, 3.6, tau_true, 0.05, 0.5, n=16,
                             n_steps_per_period=40, n_periods=8)["swing"]
    fit = fit_tau_from_swing(ladder, 3.6, target, 0.05, 0.5, n=16)
    assert fit["bracket_ok"]
    assert fit["tau"] == pytest.approx(tau_true, rel=2e-2)


def test_axisym_case_is_converted_without_crash():
    ax = axisym_from_case(default_case(), 0.2, nr=20, nz_per_layer=3)
    net = build_two_stage(ax, 3.6, 5e-3, n=16)
    res = net.run(replace(ax, dt=0.005, t_end=0.02))
    assert np.all(np.isfinite(res.t_junction))
    assert "RCsp" in res.method


def test_periodic_mean_obeys_energy_invariant(ladder):
    """T_avg = T_amb + duty*P*(R_1D + R_sp) 与 tau 无关（精确能量不变量）。

    需跑 ≫ 系统时间常数（~0.87 s）到周期稳态；平均用梯形积分，
    避免方波跳变处算术平均的端点偏差。
    """
    c = replace(ladder, power_per_length=23.56, pulse_period=0.05,
                dt=0.05 / 120, t_end=6.0)
    p, r_sp = c.power_per_length, 3.6
    r_1d = (base_network(c, 16).steady(p)[0] - c.t_amb) / p
    expected = c.t_amb + c.duty * p * (r_1d + r_sp)
    for tau in (1e-3, 5e-3, 5e-2):
        net = build_two_stage(c, r_sp, tau, n=16)
        res = net.run(c, dt=c.dt, t_end=c.t_end)
        mask = res.t >= res.t[-1] - c.pulse_period
        avg = np.trapezoid(res.t_junction[mask], res.t[mask]) / c.pulse_period
        assert abs(avg - expected) < 1.0
