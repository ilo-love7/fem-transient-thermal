"""v1.1 回归测试：轴对称装配守恒、闭合模型、非线性、闭式解、反辨识可辨识性。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fem import (default_case, build_mesh, assemble, solve_steady,
                 periodic_steady_metrics, fine_network)
from fem.axisym import (axisym_from_case, assemble_axisym, solve_axisym_steady,
                        axisym_junction, spreading_resistance_abs,
                        run_axisym_transient, semi_infinite_spreading)
from fem.spreading import (fit_spreading_closure, equivalent_ladder_case,
                           add_static_spreading, corrected_ladder)
from fem.nonlinear import (solve_nonlinear_steady, rc_periodic_closed_form,
                           theta_artifacts)
from fem.identification import (make_network, build_case, exact_step_response,
                                forward, synthesize, fit, fisher,
                                OBSERVABLES, STEP_TIMES)
from fem.cauer import CauerNetwork


@pytest.fixture(scope="module")
def ax():
    c = axisym_from_case(default_case(), 1.0, nr=40, nz_per_layer=4)
    return c, assemble_axisym(c)


# ---------------------------------------------------------- A 轴对称装配
def test_axisym_full_width_matches_1d(ax):
    c, s = ax
    T = solve_axisym_steady(s)
    assert abs(axisym_junction(s, T) - c.t_junction_1d) < 2e-3
    assert abs(c.t_junction_1d - c.with_params().t_junction_1d) < 1e-12


def test_axisym_unit_heat_load_integrates_to_one_watt(ax):
    _, s = ax
    assert abs(s.f_heat_unit.sum() - 1.0) < 1e-10


def test_axisym_mass_matrix_conserves_total_capacity(ax):
    c, s = ax
    expected = sum(L.rc * np.pi * c.radius ** 2 * L.thickness
                   for L in c.layers)
    assert np.isclose(s.M.sum(), expected, rtol=1e-10)


def test_axisym_robin_boundary_integral_exact(ax):
    c, s = ax
    ones = np.ones(s.nnode)
    assert np.isclose((s.K @ ones).sum(),
                      c.h_conv * np.pi * c.radius ** 2, rtol=1e-9)
    assert np.isclose(s.f_conv.sum(),
                      c.h_conv * c.t_amb * np.pi * c.radius ** 2, rtol=1e-9)


def test_axisym_local_source_increases_junction():
    c0 = axisym_from_case(default_case(), 1.0, nr=40, nz_per_layer=4)
    vals = []
    for q in (1.0, 0.6, 0.2):
        c = c0.with_params(source_ratio=q)
        s = assemble_axisym(c)
        vals.append(axisym_junction(s, solve_axisym_steady(s)))
    assert vals[0] < vals[1] < vals[2]
    assert spreading_resistance_abs(vals[2], c) > 0


def test_axisym_grid_convergence_under_one_percent():
    c0 = axisym_from_case(default_case(), 0.2)
    out = []
    for nr, nz in ((40, 4), (80, 8)):
        c = c0.with_params(nr=nr, nz_per_layer=nz)
        s = assemble_axisym(c)
        out.append(spreading_resistance_abs(
            axisym_junction(s, solve_axisym_steady(s)), c))
    assert abs(out[0] - out[1]) / out[1] < 0.01


# ---------------------------------------------------------- A 闭合模型
def test_closure_zero_at_full_width_and_heldout_accuracy():
    c = axisym_from_case(default_case(), 1.0, nr=40, nz_per_layer=4)
    q = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    # 用给定 g0/g1 生成伪数据，检验拟合法与 q=1 归零性质（不重复跑 FEM）
    r = c.r_1d_abs * (q ** -(0.30 + 0.235 * (1 - q)) - 1.0)
    mask = np.zeros(len(q), dtype=bool)
    mask[[2, 4, 6, 8]] = True
    clo = fit_spreading_closure(q, r, c.r_1d_abs, mask)
    assert abs(clo.predict(np.array([1.0]))[0]) < 1e-12
    pred = clo.predict(q)
    assert np.max(np.abs((pred[1:] - r[1:]) / r[1:])) < 1e-9
    assert clo.g0 > 0 and clo.g1 > 0


def test_equivalent_ladder_reproduces_1d_steady():
    c = axisym_from_case(default_case(), 0.4)
    lc = equivalent_ladder_case(c)
    net = CauerNetwork(lc, tuple([64] * lc.n_layers))
    assert abs(net.steady(lc.power_per_length)[0] - c.t_junction_1d) < 1e-9
    assert abs((net.steady(lc.power_per_length)[0] - c.t_amb)
               / lc.power_per_length - c.r_1d_abs) < 1e-9


def test_static_spreading_shifts_peak_exactly():
    c = axisym_from_case(default_case(), 0.2, nr=40, nz_per_layer=4)
    raw, corr = corrected_ladder(c, 1.25, n=16, dt=0.02, t_end=1.0)
    assert np.allclose(corr.t_junction - raw.t_junction, raw.power * 1.25)
    assert abs(periodic_steady_metrics(corr, c)["t_peak"]
               - periodic_steady_metrics(raw, c)["t_peak"]
               - c.total_power * 1.25) < 1e-9


# ---------------------------------------------------------- B 非线性 / 闭式解
def test_nonlinear_alpha_zero_is_exact_linear():
    c = default_case()
    m = build_mesh(c)
    lin = solve_steady(assemble(c, m), c.power_per_length)[m.junction_node]
    r = solve_nonlinear_steady(c, 0.0, mesh=m)
    assert r.iterations <= 2          # 首步残差 73 K，第二步归零
    assert r.residual_history[-1] < 1e-8
    assert abs(r.t_junction - lin) < 1e-9


def test_nonlinear_negative_alpha_raises_junction():
    c = default_case()
    r = solve_nonlinear_steady(c, -0.003)
    assert r.converged
    assert r.t_junction > c.t_junction_steady + 2.0
    assert r.k_min > 0


def test_rc_closed_form_matches_single_node_integration():
    c = default_case()
    R, tau = c.r_effective, c.r_effective * c.heat_capacity_per_length
    for D in (0.2, 0.5, 0.8):
        cf = rc_periodic_closed_form(c.power_per_length, R, tau,
                                     c.t_amb, c.pulse_period, D)
        dt, n = 0.001, int(12 * c.pulse_period / 0.001)
        T = c.t_amb
        tail = []
        for k in range(n):
            t = k * dt
            P = c.power_per_length if (t % c.pulse_period) / c.pulse_period < D else 0.0
            T = (T + dt / tau * (c.t_amb + R * P)) / (1 + dt / tau)
            if t >= (n - int(c.pulse_period / dt)) * dt:
                tail.append(T)
        tail = np.asarray(tail)
        assert abs(cf["t_peak"] - tail.max()) < 0.02
        assert abs(cf["t_trough"] - tail.min()) < 0.02
        assert abs(cf["t_avg_energy"] - tail.mean()) < 0.02


def test_energy_invariant_average_temperature():
    c = default_case()
    R = c.r_effective
    for D in (0.1, 0.5, 0.9):
        cc = c.with_params(duty=D)
        m = periodic_steady_metrics(fine_network(cc, 64).run(dt=0.005, t_end=12.0), cc)
        assert abs(m["t_avg"] - (c.t_amb + D * c.power_per_length * R)) < 0.2


def test_theta_artifacts_no_ringing_in_off_phase():
    c = default_case().with_params(pulse_period=0.1, dt=0.01, t_end=0.6)
    s = assemble(c, build_mesh(c))
    from fem import run_transient
    r = run_transient(s, c, dt=0.01, t_end=0.6, theta=0.5)
    a = theta_artifacts(r, c.t_amb)
    assert a["n_off_phases"] >= 3
    assert a["max_rise_in_off_phase_K"] < 1e-3


# ---------------------------------------------------------- C 反辨识
def test_exact_step_response_matches_marching():
    c = default_case()
    net = make_network(build_case(c, 5.0, 5000.0), 24)
    fld = exact_step_response(net, STEP_TIMES, c.power_per_length)
    march = net.run(dt=0.0002, t_end=5.2)
    ref = np.interp(STEP_TIMES, march.t, march.t_junction)
    assert np.max(np.abs(fld[:, 0] - ref)) < 2e-3
    assert abs(fld[-1, 0] - ref[-1]) < 2e-3
    # 长时间极限应趋于稳态解（t=30 s >> tau≈1.06 s）
    steady = net.steady(c.power_per_length)[net.junction_index]
    long = exact_step_response(net, np.array([30.0]), c.power_per_length)
    assert abs(long[0, 0] - steady) < 1e-6


def test_fisher_steady_is_rank_deficient_transient_is_not():
    c = default_case()
    th = np.log([5.0, 5000.0])
    fs = fisher(th, c, "steady")
    ft = fisher(th, c, "step_top")
    assert fs["effective_rank"] == 1
    assert fs["condition_number"] > 1e12
    assert abs(fs["correlation"]) > 0.999
    assert ft["effective_rank"] == 2
    assert ft["condition_number"] < 100


def test_steady_fit_is_degenerate_but_transient_recovers_truth():
    c = default_case()
    obs = synthesize(c, 5.0, 5000.0, "steady")
    f1 = fit(obs, c, "steady", x0=(np.log(1.0), np.log(1000.0)))
    f2 = fit(obs, c, "steady", x0=(np.log(20.0), np.log(50000.0)))
    assert f1["cost"] < 1e-15 and f2["cost"] < 1e-15
    assert abs(f1["k_tim"] - f2["k_tim"]) > 1.0  # 不同解同样完美拟合
    for o in ("step_top", "step_top_bottom"):
        f = fit(synthesize(c, 5.0, 5000.0, o), c, o)
        assert abs(f["k_tim"] / 5.0 - 1) < 1e-4
        assert abs(f["h"] / 5000.0 - 1) < 1e-4
