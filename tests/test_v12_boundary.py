"""v1.2 深化 F 回归测试：辐射四次律、边界雅可比、接触热阻、h_rad 量级。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fem import assemble, build_mesh, default_case, solve_steady
from fem.boundary_nonlinear import (
    STEFAN_BOLTZMANN, case_with_contact, celsius_to_kelvin,
    contact_rise_analytic, effective_layer_k, linearization_error,
    linearized_radiation_matrix, radiation_boundary, radiative_conductance_local,
    radiative_conductance_secant, radiative_flux,
    solve_linearized_radiative_steady, solve_radiative_steady,
)


@pytest.fixture(scope="module")
def sys0():
    case = default_case().with_params(nx=10, ny_per_layer=2)
    return case, assemble(case, build_mesh(case))


# ------------------------------------------------------------------ 纯函数
def test_kelvin_offset_and_zero_flux():
    assert celsius_to_kelvin(0.0) == pytest.approx(273.15)
    assert radiative_flux(25.0, 25.0, 0.9) == pytest.approx(0.0)
    assert radiative_flux(15.0, 25.0, 0.9) < 0.0


def test_secant_conductance_is_exact():
    t_amb = 25.0
    for t in (40.0, 100.0, 250.0, 400.0):
        q = radiative_flux(t, t_amb, 0.9)
        h = radiative_conductance_secant(t, t_amb, 0.9)
        assert q == pytest.approx(h * (t - t_amb), rel=1e-12)


def test_local_conductance_is_derivative():
    t, eps = 180.0, 0.8
    d = 1e-3
    fd = (radiative_flux(t + d, 25.0, eps)
          - radiative_flux(t - d, 25.0, eps)) / (2 * d)
    assert radiative_conductance_local(t, eps) == pytest.approx(fd, rel=1e-7)


def test_ambient_linearization_underestimates_at_high_temperature():
    out = linearization_error(np.array([25.0, 100.0, 300.0]), 25.0, 0.9)
    assert out["rel_err"][0] == pytest.approx(0.0, abs=1e-12)
    assert np.all(out["rel_err"][1:] < 0)  # 线性化低估散热
    assert abs(out["rel_err"][-1]) > 0.2   # 高温可达 ~30%


def test_h_rad_is_natural_convection_magnitude():
    """关键物理标度：h_rad ~ O(5-10) W/(m2K) ≪ 强制液冷 5000。"""
    for t in (50.0, 150.0, 300.0):
        h = radiative_conductance_local(t, 0.9)
        assert 1.0 < h < 40.0
    assert radiative_conductance_local(25.0, 0.9) < 0.02 * 5000.0


# ------------------------------------------------------ 边界装配与雅可比
def test_uniform_temperature_flux_integrates_exactly(sys0):
    case, s = sys0
    T = np.full(s.nnode, 125.0)
    R, J, q = radiation_boundary(s.mesh, T, 0.9, case.t_amb)
    q_exact = float(radiative_flux(125.0, case.t_amb, 0.9)) * case.width
    assert q == pytest.approx(q_exact, rel=1e-12)
    assert np.sum(R) == pytest.approx(q, rel=1e-12)


def test_boundary_jacobian_matches_finite_difference(sys0):
    case, s = sys0
    rng = np.random.default_rng(13)
    T = case.t_amb + rng.uniform(0, 120, s.nnode)
    _, J, _ = radiation_boundary(s.mesh, T, 0.8, case.t_amb)
    for j in (0, s.mesh.bottom_edges[3, 0], s.mesh.bottom_edges[-1, 1]):
        e = np.zeros(s.nnode)
        e[j] = 1.0
        d = 1e-4
        rp = radiation_boundary(s.mesh, T + d * e, 0.8, case.t_amb)[0]
        rm = radiation_boundary(s.mesh, T - d * e, 0.8, case.t_amb)[0]
        fd = (rp - rm) / (2 * d)
        assert np.allclose(np.asarray(J[:, j].todense()).ravel(), fd,
                           rtol=1e-5, atol=1e-6)


def test_linearized_matrix_row_sum(sys0):
    case, s = sys0
    H, f = linearized_radiation_matrix(s.mesh, 7.0, case.t_amb)
    row = np.asarray(H.sum(axis=1)).ravel()
    # 每条边 h*L/2 分配到两个端点；总行和 = h*width
    assert np.sum(row) == pytest.approx(7.0 * case.width, rel=1e-12)
    assert np.sum(f) == pytest.approx(7.0 * case.t_amb * case.width, rel=1e-12)


# ------------------------------------------------------------ 稳态 Newton
def test_zero_emissivity_reproduces_linear_steady(sys0):
    case, s = sys0
    ref = solve_steady(s, case.power_per_length)
    out = solve_radiative_steady(s, case, 0.0)
    assert out.converged
    assert np.max(np.abs(out.field - ref)) < 1e-8
    assert out.q_rad_W_per_m == pytest.approx(0.0, abs=1e-8)


def test_radiation_cools_and_energy_balances(sys0):
    case, s = sys0
    mild = case.with_params(h_conv=50.0, power_per_length=150.0)
    s2 = assemble(mild, build_mesh(mild))
    no_rad = float(solve_steady(s2, mild.power_per_length)[s2.mesh.junction_node])
    out = solve_radiative_steady(s2, mild, 0.9)
    assert out.converged
    assert out.t_junction < no_rad - 5.0          # 辐射显著降温
    assert 0.05 < out.radiation_share < 0.9
    # 能量守恒：对流 + 辐射 = 输入功率
    assert out.q_conv_W_per_m + out.q_rad_W_per_m == pytest.approx(
        mild.power_per_length, rel=2e-6)


def test_ambient_linearization_overpredicts_junction(sys0):
    case, s = sys0
    mild = case.with_params(h_conv=50.0, power_per_length=150.0)
    s2 = assemble(mild, build_mesh(mild))
    exact = solve_radiative_steady(s2, mild, 0.9).t_junction
    lin = solve_linearized_radiative_steady(s2, mild, 0.9)
    assert lin > exact  # 低估 h_rad ⇒ 高估结温


def test_strong_forced_convection_makes_radiation_negligible(sys0):
    """v1.0 默认工况 h=5000：辐射占比 <1%，属伪需求（负结果留档）。"""
    case, s = sys0
    out = solve_radiative_steady(s, case, 0.9)
    ref = float(solve_steady(s, case.power_per_length)[s.mesh.junction_node])
    assert out.radiation_share < 0.01
    assert abs(ref - out.t_junction) < 0.5


# ---------------------------------------------------------------- 接触热阻
def test_effective_layer_k_limits():
    assert effective_layer_k(5.0, 2e-4, 0.0) == pytest.approx(5.0)
    assert effective_layer_k(5.0, 2e-4, 1e-4) == pytest.approx(10.0 / 7.0, rel=1e-9)
    with pytest.raises(ValueError):
        effective_layer_k(5.0, 2e-4, -1.0)


def test_contact_rise_analytic_matches_fem(sys0):
    """等效 k 折算后的 FEM **温升增量** == 1D 解析增量（消去粗网格离散偏差）。"""
    case, s = sys0
    base = float(solve_steady(s, case.power_per_length)[s.mesh.junction_node])
    for rc in (0.0, 1e-6, 1e-5, 1e-4):
        c2 = case_with_contact(case, rc)
        s2 = assemble(c2, build_mesh(c2))
        fem_tj = float(solve_steady(s2, c2.power_per_length)[s2.mesh.junction_node])
        ana = contact_rise_analytic(case, rc)
        assert fem_tj - base == pytest.approx(ana["rise_vs_no_contact_K"], abs=2e-3)
        # 绝对值也应在粗网格离散误差（~0.012 K）内
        assert abs(fem_tj - ana["t_junction_C"]) < 0.02
    assert contact_rise_analytic(case, 0.0)["rise_vs_no_contact_K"] == pytest.approx(0.0)
