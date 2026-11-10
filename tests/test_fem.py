"""N6 单测：装配正确性、解析解、第三方 FEM、Cauer、收敛阶、物理一致性。"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fem import (default_case, build_mesh, assemble, solve_steady,
                 analytic_1d_steady, analytic_1d_field, skfem_steady,
                 error_metrics, convergence_order, fine_network,
                 lumped_network, run_transient, curve_error_full,
                 power_pulse, load_config, case_from_config,
                 default_config_path, spreading_resistance)
from fem.assembly import _tri_grad


@pytest.fixture(scope="module")
def small_case():
    return default_case().with_params(nx=20, ny_per_layer=4, t_end=1.0, dt=0.01)


@pytest.fixture(scope="module")
def small_sys(small_case):
    m = build_mesh(small_case)
    return assemble(small_case, m)


# ---------- 单元与装配 ----------
def test_triangle_gradients():
    p = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.25]])
    Ae, g = _tri_grad(p)
    assert abs(Ae - 0.5 * 0.5 * 0.25) < 1e-14
    assert np.allclose(g.sum(axis=0), 0.0, atol=1e-12)  # 形函数梯度之和为零
    assert np.allclose(g[0], [-2.0, 0.0])
    assert np.allclose(g[2], [0.0, 4.0])


def test_mass_matrix_conserves_heat_capacity(small_case, small_sys):
    total_rc = sum(L.rc * L.thickness * small_case.width for L in small_case.layers)
    assert abs(small_sys.M.sum() - total_rc) / total_rc < 1e-12


def test_conduction_stiffness_row_sum_zero(small_case, small_sys):
    # 去掉 Robin 后的刚度矩阵行和应为 0（纯 Neumann 常系数）
    edges = small_sys.mesh.bottom_edges
    n = small_sys.nnode
    H = np.zeros((n, n))
    for a, b in edges:
        L = small_case.width / small_sys.mesh.nx
        h = small_case.h_conv * L / 6
        H[a, a] += 2 * h
        H[b, b] += 2 * h
        H[a, b] += h
        H[b, a] += h
    from scipy.sparse import csr_matrix
    Kc = small_sys.K - csr_matrix(H)
    assert np.max(np.abs(np.asarray(Kc.sum(axis=1)).ravel())) < 1e-9


def test_heat_source_unit_power(small_case, small_sys):
    assert abs(small_sys.f_heat_unit.sum() - 1.0) < 1e-12


def test_energy_balance_steady(small_case, small_sys):
    T = solve_steady(small_sys, small_case.power_per_length)
    # 对流散热 = h*W*(底边平均温度 - T_amb)
    edges = small_sys.mesh.bottom_edges
    Tb = T[edges].mean()
    q_out = small_case.h_conv * small_case.width * (Tb - small_case.t_amb)
    assert abs(q_out - small_case.power_per_length) / small_case.power_per_length < 2e-3


# ---------- 解析解与第三方库 ----------
def test_steady_matches_analytic_1d(small_case, small_sys):
    T = solve_steady(small_sys, small_case.power_per_length)
    Ta = analytic_1d_field(small_case, small_sys.mesh)
    assert error_metrics(T, Ta)["max_abs"] < 0.02
    assert abs(T[small_sys.mesh.junction_node] - small_case.t_junction_steady) < 0.02


def test_steady_matches_skfem(small_case, small_sys):
    T = solve_steady(small_sys, small_case.power_per_length)
    Tsf, _ = skfem_steady(small_case, small_sys.mesh)
    assert error_metrics(T, Tsf)["max_abs"] < 1e-8


def test_volume_vs_surface_source_gap(small_case):
    qflux = small_case.power_per_length / small_case.width
    gap = qflux * small_case.die.thickness / (2 * small_case.die.k)
    assert abs((small_case.t_junction_surface_source
                - small_case.t_junction_steady) - gap) < 1e-12
    assert gap > 0.3  # 该工况下口径差 0.346 K，不可忽略


def test_cauer_steady_matches_fem_and_independent_of_n(small_case, small_sys):
    # Cauer 对一维稳态是节点精确的（热阻串联）；FEM 线性元对芯片层
    # 抛物线温度场有正常 O(h^2) 离散误差（npl=4 时约 3.4e-3 K）。
    T = solve_steady(small_sys, small_case.power_per_length)
    tj = T[small_sys.mesh.junction_node]
    vals = []
    for n in (1, 4, 32, 128):
        net = fine_network(small_case, n)
        vals.append(net.steady(small_case.power_per_length)[0])
    assert max(vals) - min(vals) < 1e-9  # 稳态与热容离散无关
    assert abs(vals[0] - small_case.t_junction_steady) < 1e-9
    assert abs(vals[0] - tj) < 0.01  # FEM 离散误差


# ---------- 瞬态与 Cauer 交叉验证 ----------
def test_fem_vs_fine_cauer_transient(small_case, small_sys):
    c = small_case
    rf = run_transient(small_sys, c, dt=0.002, t_end=0.6)
    rc = fine_network(c, 64).run(dt=0.002, t_end=0.6)
    e = curve_error_full(rf, rc, c.pulse_period)
    assert e["peak_rel_err_pct"] < 1.0
    assert e["rmse_rel_amp_pct"] < 2.0


def test_lumped_cauer_quasi_static_close(small_case):
    """低频准静态下集总网络与精细网络接近（物理预期）。"""
    c = small_case
    rl = lumped_network(c).run(dt=0.002, t_end=0.6)
    rc = fine_network(c, 64).run(dt=0.002, t_end=0.6)
    e = curve_error_full(rl, rc, c.pulse_period)
    assert e["peak_rel_err_pct"] < 0.5


def test_power_pulse_duty():
    c = default_case().with_params(pulse_period=2.0, duty=0.25)
    assert power_pulse(0.0, c) == c.power_per_length
    assert power_pulse(0.49, c) == c.power_per_length
    assert power_pulse(0.51, c) == 0.0
    assert power_pulse(2.1, c) == c.power_per_length


# ---------- 收敛阶 ----------
def test_time_convergence_be(small_case, small_sys):
    c = small_case
    dt_ref, t_end = 0.0005, 0.2
    const = lambda t: c.power_per_length
    ref = run_transient(small_sys, c, dt=dt_ref, t_end=t_end, power_fn=const)
    dts, errs = [], []
    for dt in (0.02, 0.01, 0.005, 0.0025):
        r = run_transient(small_sys, c, dt=dt, t_end=t_end, power_fn=const)
        dts.append(dt)
        errs.append(abs(r.t_junction[-1] - ref.t_junction[-1]))
    p, _ = convergence_order(np.array(dts), np.array(errs))
    assert 0.7 < p < 1.5, f"BE 时间收敛阶异常 p={p:.2f}"


def test_space_convergence_steady(small_case):
    c = small_case
    errs, hs = [], []
    for npl in (2, 4, 8, 16):
        m = build_mesh(c, nx=c.nx, ny_per_layer=npl)
        s = assemble(c, m)
        T = solve_steady(s, c.power_per_length)
        Ta = analytic_1d_field(c, m)
        errs.append(error_metrics(T, Ta)["max_abs"])
        hs.append(1.0 / npl)
    p, _ = convergence_order(np.array(hs), np.array(errs))
    assert 1.5 < p < 2.6, f"空间收敛阶异常 p={p:.2f}"


# ---------- 二维扩散热阻 ----------
def test_2d_spreading_raises_junction(small_case):
    c1 = small_case
    c2 = small_case.with_params(heat_source_width=0.002)  # nx=20 -> 4 格，对齐
    m1, m2 = build_mesh(c1), build_mesh(c2)
    T1 = solve_steady(assemble(c1, m1), c1.power_per_length)
    T2 = solve_steady(assemble(c2, m2), c2.power_per_length)
    tj1, tj2 = T1[m1.junction_node], T2[m2.junction_node]
    assert tj2 > tj1 + 1.0
    r = spreading_resistance(c2, tj2, c1.t_junction_steady)
    assert r["r_spread"] > 0 and r["r_spread_ratio_pct"] > 1


# ---------- 配置 ----------
def test_config_roundtrip_defaults():
    cfg = load_config(default_config_path())
    c = case_from_config(cfg)
    d = default_case()
    assert c.width == d.width and c.h_conv == d.h_conv
    assert c.power_per_length == d.power_per_length
    assert len(c.layers) == 3 and c.layers[-1].name == "Si_die"
    assert abs(c.r_effective - d.r_effective) < 1e-15
