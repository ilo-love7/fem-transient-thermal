"""v1.2 深化 D 回归测试：非线性瞬态切线、Newton/Picard、离散能量守恒。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fem import (assemble, build_mesh, default_case, run_transient,
                 solve_steady)
from fem.nonlinear import solve_nonlinear_steady
from fem.nonlinear_transient import (
    GeometryCache, build_geometry_cache, conduction_and_tangent,
    conduction_matrix, energy_from_residual, energy_ledger,
    k_elements, nonlinear_residual, solve_nonlinear_transient, split_system,
)


@pytest.fixture(scope="module")
def small():
    case = default_case().with_params(nx=8, ny_per_layer=2, dt=0.01,
                                      t_end=0.1, pulse_period=0.1)
    mesh = build_mesh(case)
    sys0, k0, K0, H = split_system(case, mesh)
    return case, mesh, sys0, k0, K0, H


# ------------------------------------------------------------ 纯函数装配
def test_split_system_reconstructs_stiffness(small):
    case, mesh, sys0, k0, K0, H = small
    diff = (sys0.K - (K0 + H)).toarray()
    assert np.max(np.abs(diff)) < 1e-9


def test_k_elements_linear_and_alpha_zero(small):
    case, mesh, _, k0, _, _ = small
    rng = np.random.default_rng(3)
    T = case.t_amb + rng.uniform(0, 40, mesh.nnode)
    assert np.allclose(k_elements(k0, 0.0, T, mesh.elems, case.t_amb), k0)
    ke = k_elements(k0, 0.004, T, mesh.elems, case.t_amb)
    tc = T[mesh.elems].mean(axis=1)
    assert np.allclose(ke, k0 * (1.0 + 0.004 * (tc - case.t_amb)))


def test_tangent_matches_finite_difference(small):
    """切线矩阵必须等于 d[K(T)T]/dT（不是冻结 K 的简化 Newton）。"""
    case, mesh, _, k0, _, _ = small
    alpha = 0.004
    rng = np.random.default_rng(7)
    T = case.t_amb + rng.uniform(0, 60, mesh.nnode)
    ke = k_elements(k0, alpha, T, mesh.elems, case.t_amb)
    _, Jcond = conduction_and_tangent(mesh, k0, alpha, T, ke)

    def kt(Tx):
        kx = k_elements(k0, alpha, Tx, mesh.elems, case.t_amb)
        Kx, _ = conduction_and_tangent(mesh, k0, alpha, Tx, kx)
        return np.asarray(Kx @ Tx)

    for j in (0, mesh.nnode // 3, mesh.junction_node):
        e = np.zeros(mesh.nnode)
        e[j] = 1.0
        eps = 1e-4
        fd = (kt(T + eps * e) - kt(T - eps * e)) / (2 * eps)
        exact = np.asarray(Jcond[:, j].todense()).ravel()
        assert np.allclose(exact, fd, rtol=2e-5, atol=2e-4)


def test_conduction_equals_v11_assembler(small):
    from fem.nonlinear import _assemble_k_cond

    case, mesh, _, k0, _, _ = small
    rng = np.random.default_rng(11)
    T = case.t_amb + rng.uniform(0, 30, mesh.nnode)
    ke = k_elements(k0, 0.002, T, mesh.elems, case.t_amb)
    Kc, _ = conduction_and_tangent(mesh, k0, 0.002, T, ke)
    ref = _assemble_k_cond(mesh, ke).toarray()
    assert np.allclose(Kc.toarray(), ref, atol=1e-12)


def test_geometry_cache_reuse_is_equivalent(small):
    """向量化缓存装配 vs 每次重建几何：结果必须逐位一致（E13 提速的正确性）。"""
    case, mesh, _, k0, _, _ = small
    rng = np.random.default_rng(19)
    T = case.t_amb + rng.uniform(0, 80, mesh.nnode)
    ke = k_elements(k0, 0.004, T, mesh.elems, case.t_amb)
    geom = build_geometry_cache(mesh)
    assert isinstance(geom, GeometryCache)
    Kc1, J1 = conduction_and_tangent(mesh, k0, 0.004, T, ke, geom)
    Kc2, J2 = conduction_and_tangent(mesh, k0, 0.004, T, ke)
    assert np.allclose(Kc1.toarray(), Kc2.toarray(), atol=1e-12)
    assert np.allclose(J1.toarray(), J2.toarray(), atol=1e-12)
    # 复用同一缓存、换温度后仍与 v1.1 装配器一致
    T2 = T + rng.uniform(-5, 5, mesh.nnode)
    ke2 = k_elements(k0, 0.004, T2, geom.elems, case.t_amb)
    from fem.nonlinear import _assemble_k_cond

    assert np.allclose(conduction_matrix(geom, ke2).toarray(),
                       _assemble_k_cond(mesh, ke2).toarray(), atol=1e-12)


# ------------------------------------------------------------ 能量恒等式
def test_energy_ledger_equals_one_transpose_residual(small):
    """1^T R == 显式能量账本（不依赖迭代是否收敛）。"""
    case, mesh, sys0, k0, K0, H = small
    rng = np.random.default_rng(5)
    T_prev = case.t_amb + rng.uniform(0, 50, mesh.nnode)
    T = T_prev + rng.uniform(-3, 3, mesh.nnode)
    p_prev, p_now, dt, theta = 3000.0, 0.0, 0.01, 1.0
    f_now, f_prev = sys0.load(p_now), sys0.load(p_prev)
    R = nonlinear_residual(sys0.M, H, K0, K0, T, T_prev,
                           f_now, f_prev, dt, theta)
    led = energy_ledger(sys0.M, H, sys0.f_conv, T_prev, T,
                        p_prev, p_now, dt, theta)
    assert abs(energy_from_residual(R) - led["residual_W_per_m"]) < 1e-6
    # 人为未收敛 ⇒ 能量失衡非零（可作为诊断）
    assert abs(led["residual_W_per_m"]) > 1.0


def test_energy_ledger_vanishes_at_converged_linear_step(small):
    case, mesh, sys0, k0, K0, H = small
    from fem.solver import power_pulse

    dt, theta = case.dt, case.theta
    T_prev = np.full(mesh.nnode, case.t_amb)
    p_prev, p_now = power_pulse(0.0, case), power_pulse(dt, case)
    A = (sys0.M / dt + theta * sys0.K).tocsc()
    B = (sys0.M / dt - (1 - theta) * sys0.K).tocsr()
    from scipy.sparse.linalg import splu

    T = splu(A).solve(B @ T_prev + theta * sys0.load(p_now)
                      + (1 - theta) * sys0.load(p_prev))
    led = energy_ledger(sys0.M, H, sys0.f_conv, T_prev, T,
                        p_prev, p_now, dt, theta)
    assert led["relative_residual"] < 1e-9


# ------------------------------------------------------------ 求解器交叉验证
def test_alpha_zero_reproduces_linear_theta_solver(small):
    case, mesh, sys0, _, _, _ = small
    lin = run_transient(sys0, case)
    non = solve_nonlinear_transient(case, 0.0, method="newton", mesh=mesh)
    assert np.max(np.abs(lin.t_junction - non.result.t_junction)) < 1e-8
    assert non.max_iterations_per_step <= 2  # 线性问题 Newton 至多两步
    assert non.max_relative_energy_residual < 1e-8


def test_newton_and_picard_agree(small):
    case, mesh, _, _, _, _ = small
    new = solve_nonlinear_transient(case, 0.003, method="newton", mesh=mesh)
    pic = solve_nonlinear_transient(case, 0.003, method="picard", mesh=mesh,
                                    omega=0.8)
    assert new.converged and pic.converged
    assert np.max(np.abs(new.result.t_junction - pic.result.t_junction)) < 2e-6
    assert new.total_iterations <= pic.total_iterations


def test_constant_power_matches_v11_steady_picard(small):
    """常功率下瞬态终值 → v1.1 独立实现的稳态 Picard 解。"""
    from dataclasses import replace

    case, mesh, sys0, _, _, _ = small
    c = replace(case, t_end=12.0, dt=0.02)  # ≫ 11*tau，稳态误差 <0.01 K
    const = lambda t: case.power_per_length  # noqa: E731
    out = solve_nonlinear_transient(c, -0.003, method="newton", mesh=mesh,
                                    power_fn=const)
    steady = solve_nonlinear_steady(c, -0.003, mesh=mesh)
    assert steady.converged
    assert abs(out.result.t_junction[-1] - steady.t_junction) < 0.02


def test_energy_residual_flags_unconverged_iteration(small):
    """人为截断非线性迭代 ⇒ 能量失衡显著放大（诊断有效）。

    注意：本工况 k(T) 非线性很弱，Newton 一步即收敛，因此单纯放松 tol
    并不能制造未收敛状态（这本身是 v1.2 的负结果之一）；这里用
    max_iter=1 的强欠松弛 Picard 构造未收敛解。
    """
    case, mesh, _, _, _, _ = small
    tight = solve_nonlinear_transient(case, 0.004, mesh=mesh, tol=1e-10)
    truncated = solve_nonlinear_transient(
        case, 0.004, mesh=mesh, method="picard", omega=0.4, max_iter=1)
    assert tight.max_energy_residual_W_per_m < 1e-5
    assert tight.max_energy_residual_W_per_m < (
        truncated.max_energy_residual_W_per_m / 100.0)
