"""v1.1 深化 B：温度相关导热系数、一阶 RC 周期方波闭式解、θ 法数值振荡诊断。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from .assembly import _tri_grad, _coo_accum, assemble
from .materials import ThermalCase
from .mesh import StructuredMesh


# ---------------------------------------------------------------- 非线性稳态
@dataclass
class NonlinearResult:
    field: np.ndarray
    t_junction: float
    iterations: int
    residual_history: list[float]
    k_min: float
    k_max: float
    converged: bool


def _assemble_k_cond(mesh: StructuredMesh, k_elem: np.ndarray):
    rows, cols, vals = [], [], []
    for e, tri in enumerate(mesh.elems):
        Ae, grad = _tri_grad(mesh.nodes[tri])
        _coo_accum(rows, cols, vals, tri, k_elem[e] * Ae * (grad @ grad.T))
    return coo_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(mesh.nnode, mesh.nnode)).tocsr()


def solve_nonlinear_steady(case: ThermalCase, alpha: float,
                           *, t_ref: float | None = None,
                           omega: float = 0.7, tol: float = 1e-8,
                           max_iter: int = 120,
                           mesh: StructuredMesh | None = None) -> NonlinearResult:
    """k(T) = k0 * [1 + alpha*(T - T_ref)]，逐单元取重心温度；Picard + 欠松弛。

    alpha=0 时一次迭代收敛到常物性 FEM 解（回归测试用）。
    返回收敛史、等效 k 的范围与结温。
    """
    from .mesh import build_mesh

    mesh = mesh or build_mesh(case)
    sys0 = assemble(case, mesh)
    # K_cond 单独拆出：sys0.K = K_cond(k0) + H_conv
    k0_elem = np.array([case.layers[j].k for j in mesh.elem_layer], dtype=float)
    K0 = _assemble_k_cond(mesh, k0_elem)
    H = (sys0.K - K0).tocsr()
    f = sys0.load(case.power_per_length)
    t_ref = case.t_amb if t_ref is None else t_ref
    if alpha == 0.0:
        omega = 1.0  # 线性问题一步收敛，便于回归测试

    T = np.full(mesh.nnode, case.t_amb)
    hist: list[float] = []
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        T_c = T[mesh.elems].mean(axis=1)
        k_e = k0_elem * (1.0 + alpha * (T_c - t_ref))
        if k_e.min() <= 0:
            raise ValueError("k(T) <= 0，alpha 过负或温升过大，物性外推失效")
        K = (_assemble_k_cond(mesh, k_e) + H).tocsc()
        T_new = splu(K).solve(f)
        res = float(np.max(np.abs(T_new - T)))
        hist.append(res)
        T = omega * T_new + (1.0 - omega) * T
        if res < tol:
            converged = True
            break
    return NonlinearResult(T, float(T[mesh.junction_node]), it, hist,
                           float(k_e.min()), float(k_e.max()), converged)


# ------------------------------------------------- 一阶 RC 周期方波闭式解
def rc_periodic_closed_form(power: float, r_eff: float, tau: float,
                            t_amb: float, period: float, duty: float) -> dict:
    """单节集总 RC 在周期方波下的**精确**稳态峰/谷/摆幅/均值。

    C dT/dt = -(T-T_amb)/R + P(t)，P(t)=P (0<=t<DT) 否则 0。
    令 a=D*T/tau, b=(1-D)*T/tau：
        T_high = T_amb + P*R*(1-e^-a)/(1-e^-(a+b))
        T_low  = T_amb + (T_high-T_amb)*e^-b
    能量守恒给出 T_avg = T_amb + D*P*R（与 T 无关）。
    """
    a = duty * period / tau
    b = (1.0 - duty) * period / tau
    denom = 1.0 - np.exp(-(a + b))
    rise = power * r_eff * (1.0 - np.exp(-a)) / denom
    t_high = t_amb + rise
    t_low = t_amb + rise * np.exp(-b)
    return {
        "t_peak": float(t_high), "t_trough": float(t_low),
        "swing": float(t_high - t_low),
        "t_avg_energy": float(t_amb + duty * power * r_eff),
        "peak_to_mean_ratio": float(rise / (duty * power * r_eff)),
    }


# ------------------------------------------------------------ θ 法振荡诊断
def theta_artifacts(res, t_amb: float = 25.0) -> dict:
    """诊断陡脉冲下 θ 法的非物理振铃。

    只在**完整的关断相内部**统计温度不降反升的步数（不能跨到下一个导通相，
    否则会把合法升温误判为振铃）。同时给出低于环境温度的下冲量。
    """
    y = res.t_junction
    p = res.power
    off = p < 0.5 * max(float(p.max()), 1e-12)
    idx = np.nonzero(off)[0]
    segments = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1) if idx.size else []
    rises = []
    for seg in segments:
        if seg.size >= 3:
            rises.append(float(np.max(np.diff(y[seg]))))
    return {
        "min_tj_C": float(y.min()),
        "undershoot_below_ambient_K": float(y.min() - t_amb),
        "max_rise_in_off_phase_K": max(rises) if rises else 0.0,
        "ringing_phases": int(sum(1 for x in rises if x > 1e-4)),
        "n_off_phases": int(len(segments)),
    }
