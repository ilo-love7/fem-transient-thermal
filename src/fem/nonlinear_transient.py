"""v1.2 深化 D：k(T) 与瞬态 theta 法的 Newton/Picard 耦合 + 离散能量守恒审计。

v1.1 的 `nonlinear.solve_nonlinear_steady` 只做**稳态** Picard；本模块把温度相关
导热系数推进到瞬态，并给出完整切线刚度（不是冻结 K 的简化 Newton）。

离散格式（alpha=0 时严格退化为 v1.0 `solver.run_transient` 的 theta 法）：

    R(T) = M/dt (T - T_prev)
           + theta * [K_cond(T) + H] T
           + (1-theta) * [K_cond(T_prev) + H] T_prev
           - theta * f_now - (1-theta) * f_prev = 0

其中 k_e(T) = k0_e * [1 + alpha*(mean(T_e) - T_ref)]，逐单元取重心温度。
Newton 切线（K_cond 对 T 的依赖不可忽略）：

    d[K_cond(T) T]/dT_j = K_cond[:, j] + (k0_e*alpha/3) * (G_e @ T_e)
                          （对每个含 j 的单元 e 累加，G_e = Ae*gradN.gradN^T）

于是 J = M/dt + theta * [H + K_cond(T) + D(T)]。

**能量守恒恒等式**：P1 拉普拉斯刚度对称且行和为零、f_heat_unit 行和为 1 W/m、
1^T(H T - f_conv) = h∫(T-T_amb) ds，因此

    1^T R(T) == ΔU/dt + Q_conv - Q_in    （恒等，不依赖迭代是否收敛）

这给出一个**免费的收敛诊断**：把残差向量求和即得能量失衡，无需额外装配。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from .assembly import _tri_grad, assemble
from .materials import ThermalCase
from .mesh import StructuredMesh, build_mesh
from .nonlinear import _assemble_k_cond
from .solver import TransientResult, power_pulse

STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m^2 K^4)


# --------------------------------------------------------------- 纯函数装配
def split_system(case: ThermalCase, mesh: StructuredMesh):
    """拆出 (FEMSystem, k0 逐单元, 传导刚度 K0, 对流刚度 H)。

    v1.1 `solve_nonlinear_steady` 的同一技巧：sys.K = K_cond(k0) + H_conv。
    """
    sys0 = assemble(case, mesh)
    k0_elem = np.array([case.layers[j].k for j in mesh.elem_layer], dtype=float)
    K0 = _assemble_k_cond(mesh, k0_elem)
    H = (sys0.K - K0).tocsr()
    return sys0, k0_elem, K0, H


def k_elements(k0_elem: np.ndarray, alpha: float, T: np.ndarray,
               elems: np.ndarray, t_ref: float) -> np.ndarray:
    """逐单元重心温度下的 k(T) = k0*[1+alpha*(T_c-T_ref)]。"""
    t_c = T[elems].mean(axis=1)
    return k0_elem * (1.0 + alpha * (t_c - t_ref))


@dataclass
class GeometryCache:
    """单元几何的预计算结果（网格不变时只需算一次）。

    G[e] = Ae*(gradN @ gradN.T)；rows/cols 是 9 个 COO 条目索引，
    对 K_e[i,j] 与 D[tri_i,tri_j] 通用（见 conduction_and_tangent）。
    """

    elems: np.ndarray
    G: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    nnode: int


def build_geometry_cache(mesh: StructuredMesh) -> GeometryCache:
    """逐单元跑一次 _tri_grad，之后所有装配都走向量化路径。

    原先每步两次逐单元 Python 循环（576 单元 × 600 步 × 32 组）是 E13 的
    主要瓶颈；缓存后单次装配由 576 次 numpy 小数组创建降为一次 COO 构造。
    """
    elems = mesh.elems.astype(np.int64)
    nelem = elems.shape[0]
    G = np.empty((nelem, 3, 3))
    for e, tri in enumerate(elems):
        Ae, grad = _tri_grad(mesh.nodes[tri])
        G[e] = Ae * (grad @ grad.T)
    rows = np.repeat(elems, 3, axis=1)   # [a,a,a,b,b,b,c,c,c] = tri_i
    cols = np.tile(elems, (1, 3))       # [a,b,c,a,b,c,a,b,c] = tri_j
    return GeometryCache(elems, G, rows, cols, mesh.nnode)


def conduction_matrix(geom: GeometryCache, k_e: np.ndarray):
    """向量化装配 K_cond(k_e)（与 v1.1 `_assemble_k_cond` 逐位一致）。"""
    vals = (geom.G * np.asarray(k_e)[:, None, None]).ravel()
    return coo_matrix((vals, (geom.rows.ravel(), geom.cols.ravel())),
                      shape=(geom.nnode, geom.nnode)).tocsr()


def conduction_and_tangent(mesh: StructuredMesh, k0_elem: np.ndarray,
                           alpha: float, T: np.ndarray,
                           k_e: np.ndarray,
                           geom: GeometryCache | None = None):
    """返回 (K_cond(T), d[K_cond(T) T]/dT)。

    后者 = K_cond(T) + D；D[tri_i, tri_j] += (k0_e*alpha/3)*(G_e @ T_e)_i。
    alpha=0 时 D=0。传入 geom 可复用预计算几何（求解器路径）。
    """
    geom = geom or build_geometry_cache(mesh)
    Kc = conduction_matrix(geom, k_e)
    if alpha == 0.0:
        return Kc, Kc
    g_t = np.einsum("eij,ej->ei", geom.G, T[geom.elems])
    v = (np.asarray(k0_elem) * alpha / 3.0)[:, None] * g_t
    vals_d = np.repeat(v, 3, axis=1).ravel()
    D = coo_matrix((vals_d, (geom.rows.ravel(), geom.cols.ravel())),
                   shape=(geom.nnode, geom.nnode)).tocsr()
    return Kc, (Kc + D).tocsr()


def nonlinear_residual(M, H, Kc, Kc_prev, T, T_prev, f_now, f_prev,
                       dt: float, theta: float) -> np.ndarray:
    """theta 法非线性残差（含对流刚度 H 与 Robin 常数载荷 f_conv）。"""
    return (np.asarray(M @ (T - T_prev)) / dt
            + theta * (np.asarray(Kc @ T) + np.asarray(H @ T))
            + (1.0 - theta) * (np.asarray(Kc_prev @ T_prev)
                               + np.asarray(H @ T_prev))
            - theta * f_now - (1.0 - theta) * f_prev)


def energy_ledger(M, H, f_conv, T_prev, T, p_prev, p_now,
                  dt: float, theta: float, p_ref: float = 1.0) -> dict:
    """显式离散能量账本（单位厚度 W/m、J/m）。

    1^T M T = ∫rho*c*T dx（一致质量矩阵行和性质）；
    1^T(H T - f_conv) = h∫(T-T_amb) ds 为对流散热功率。

    相对残差的分母必须用**特征功率 p_ref**（如额定线功率）而非本步 q_in：
    关断相 q_in=0，用 q_in 归一化会把机器噪声放大成天文数字。
    """
    u_prev = float(np.sum(M @ T_prev))
    u_now = float(np.sum(M @ T))
    q_prev = float(np.sum(H @ T_prev - f_conv))
    q_now = float(np.sum(H @ T - f_conv))
    q_in = float(theta * p_now + (1.0 - theta) * p_prev)
    q_out = float(theta * q_now + (1.0 - theta) * q_prev)
    resid = (u_now - u_prev) / dt + q_out - q_in
    return {
        "delta_internal_energy_J_per_m": u_now - u_prev,
        "q_conv_W_per_m": q_out,
        "q_in_W_per_m": q_in,
        "residual_W_per_m": resid,
        "relative_residual": abs(resid) / max(abs(p_ref), 1e-12),
    }


def energy_from_residual(residual: np.ndarray) -> float:
    """1^T R：与 energy_ledger 的 residual 恒等（见模块 docstring）。"""
    return float(np.sum(residual))


# ------------------------------------------------------------------ 求解器
@dataclass
class NonlinearTransientResult:
    result: TransientResult
    method: str
    total_iterations: int
    max_iterations_per_step: int
    mean_iterations_per_step: float
    max_energy_residual_W_per_m: float
    max_relative_energy_residual: float
    residual_history: list[float] = field(default_factory=list)
    converged: bool = True
    k_min: float = 0.0
    k_max: float = 0.0


def solve_nonlinear_transient(
    case: ThermalCase, alpha: float, *, method: str = "newton",
    mesh: StructuredMesh | None = None, dt: float | None = None,
    theta: float | None = None, t_end: float | None = None,
    tol: float = 1e-9, max_iter: int = 40, omega: float = 0.8,
    t_ref: float | None = None, power_fn=None,
) -> NonlinearTransientResult:
    """k(T) 瞬态求解；method ∈ {"newton", "picard"}；alpha=0 单步线性求解。

    power_fn(t) -> float（W/m）；缺省用周期方波 power_pulse。
    每步从上一步温度出发做非线性迭代，返回迭代统计与能量失衡指标。
    """
    if method not in ("newton", "picard"):
        raise ValueError("method 必须是 newton 或 picard")
    mesh = mesh or build_mesh(case)
    sys0, k0_elem, K0, H = split_system(case, mesh)
    geom = build_geometry_cache(mesh)  # 单元几何只算一次
    dt = case.dt if dt is None else dt
    theta = case.theta if theta is None else theta
    t_end = case.t_end if t_end is None else t_end
    t_ref = case.t_amb if t_ref is None else t_ref
    pfn = (lambda t: power_pulse(t, case)) if power_fn is None else power_fn
    if alpha == 0.0:
        omega = 1.0  # 线性问题一步收敛，便于回归对照

    M = sys0.M.tocsr()
    n_steps = int(round(t_end / dt))
    times = np.arange(n_steps + 1) * dt
    T = np.full(mesh.nnode, case.t_amb, dtype=float)
    p_prev = pfn(0.0)
    f_prev = sys0.load(p_prev)
    Kc_prev = (K0 if alpha == 0.0 else conduction_matrix(
        geom, k_elements(k0_elem, alpha, T, geom.elems, t_ref)))

    tj = np.empty(n_steps + 1)
    tmax = np.empty(n_steps + 1)
    tb = np.empty(n_steps + 1)
    pw = np.empty(n_steps + 1)
    tj[0], tmax[0], tb[0] = T[mesh.junction_node], T.max(), T[mesh.bottom_edges[mesh.nx // 2, 0]]
    pw[0] = p_prev

    it_total = 0
    it_max = 0
    max_energy = 0.0
    max_rel_energy = 0.0
    hist: list[float] = []
    converged = True
    k_lo, k_hi = float(k0_elem.min()), float(k0_elem.max())

    for n in range(1, n_steps + 1):
        p_now = pfn(times[n])
        f_now = sys0.load(p_now)
        T_prev = T.copy()
        it_step = 0
        for it_step in range(1, max_iter + 1):
            k_e = k_elements(k0_elem, alpha, T, mesh.elems, t_ref) if alpha else k0_elem
            if k_e.min() <= 0.0:
                raise ValueError("k(T) <= 0：alpha 过负或温升过大，物性外推失效")
            Kc, Ktan = conduction_and_tangent(mesh, k0_elem, alpha, T, k_e,
                                              geom)
            R = nonlinear_residual(M, H, Kc, Kc_prev, T, T_prev,
                                   f_now, f_prev, dt, theta)
            if method == "newton":
                J = (M / dt + theta * (H + Ktan)).tocsc()
                T_new = T - splu(J).solve(R)
            else:
                A = (M / dt + theta * (Kc + H)).tocsc()
                rhs = (np.asarray(M @ T_prev) / dt
                       - (1.0 - theta) * (np.asarray(Kc_prev @ T_prev)
                                          + np.asarray(H @ T_prev))
                       + theta * f_now + (1.0 - theta) * f_prev)
                T_new = omega * splu(A).solve(rhs) + (1.0 - omega) * T
            # 收敛判据落在**更新后**的状态上，避免把未迭代的初值当成解
            k_new = k_elements(k0_elem, alpha, T_new, mesh.elems, t_ref) if alpha else k0_elem
            Kc_new, _ = conduction_and_tangent(mesh, k0_elem, alpha, T_new,
                                               k_new, geom)
            R_new = nonlinear_residual(M, H, Kc_new, Kc_prev, T_new, T_prev,
                                       f_now, f_prev, dt, theta)
            res_inf = float(np.max(np.abs(R_new)))
            hist.append(res_inf)
            T = T_new
            if res_inf < tol:
                break
        else:
            converged = False

        it_total += it_step
        it_max = max(it_max, it_step)
        lead = energy_ledger(M, H, sys0.f_conv, T_prev, T,
                             p_prev, p_now, dt, theta,
                             p_ref=case.power_per_length)
        max_energy = max(max_energy, abs(lead["residual_W_per_m"]))
        max_rel_energy = max(max_rel_energy, lead["relative_residual"])

        Kc_prev = (K0 if alpha == 0.0 else conduction_matrix(
            geom, k_elements(k0_elem, alpha, T, geom.elems, t_ref)))
        f_prev, p_prev = f_now, p_now
        tj[n], tmax[n], tb[n] = (T[mesh.junction_node], T.max(),
                                 T[mesh.bottom_edges[mesh.nx // 2, 0]])
        pw[n] = p_now
        k_final = (k_elements(k0_elem, alpha, T, mesh.elems, t_ref)
                   if alpha else k0_elem)
        k_lo, k_hi = min(k_lo, float(k_final.min())), max(k_hi, float(k_final.max()))

    res = TransientResult(t=times, t_junction=tj, t_max=tmax, t_bottom=tb,
                          power=pw, method=f"FEM-{method}-k(T)-theta{theta:g}")
    return NonlinearTransientResult(
        res, method, it_total, it_max, it_total / max(n_steps, 1),
        max_energy, max_rel_energy, hist, converged, k_lo, k_hi)
