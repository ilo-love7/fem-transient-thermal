"""v1.2 深化 F：辐射/接触热阻的非线性边界与等效 h 线性化误差。

v1.0/1.1 的底面只有**线性** Robin 对流。本模块补上两类真实封装效应：

1) 辐射（四次非线性，Stefan-Boltzmann）：
       q_rad(T) = eps*sigma*(T_K^4 - T_amb,K^4)   [W/m^2]
   底边用 3 点 Gauss 积分装配残差与切线雅可比（T 沿边线性 → T^4 为四次，
   3 点 Gauss 对五次以下精确，故残差/雅可比均无积分误差）。
   对比工程常用的「环境温度处线性化」 h0 = 4*eps*sigma*T_amb^3，
   给出割线精确系数 h_rad(T) = eps*sigma*(T^2+T_amb^2)*(T+T_amb)。

2) 接触热阻（面积比热阻 r_c'' [m^2 K/W] 串在 TIM 层）：
       k_eff = L_TIM / (L_TIM/k_TIM + r_c'')
   等效为降低 TIM 导热系数，从而复用已验证的 FEM 装配链。

**预设可证伪**：h_rad 只有 O(5-10) W/(m^2 K) 量级，因此在 h_conv=5000 的
强制液冷工况下辐射应几乎无效；只有自然对流/近真空封装才需计入。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from .assembly import FEMSystem, _coo_accum
from .materials import ThermalCase
from .mesh import StructuredMesh

STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m^2 K^4)
KELVIN_0 = 273.15

# 3 点 Gauss-Legendre（区间 [0,1]）：对 5 次以下多项式精确
_GAUSS_XI = np.array([0.5 * (1.0 - np.sqrt(3.0 / 5.0)), 0.5,
                      0.5 * (1.0 + np.sqrt(3.0 / 5.0))])
_GAUSS_W = np.array([5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0])  # 已含 1/2 映射因子


def celsius_to_kelvin(t_c) -> np.ndarray:
    return np.asarray(t_c, dtype=float) + KELVIN_0


def radiative_flux(t_c, t_amb_c: float, epsilon: float,
                   sigma: float = STEFAN_BOLTZMANN):
    """净辐射热流密度 [W/m^2]，正号表示向外散热。"""
    tk = celsius_to_kelvin(t_c)
    ta = celsius_to_kelvin(t_amb_c)
    return epsilon * sigma * (tk ** 4 - ta ** 4)


def radiative_conductance_secant(t_c, t_amb_c: float, epsilon: float,
                                 sigma: float = STEFAN_BOLTZMANN):
    """精确割线系数：q_rad = h_rad(T)*(T - T_amb)。"""
    tk = celsius_to_kelvin(t_c)
    ta = celsius_to_kelvin(t_amb_c)
    return epsilon * sigma * (tk ** 2 + ta ** 2) * (tk + ta)


def radiative_conductance_local(t_c, epsilon: float,
                                sigma: float = STEFAN_BOLTZMANN):
    """切线系数 dq/dT = 4*eps*sigma*T_K^3。"""
    return 4.0 * epsilon * sigma * celsius_to_kelvin(t_c) ** 3


def linearization_error(t_c, t_amb_c: float, epsilon: float) -> dict:
    """「在 T_amb 处线性化」相对精确四次律的误差。"""
    q_exact = radiative_flux(t_c, t_amb_c, epsilon)
    h0 = float(radiative_conductance_local(t_amb_c, epsilon))
    q_lin = h0 * (np.asarray(t_c, dtype=float) - t_amb_c)
    denom = np.maximum(np.abs(q_exact), 1e-12)
    return {
        "h_linearized_W_m2K": h0,
        "h_secant_W_m2K": np.asarray(radiative_conductance_secant(
            t_c, t_amb_c, epsilon)),
        "q_exact_W_m2": np.asarray(q_exact),
        "q_linearized_W_m2": q_lin,
        "rel_err": np.asarray((q_lin - q_exact) / denom),
    }


def radiation_boundary(mesh: StructuredMesh, T: np.ndarray,
                       epsilon: float, t_amb_c: float,
                       sigma: float = STEFAN_BOLTZMANN):
    """底边辐射残差向量、切线雅可比、总辐射散热功率 [W/m]。

    节点温度沿边线性插值：T(ξ) = (1-ξ)T_i + ξ T_j。
    """
    n = mesh.nnode
    R = np.zeros(n)
    rows: list = []
    cols: list = []
    vals: list = []
    q_total = 0.0
    ta_k = celsius_to_kelvin(t_amb_c)
    for edge in mesh.bottom_edges:
        p = mesh.nodes[edge]
        L = float(np.linalg.norm(p[1] - p[0]))
        ti, tj = float(T[edge[0]]), float(T[edge[1]])
        for xi, w in zip(_GAUSS_XI, _GAUSS_W):
            n0, n1 = 1.0 - xi, xi
            tk = celsius_to_kelvin(n0 * ti + n1 * tj)
            q = epsilon * sigma * (tk ** 4 - ta_k ** 4)
            dq = 4.0 * epsilon * sigma * tk ** 3
            c = L * w
            R[edge[0]] += c * q * n0
            R[edge[1]] += c * q * n1
            q_total += c * q
            jac = c * dq * np.array([[n0 * n0, n0 * n1],
                                     [n0 * n1, n1 * n1]])
            _coo_accum(rows, cols, vals, edge, jac)
    J = coo_matrix((np.concatenate(vals) if vals else np.zeros(0),
                    (np.concatenate(rows) if rows else np.zeros(0, dtype=int),
                     np.concatenate(cols) if cols else np.zeros(0, dtype=int))),
                   shape=(n, n)).tocsr()
    return R, J, float(q_total)


def linearized_radiation_matrix(mesh: StructuredMesh, h_rad: float,
                                t_amb_c: float):
    """等效常系数 Robin 边界 h_rad*(T-T_amb) 的刚度项与常数载荷。"""
    rows: list = []
    cols: list = []
    vals: list = []
    f = np.zeros(mesh.nnode)
    for edge in mesh.bottom_edges:
        p = mesh.nodes[edge]
        L = float(np.linalg.norm(p[1] - p[0]))
        _coo_accum(rows, cols, vals, edge,
                   h_rad * L / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]]))
        f[edge] += h_rad * t_amb_c * L / 2.0
    H = coo_matrix((np.concatenate(vals),
                    (np.concatenate(rows), np.concatenate(cols))),
                   shape=(mesh.nnode, mesh.nnode)).tocsr()
    return H, f


@dataclass
class RadiativeResult:
    field: np.ndarray
    t_junction: float
    iterations: int
    history: list[float]
    converged: bool
    q_rad_W_per_m: float
    q_conv_W_per_m: float
    radiation_share: float


def _convection_power(sys: FEMSystem, T: np.ndarray) -> float:
    """1^T(K T - f_conv) = h∫(T-T_amb) ds（K_cond 对称且行和为零）。"""
    return float(np.sum(sys.K @ T - sys.f_conv))


def solve_radiative_steady(sys: FEMSystem, case: ThermalCase, epsilon: float,
                           *, power: float | None = None, tol: float = 1e-9,
                           max_iter: int = 40,
                           sigma: float = STEFAN_BOLTZMANN) -> RadiativeResult:
    """Newton 求解含四次辐射边界的稳态场（初值取无辐射线性解）。"""
    P = case.power_per_length if power is None else power
    f = sys.load(P)
    T = splu(sys.K.tocsc()).solve(f)  # 无辐射解作为初值
    hist: list[float] = []
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        R_rad, J_rad, q_rad = radiation_boundary(
            sys.mesh, T, epsilon, case.t_amb, sigma)
        R = np.asarray(sys.K @ T) - f + R_rad
        res = float(np.max(np.abs(R)))
        hist.append(res)
        if res < tol:
            converged = True
            break
        J = (sys.K + J_rad).tocsc()
        delta = splu(J).solve(R)
        # T^4 强非线性：残差下降才接受，否则折半步长（保证高温工况收敛）
        damp = 1.0
        for _ in range(12):
            trial = T - damp * delta
            R_trial = (np.asarray(sys.K @ trial) - f
                       + radiation_boundary(sys.mesh, trial, epsilon,
                                            case.t_amb, sigma)[0])
            if np.max(np.abs(R_trial)) < res:
                T = trial
                break
            damp *= 0.5
        else:
            T = T - damp * delta
    R_rad, _, q_rad = radiation_boundary(sys.mesh, T, epsilon, case.t_amb, sigma)
    q_conv = _convection_power(sys, T)
    return RadiativeResult(
        T, float(T[sys.mesh.junction_node]), it, hist, converged,
        q_rad, q_conv, float(q_rad / max(q_rad + q_conv, 1e-12)))


def solve_linearized_radiative_steady(sys: FEMSystem, case: ThermalCase,
                                      epsilon: float, *,
                                      power: float | None = None,
                                      sigma: float = STEFAN_BOLTZMANN) -> float:
    """工程近似：在 T_amb 处线性化 h0=4eps*sigma*T_amb^3，一次线性求解。"""
    P = case.power_per_length if power is None else power
    h0 = float(radiative_conductance_local(case.t_amb, epsilon, sigma))
    H, f_extra = linearized_radiation_matrix(sys.mesh, h0, case.t_amb)
    K = (sys.K + H).tocsc()
    T = splu(K).solve(sys.load(P) + f_extra)
    return float(T[sys.mesh.junction_node])


def effective_layer_k(k: float, thickness: float,
                      r_contact_specific: float) -> float:
    """接触热阻折算为等效导热系数：k_eff = L/(L/k + r_c'')。"""
    if r_contact_specific < 0:
        raise ValueError("r_c'' 必须非负")
    return float(thickness / (thickness / k + r_contact_specific))


def contact_rise_analytic(case: ThermalCase, r_contact_specific: float) -> dict:
    """1D 体积热源口径下，TIM 层串入接触热阻后的稳态结温与热阻分解。"""
    tim = case.layers[1]
    k_eff = effective_layer_k(tim.k, tim.thickness, r_contact_specific)
    r_cond = sum(L.thickness / ((k_eff if L.name == tim.name else L.k) * case.width)
                 for L in case.layers[:-1])
    r_eff = r_cond + case.die.thickness / (2.0 * case.die.k * case.width) + case.r_conv
    return {
        "k_eff_W_mK": k_eff,
        "r_eff_K_m_per_W": r_eff,
        "t_junction_C": case.t_amb + case.power_per_length * r_eff,
        "rise_vs_no_contact_K": case.power_per_length * (r_eff - case.r_effective),
    }


def case_with_contact(case: ThermalCase, r_contact_specific: float) -> ThermalCase:
    """把接触热阻折算进 TIM 层，返回新工况（可送进同一 FEM 装配链）。"""
    from dataclasses import replace

    tim = case.layers[1]
    k_eff = effective_layer_k(tim.k, tim.thickness, r_contact_specific)
    layers = tuple(replace(L, k=k_eff) if L.name == tim.name else L
                   for L in case.layers)
    return replace(case, layers=layers)
