"""v1.1 深化 C：k_TIM / h 反辨识与可辨识性分析。

核心工具是 Cauer 网络的**精确阶跃响应解析解**（无需逐时间步推进）：
状态方程 C dT/dt = -G T + b，G 对称、C=diag(caps)。
对称化 S = C^{-1/2} G C^{-1/2} = V diag(lam) V^T，则

    e^{A t} = C^{-1/2} V diag(e^{-lam t}) V^T C^{1/2},  A = -C^{-1} G
    T(t) = T_ss + e^{A t} (T0 - T_ss),  T_ss = G^{-1} b

一次 eigh 即可求任意多时刻温度，使上百次最小二乘反演可在秒级完成。
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import least_squares

from .cauer import CauerNetwork
from .materials import ThermalCase, Layer


def build_case(base: ThermalCase, k_tim: float, h_conv: float,
               *, duty: float = 1.0, dt: float = 0.002,
               t_end: float = 6.0) -> ThermalCase:
    layers = list(base.layers)
    L = layers[1]
    layers[1] = Layer(L.name, L.thickness, float(k_tim), L.rc)
    return base.with_params(layers=tuple(layers), h_conv=float(h_conv),
                            duty=duty, dt=dt, t_end=t_end,
                            heat_source_width=base.width)


def exact_step_response(net: CauerNetwork, times: np.ndarray,
                        power: float, t0: float | None = None) -> np.ndarray:
    """返回 (n_times, n_nodes) 的精确温度场；t0 默认为环境温度。

    只适用于从零时刻起功率恒为 power 的阶跃（辨识实验正是这种激励）。
    """
    C = net.caps
    rootC = np.sqrt(C)
    inv_rootC = 1.0 / rootC
    # G = -A * C（A 已按行除以 caps）
    G = -net.A * C[:, None]
    S = inv_rootC[:, None] * G * inv_rootC[None, :]
    S = 0.5 * (S + S.T)  # 抑制浮点不对称
    lam, V = eigh(S)
    lam = np.maximum(lam, 0.0)
    b = net.g_amb * net.case.t_amb + power * net.q_pattern
    T_ss = np.linalg.solve(G, b)
    T_init = np.full(net.n, net.case.t_amb if t0 is None else t0)
    d = T_init - T_ss
    # e^{At} d = C^{-1/2} V diag(exp(-lam t)) V^T C^{1/2} d
    w = V.T @ (rootC * d)
    out = np.empty((len(times), net.n))
    for i, t in enumerate(times):
        out[i] = T_ss + inv_rootC * (V @ (np.exp(-lam * t) * w))
    return out


# ------------------------------------------------------------------ 观测集
OBSERVABLES = ("steady", "step_top", "step_top_bottom")
POWER_LEVELS = (0.5, 1.0, 1.5)
STEP_TIMES = np.geomspace(0.002, 5.0, 40)


def make_network(case: ThermalCase, n: int = 24) -> CauerNetwork:
    return CauerNetwork(case, tuple([n] * case.n_layers))


def forward(theta: np.ndarray, base: ThermalCase, observable: str,
            n: int = 24) -> np.ndarray:
    """theta = [ln(k_TIM), ln(h)]；返回与观测集对应的扁平化温度向量。"""
    k_tim, h = np.exp(theta)
    case = build_case(base, k_tim, h)
    net = make_network(case, n)
    P0 = base.power_per_length
    if observable == "steady":
        return np.array([net.steady(p * P0)[net.junction_index]
                         for p in POWER_LEVELS])
    field = exact_step_response(net, STEP_TIMES, P0)
    top = field[:, net.junction_index]
    if observable == "step_top":
        return top
    return np.concatenate([top, field[:, net.bottom_index]])


def synthesize(base: ThermalCase, k_true: float, h_true: float,
               observable: str, n: int = 24) -> np.ndarray:
    return forward(np.log([k_true, h_true]), base, observable, n)


def fit(obs: np.ndarray, base: ThermalCase, observable: str,
        x0=(np.log(3.0), np.log(3000.0)), n: int = 24) -> dict:
    def res(th):
        return forward(th, base, observable, n) - obs
    sol = least_squares(res, np.asarray(x0, dtype=float),
                        bounds=([np.log(0.2), np.log(50.0)],
                                [np.log(200.0), np.log(2.0e6)]),
                        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=200)
    return {"theta": sol.x, "k_tim": float(np.exp(sol.x[0])),
            "h": float(np.exp(sol.x[1])), "cost": float(sol.cost),
            "success": bool(sol.success), "nfev": int(sol.nfev)}


def fisher(theta: np.ndarray, base: ThermalCase, observable: str,
           sigma: float = 1.0, n: int = 24, eps: float = 1e-5) -> dict:
    """FIM = J^T J / sigma^2；返回特征值、条件数与线性化 1σ 协方差。"""
    f0 = forward(theta, base, observable, n)
    J = np.empty((f0.size, 2))
    for j in range(2):
        tp = theta.copy(); tp[j] += eps
        tm = theta.copy(); tm[j] -= eps
        J[:, j] = (forward(tp, base, observable, n)
                   - forward(tm, base, observable, n)) / (2 * eps)
    F = J.T @ J / sigma ** 2
    eig = np.linalg.eigvalsh(F)
    cov = np.linalg.pinv(F)
    return {"eigenvalues": eig,
            "condition_number": float(eig[-1] / max(eig[0], 1e-300)),
            "effective_rank": int(np.sum(eig > eig[-1] * 1e-10)),
            "cov_theta": cov,
            "sigma_theta": np.sqrt(np.diag(cov)),
            "correlation": float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])),
            "jacobian": J}
