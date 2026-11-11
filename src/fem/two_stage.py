"""v1.2 深化 E：给扩散热阻补热容的「两阶段动态网络」。

v1.1 `spreading.add_static_spreading` 把扩散区当作**零热容纯热阻**：
T_j = T_1D + P(t)*R_sp。慢脉冲极准，快脉冲则高估摆幅，报告 §6.6 给出
有效边界 T/tau_spread ≳ 35。本模块把扩散区升级为一阶 RC 极点：

    C_sp d(ΔT)/dt = P(t) - ΔT/R_sp ,     T_j = T_1D,top + ΔT

静态极限 ΔT = P*R_sp 与 v1.1 严格一致；动态时该极点低通滤波，
恰好补上被静态模型高估的那部分摆幅。

数值上把「扩散温差」作为第 0 个状态并入精细 Cauer 网络的状态空间，
仍是分段常数功率下的**矩阵指数精确推进**（无时间离散误差）：

    A_aug = diag(-1/(R_sp*C_sp), A_Cauer)
    b_aug(P) = [P/C_sp ; b_Cauer(P)]

本模块全部为线性系统，因此「单极点能否跨周期泛化」是一个可证伪命题：
若由某一周期标定的 tau 在其它周期失效，说明 constriction 区是分布参数
系统而非单极点（见脚本 11 的留一验证）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from .cauer import CauerNetwork
from .materials import ThermalCase
from .nonlinear import rc_periodic_closed_form
from .solver import TransientResult, periodic_steady_metrics, power_pulse
from .spreading import equivalent_ladder_case


def diffusion_time_constant(r_sp: float, c_sp: float) -> float:
    """tau_sp = R_sp * C_sp [s]；c_sp <= 0 时无动态（退化为静态修正）。"""
    if r_sp < 0 or c_sp < 0:
        raise ValueError("R_sp 与 C_sp 必须非负")
    return float(r_sp * c_sp)


def diffusion_pole_closed_form(power: float, r_sp: float, tau: float,
                               period: float, duty: float) -> dict:
    """单极点扩散温差在周期方波下的精确峰/谷/摆幅（解析对照）。"""
    return rc_periodic_closed_form(power, r_sp, tau, 0.0, period, duty)


@dataclass
class TwoStageNetwork:
    """精细 Cauer 网络 + 一个带热容的扩散极点。"""

    base: CauerNetwork
    r_sp: float
    c_sp: float

    def __post_init__(self):
        if self.r_sp <= 0 or self.c_sp <= 0:
            raise ValueError("动态扩散修正要求 R_sp>0 且 C_sp>0")
        n = self.base.n
        self.n = n + 1
        self.tau = diffusion_time_constant(self.r_sp, self.c_sp)
        A = np.zeros((self.n, self.n))
        A[0, 0] = -1.0 / self.tau
        A[1:, 1:] = self.base.A
        self.A = A
        self.junction_index = 0  # 增广后第 0 状态即 ΔT
        self.top_index = 1       # 原网络顶节点
        self.bottom_index = self.n - 1

    def b_vec(self, power: float) -> np.ndarray:
        b = np.empty(self.n)
        b[0] = power / self.c_sp
        b[1:] = self.base.b_vec(power)
        return b

    def propagator(self, power: float, dt: float) -> np.ndarray:
        m = self.n
        M = np.zeros((m + 1, m + 1))
        M[:m, :m] = self.A
        M[:m, m] = self.b_vec(power)
        return expm(M * dt)

    def steady(self, power: float) -> np.ndarray:
        return np.linalg.solve(-self.A, self.b_vec(power))

    def run(self, case, *, dt: float | None = None,
            t_end: float | None = None, power_fn=None) -> TransientResult:
        # AxisymCase 没有 power_per_length：折成等效单位深度条形工况
        c = equivalent_ladder_case(case) if hasattr(case, "area") else case
        dt = c.dt if dt is None else dt
        t_end = c.t_end if t_end is None else t_end
        pfn = power_pulse if power_fn is None else power_fn
        n_steps = int(round(t_end / dt))
        times = np.arange(n_steps + 1) * dt
        props = {p: self.propagator(p, dt)
                 for p in (0.0, c.power_per_length)}

        T = np.full(self.n, c.t_amb)
        T[0] = 0.0  # 扩散温差初值为 0（不是环境温度）
        tj = np.empty(n_steps + 1)
        tmax = np.empty(n_steps + 1)
        tb = np.empty(n_steps + 1)
        pw = np.empty(n_steps + 1)
        tj[0] = T[0] + T[self.top_index]
        tmax[0] = tj[0]
        tb[0] = T[self.bottom_index]
        pw[0] = pfn(0.0, c) if power_fn is None else pfn(0.0)

        ext = np.concatenate([T, [1.0]])
        for k in range(1, n_steps + 1):
            p = pfn(times[k], c) if power_fn is None else pfn(times[k])
            P = props.get(p)
            if P is None:
                P = self.propagator(p, dt)
                props[p] = P
            ext = P @ ext
            T = ext[:self.n]
            tj[k] = T[0] + T[self.top_index]
            tmax[k] = tj[k]
            tb[k] = T[self.bottom_index]
            pw[k] = p

        return TransientResult(t=times, t_junction=tj, t_max=tmax,
                               t_bottom=tb, power=pw,
                               method=f"Cauer-N{self.base.n}+RCsp(tau={self.tau:.4g}s)")


def build_two_stage(case: ThermalCase, r_sp: float, tau: float,
                    n: int = 64) -> TwoStageNetwork:
    """由 (R_sp, tau) 构造网络：C_sp = tau/R_sp。"""
    if tau <= 0:
        raise ValueError("tau 必须为正")
    ladder = equivalent_ladder_case(case) if hasattr(case, "area") else case
    base = CauerNetwork(ladder, tuple([n] * ladder.n_layers))
    return TwoStageNetwork(base, r_sp, tau / r_sp)


def two_stage_swing(case, r_sp: float, tau: float,
                    period: float, duty: float, *, n: int = 64,
                    n_steps_per_period: int = 60,
                    n_periods: int = 8,
                    dt: float | None = None,
                    t_end: float | None = None,
                    return_result: bool = False):
    """给定 tau 的周期稳态指标。

    dt/t_end 显式给出时可**精确复刻 v1.1 的实验协议**（快脉冲下 8 个周期
    未必够到达周期稳态，必须与 FEM 参考用同一时间窗才可比）。
    """
    from dataclasses import replace

    dt = period / n_steps_per_period if dt is None else dt
    t_end = n_periods * period if t_end is None else t_end
    c = replace(case, pulse_period=period, duty=duty, dt=dt, t_end=t_end)
    net = build_two_stage(c, r_sp, tau, n=n)
    res = net.run(c, dt=dt, t_end=t_end)
    met = periodic_steady_metrics(res, c)
    return (met, res) if return_result else met


def fit_tau_from_swing(case, r_sp: float, swing_target: float,
                       period: float, duty: float, *, n: int = 64,
                       tau_lo: float = 1e-6, tau_hi: float = 10.0,
                       n_iter: int = 40, dt: float | None = None,
                       t_end: float | None = None) -> dict:
    """二分标定 tau 使两阶段模型摆幅等于 FEM 摆幅。

    摆幅随 tau 单调递减（tau→0 退化为静态修正摆幅最大，tau→∞ 时扩散极点
    被完全滤平、摆幅只剩一维网络部分），故二分稳定。
    """
    def f(tau):
        return two_stage_swing(case, r_sp, tau, period, duty, n=n,
                               dt=dt, t_end=t_end)["swing"]

    lo, hi = tau_lo, tau_hi
    f_lo, f_hi = f(lo), f(hi)
    if not (f_lo >= swing_target >= f_hi):
        return {"tau": float("nan"), "swing_lo": float(f_lo),
                "swing_hi": float(f_hi), "bracket_ok": False,
                "n_eval": 2}
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if f(mid) > swing_target:
            lo = mid
        else:
            hi = mid
    tau = 0.5 * (lo + hi)
    return {"tau": float(tau), "swing_lo": float(f_lo),
            "swing_hi": float(f_hi), "bracket_ok": True,
            "n_eval": n_iter + 2,
            "c_sp_J_per_K": float(tau / r_sp)}
