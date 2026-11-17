"""v1.3 深化 E17：多工况联合辨识，破解稳态 FIM 秩 1。

v1.1 §8 发现：稳态观测下 k_TIM 与 h 的 Fisher 信息矩阵秩 1（两参数沿
「稳态热阻」方向完全相关），无论加多少稳态测点都无法同时辨识。本模块
把不同**功率水平 / 脉冲周期 / 占空比 / 观测时刻**的观测向量拼接成联合
FIM，检验「跨工况信息」是否能把有效秩提到 2。

关键可证伪命题：
  * 稳态多功率：若各功率只是缩放同一稳态温度场，则 FIM 仍秩 1（功率维度
    不带新信息）——这是一个必须验证的负结果。
  * 周期方波：不同周期 T 对扩散时间常数 tau 的敏感度不同（T ~ tau 附近
    最敏感），联合多周期应能降条件数。
  * 观测时刻：阶跃早期 vs 晚期对 k/h 的敏感度方向不同，联合可破秩。

实现复用 identification.exact_step_response / CauerNetwork 的解析能力；
周期方波用精确增广矩阵指数推进（与 v1.2 two_stage 同口径）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from .cauer import CauerNetwork
from .identification import build_case
from .materials import ThermalCase


@dataclass(frozen=True)
class Condition:
    """一个可辨识工况：稳态功率水平 + 周期方波 + 观测时刻。"""

    name: str
    power: float = 1.0          # 相对额定功率倍数
    period: float | None = None  # None = 稳态
    duty: float = 0.5
    phases: tuple[float, ...] = ()  # 周期工况的观测相位（归一化 0-1）

    @property
    def is_transient(self) -> bool:
        return self.period is not None


def steady_observations(theta: np.ndarray, base: ThermalCase,
                        powers: tuple[float, ...], n: int = 24) -> np.ndarray:
    """多功率稳态结温。k_tim/h 由 theta=[ln k, ln h] 给出。"""
    k_tim, h = np.exp(theta)
    case = build_case(base, k_tim, h)
    net = CauerNetwork(case, tuple([n] * case.n_layers))
    p0 = base.power_per_length
    return np.array([net.steady(p * p0)[net.junction_index] for p in powers])


def periodic_phase_observations(theta: np.ndarray, base: ThermalCase,
                                power: float, period: float, duty: float,
                                phases: tuple[float, ...], *,
                                n: int = 24, n_steps_per_period: int = 48,
                                n_periods: int = 10) -> np.ndarray:
    """周期方波在给定相位处的结温（周期稳态），精确矩阵指数推进。"""
    k_tim, h = np.exp(theta)
    case = build_case(base, k_tim, h)
    case = case.with_params(pulse_period=period, duty=duty)
    net = CauerNetwork(case, tuple([n] * case.n_layers))
    dt = period / n_steps_per_period
    t_end = n_periods * period
    p_on = power * base.power_per_length
    P_on = net.propagator(p_on, dt)
    P_off = net.propagator(0.0, dt)
    T = np.full(net.n, case.t_amb)
    out_steps = {int(round(ph * n_steps_per_period)) % n_steps_per_period:
                 ph for ph in phases}
    # 先跑到周期稳态（丢弃前 n_periods-1 个周期）
    total = int(round(t_end / dt))
    last_period_start = total - n_steps_per_period
    vals: dict[float, float] = {}
    for k in range(1, total + 1):
        ph_step = (k % n_steps_per_period)
        p = p_on if ph_step / n_steps_per_period < duty else 0.0
        # 增广向量 [T; 1] 使 affine 推进 P=expm([[A,b],[0,0]]*dt) 可线性作用
        aug = np.concatenate([T, [1.0]])
        T = ((P_on if p > 0 else P_off) @ aug)[:net.n]
        if k > last_period_start and ph_step in out_steps:
            vals[out_steps[ph_step]] = float(T[net.junction_index])
    return np.array([vals[ph] for ph in phases])


def joint_forward(theta: np.ndarray, base: ThermalCase,
                  conditions: tuple[Condition, ...], *, n: int = 24) -> np.ndarray:
    """按 conditions 顺序拼接所有工况的观测向量。"""
    chunks: list[np.ndarray] = []
    for c in conditions:
        if not c.is_transient:
            chunks.append(steady_observations(theta, base, (c.power,), n))
        else:
            chunks.append(periodic_phase_observations(
                theta, base, c.power, c.period, c.duty, c.phases, n=n))
    return np.concatenate(chunks)


def joint_fisher(theta: np.ndarray, base: ThermalCase,
                 conditions: tuple[Condition, ...], *, sigma: float = 1.0,
                 n: int = 24, eps: float = 1e-5) -> dict:
    """联合 FIM = Σ_c J_cᵀJ_c/σ²；返回特征值/条件数/有效秩/协方差。"""
    f0 = joint_forward(theta, base, conditions, n=n)
    J = np.empty((f0.size, 2))
    for j in range(2):
        tp = np.asarray(theta, dtype=float).copy(); tp[j] += eps
        tm = np.asarray(theta, dtype=float).copy(); tm[j] -= eps
        J[:, j] = (joint_forward(tp, base, conditions, n=n)
                   - joint_forward(tm, base, conditions, n=n)) / (2 * eps)
    F = J.T @ J / float(sigma) ** 2
    eig = np.linalg.eigvalsh(F)
    cov = np.linalg.pinv(F)
    d = np.sqrt(max(cov[0, 0] * cov[1, 1], 1e-300))
    return {
        "eigenvalues": eig,
        "condition_number": float(eig[-1] / max(eig[0], 1e-300)),
        "effective_rank": int(np.sum(eig > eig[-1] * 1e-8)),
        "cov_theta": cov,
        "sigma_theta": np.sqrt(np.maximum(np.diag(cov), 0.0)),
        "correlation": float(cov[0, 1] / d),
        "n_observations": int(f0.size),
        "jacobian": J,
    }


def condition_suite(base: ThermalCase) -> dict[str, tuple[Condition, ...]]:
    """预定义工况套件：用于对比「稳态 / 单周期 / 多周期 / 多相位」。"""
    steady = tuple(Condition(f"steady_p{p:g}", power=p) for p in (0.5, 1.0, 1.5))
    single = tuple(
        [Condition("steady_p1.0", power=1.0)]
        + [Condition("T0.5", power=1.0, period=0.5, duty=0.5, phases=(0.25, 0.75)),
           Condition("T0.05", power=1.0, period=0.05, duty=0.5, phases=(0.25, 0.75))]
    )
    multi_period = tuple(
        [Condition("steady_p1.0", power=1.0)]
        + [Condition(f"T{T:g}", power=1.0, period=T, duty=0.5,
                     phases=(0.1, 0.25, 0.5, 0.75))
           for T in (2.0, 0.5, 0.2, 0.05, 0.02)]
    )
    multi_power = tuple(
        Condition(f"T{T:g}_p{p:g}", power=p, period=T, duty=d,
                  phases=(0.25, 0.75))
        for T in (0.5, 0.05) for p in (0.5, 1.0, 1.5) for d in (0.3, 0.7)
    )
    return {
        "steady_only": steady,
        "single_period": single,
        "multi_period": multi_period,
        "multi_power_duty": multi_power,
    }
