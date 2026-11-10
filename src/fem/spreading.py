"""v1.1 深化 A（续）：扩散热阻闭合模型 + 带扩散修正的集总 RC 网络。

v1.0 只报告「Cauer 在二维局部热源下低估 22.84 K」却没有给修法。
这里用轴对称 FEM 标定一个两参数闭合式，并把 R_sp 串进集总网络。

闭合式（无量纲 q = a/R）：

    R_sp(q) = R_1D * ( q^{-gamma(q)} - 1 ),   gamma(q) = g0 + g1*(1-q)

- q=1 时自动 R_sp=0（全宽热源严格退化为一维），无需额外约束；
- gamma(q) 是「有效散热面积指数」：R_total = R_1D * q^{-gamma}；
- g0/g1 由少量 q 的 FEM 结果线性最小二乘标定，其余 q 做留一验证。

注意：这是**工程闭合式**而非解析解；报告里同时给出经典半空间公式
8/(3*pi^2*k*a) 的失准倍数，说明为什么不能直接套半无限体解。
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .axisym import AxisymCase, semi_infinite_spreading, spreading_resistance_abs
from .cauer import CauerNetwork
from .materials import ThermalCase
from .solver import TransientResult


@dataclass(frozen=True)
class SpreadingClosure:
    g0: float
    g1: float
    r_1d: float
    calibration_q: tuple[float, ...]

    def gamma(self, q) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return self.g0 + self.g1 * (1.0 - q)

    def predict(self, q) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return self.r_1d * (np.power(q, -self.gamma(q)) - 1.0)

    def t_junction(self, q, t_amb: float, power: float) -> np.ndarray:
        return t_amb + power * (self.r_1d + self.predict(q))


def fit_spreading_closure(q: np.ndarray, r_sp: np.ndarray, r_1d: float,
                          calibration_mask: np.ndarray) -> SpreadingClosure:
    """由标定子集线性拟合 gamma(q)=g0+g1(1-q)。

    gamma_obs(q) = ln(1 + R_sp/R_1D) / ln(1/q)，q=1 处跳过。
    """
    q = np.asarray(q, dtype=float)
    r_sp = np.asarray(r_sp, dtype=float)
    mask = np.asarray(calibration_mask, dtype=bool) & (q < 1.0 - 1e-12) & (r_sp > 0)
    gamma_obs = np.log(1.0 + r_sp[mask] / r_1d) / np.log(1.0 / q[mask])
    X = np.column_stack([np.ones(mask.sum()), 1.0 - q[mask]])
    g0, g1 = np.linalg.lstsq(X, gamma_obs, rcond=None)[0]
    return SpreadingClosure(float(g0), float(g1), float(r_1d),
                            tuple(float(x) for x in q[mask]))


def equivalent_ladder_case(case: AxisymCase, nx: int = 8,
                           ny_per_layer: int = 8) -> ThermalCase:
    """把圆柱工况折成「单位深度 1 m」的等效条形工况（绝对单位 K/W、W）。

    width 取圆柱底面积（除以 1 m 深度），power_per_length 取总功率，
    于是 CauerNetwork 的热阻/热容数值直接等于绝对单位下的值。
    """
    return ThermalCase(
        width=case.area,
        layers=case.layers,
        h_conv=case.h_conv,
        t_amb=case.t_amb,
        power_per_length=case.total_power,
        heat_source_width=case.area,
        nx=nx,
        ny_per_layer=ny_per_layer,
        theta=case.theta,
        dt=case.dt,
        t_end=case.t_end,
        pulse_period=case.pulse_period,
        duty=case.duty,
    )


def add_static_spreading(res: TransientResult, r_sp: float) -> TransientResult:
    """把无热容的扩散热阻串在热源与网络顶节点之间：T_src = T_node + P(t)*R_sp。

    这是**最小修正**：只补稳态偏移，不给 constriction 区增加热容。
    与三维 FEM 瞬态对比即可检验「扩散区热容是否可忽略」这一假设。
    """
    return TransientResult(
        t=res.t,
        t_junction=res.t_junction + res.power * r_sp,
        t_max=res.t_max + res.power * r_sp,
        t_bottom=res.t_bottom,
        power=res.power,
        snapshots=res.snapshots,
        method=f"{res.method}+Rsp{r_sp:.4g}",
    )


def corrected_ladder(case: AxisymCase, r_sp: float, n: int = 64,
                     *, dt: float | None = None,
                     t_end: float | None = None) -> tuple[TransientResult, TransientResult]:
    """返回 (未修正, 修正后) 两条集总网络瞬态曲线。"""
    c = equivalent_ladder_case(case)
    if dt is not None or t_end is not None:
        c = replace(c, dt=c.dt if dt is None else dt,
                    t_end=c.t_end if t_end is None else t_end)
    net = CauerNetwork(c, tuple([n] * c.n_layers))
    dt = c.dt
    base = net.run(dt=dt, t_end=c.t_end)
    return base, add_static_spreading(base, r_sp)
