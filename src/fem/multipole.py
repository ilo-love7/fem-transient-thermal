"""v1.3 深化 E16：多极点 / 分布参数扩散热阻网络。

v1.2 §11.2 证伪了「单极点扩散修正」——逐周期二分标定的 tau_best 跨周期
漂移 17.06×、留一摆幅误差 47.78%。本模块用 **Foster 并联 RC 链** 描述
扩散区的阶跃热阻抗：

    Z_sp(t) = ΔT_sp(t) / P = Σ_i R_i (1 - exp(-t / tau_i)),  Σ_i R_i = R_sp

关键结构与可证伪命题：
  * 给定 tau 网格后，R_i >= 0 的拟合是**非负最小二乘（凸）**，无需非线性
    优化；因此「多极点是否优于单极点」是一个确定性、可复现的判据。
  * 用一次**阶跃** FEM 响应拟合出的固定极点组，去预测**全周期方波**的摆幅，
    是真正的零样本跨尺度泛化检验（v1.2 的单极点必须逐周期重新标定）。
  * 极点稳定性：若从不同的截断时间窗 [0,t_w] 拟合出的 tau_i 位置剧烈漂移，
    说明极点是数值 artifact（分布参数系统需要连续谱而非有限极点）。

时域推进用分段常数功率下的**精确指数更新**（非 theta 法，无时间离散误差）：

    x_i(t+dt) = exp(-dt/tau_i) * x_i(t) + R_i * (1-exp(-dt/tau_i)) * P(t)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls

from .solver import TransientResult


@dataclass(frozen=True)
class FosterNetwork:
    """并联 RC 链（Foster 形式）：电阻 R_i 与时间常数 tau_i。"""

    resistances: np.ndarray
    taus: np.ndarray

    def __post_init__(self):
        r = np.asarray(self.resistances, dtype=float)
        t = np.asarray(self.taus, dtype=float)
        if r.shape != t.shape or r.ndim != 1 or r.size == 0:
            raise ValueError("R_i 与 tau_i 必须是一维等长非空数组")
        if np.any(r < 0) or np.any(t <= 0):
            raise ValueError("要求 R_i >= 0 且 tau_i > 0")

    @property
    def n_poles(self) -> int:
        return len(self.resistances)

    @property
    def r_total(self) -> float:
        return float(np.sum(self.resistances))

    def step(self, t) -> np.ndarray:
        """阶跃热阻抗 Z(t)=ΣR_i(1-exp(-t/tau_i)) [K/W]。"""
        tt = np.atleast_1d(np.asarray(t, dtype=float))
        r, tau = self.resistances[:, None], self.taus[:, None]
        return np.sum(r * (1.0 - np.exp(-tt[None, :] / tau)), axis=0)

    def periodic_swing(self, period: float, duty: float, *,
                       n_steps_per_period: int = 200, n_periods: int = 12,
                       dt: float | None = None,
                       t_end: float | None = None) -> dict:
        """固定功率幅值 P=1 W 的方波下，Foster 链的周期稳态摆幅 [K/W]。

        用精确指数更新逐时间步推进；返回峰/谷/摆幅。dt/t_end 显式给出时
        可与 FEM 用同一时间协议对照。
        """
        if period <= 0 or not (0 < duty < 1):
            raise ValueError("period>0 且 0<duty<1")
        dt = period / n_steps_per_period if dt is None else float(dt)
        t_end = n_periods * period if t_end is None else float(t_end)
        n_steps = int(round(t_end / dt))
        times = np.arange(n_steps + 1) * dt
        decay = np.exp(-dt / self.taus)
        gain = self.resistances * (1.0 - decay)  # P=1 W
        x = np.zeros(self.n_poles)
        y = np.empty(n_steps + 1)
        y[0] = 0.0
        for k in range(1, n_steps + 1):
            phase = (times[k] % period) / period
            p = 1.0 if phase < duty else 0.0
            x = decay * x + gain * p
            y[k] = x.sum()
        last = times >= t_end - period
        peak = float(y[last].max())
        trough = float(y[last].min())
        return {"peak": peak, "trough": trough, "swing": peak - trough,
                "dt": dt, "t_end": t_end}


def log_tau_grid(t_min: float, t_max: float, n: int) -> np.ndarray:
    """对数均匀极点网格；t_min/t_max 需为正且 t_max>t_min。"""
    if not (0 < t_min < t_max) or n < 1:
        raise ValueError("要求 0<t_min<t_max 且 n>=1")
    return np.geomspace(t_min, t_max, int(n))


def fit_foster(times, impedance, taus, *, enforce_r_total: float | None = None,
               cap: float | None = None) -> dict:
    """在给定 tau 网格上用非负最小二乘拟合 R_i。

    impedance = ΔT(t)/P [K/W]。enforce_r_total 给定时，把稳态热阻作为
    额外的大权重方程加入（软约束，保持凸性）；cap 可限制单个 R_i 上界。
    """
    tt = np.asarray(times, dtype=float)
    z = np.asarray(impedance, dtype=float)
    tv = np.asarray(taus, dtype=float)
    if tt.ndim != 1 or z.shape != tt.shape or tv.ndim != 1:
        raise ValueError("times/impedance 须等长一维，taus 一维")
    A = 1.0 - np.exp(-tt[:, None] / tv[None, :])
    b = z
    if enforce_r_total is not None:
        w = 1000.0 / max(len(tt), 1) ** 0.5
        A = np.vstack([A, np.full(len(tv), w)])
        b = np.concatenate([b, [w * float(enforce_r_total)]])
    r, residual = nnls(A, b)
    if cap is not None:
        r = np.minimum(r, float(cap))
    net = FosterNetwork(r, tv)
    pred = net.step(tt)
    rmse = float(np.sqrt(np.mean((pred - z) ** 2)))
    rel = float(rmse / max(np.ptp(z), 1e-30))
    active = int(np.sum(r > max(r.max() * 1e-3, 1e-12))) if r.max() > 0 else 0
    return {
        "network": net, "r": r, "tau": tv.copy(), "residual": float(residual),
        "rmse_K_per_W": rmse, "rel_rmse": rel, "r_total": net.r_total,
        "active_poles": active, "taus_active": tv[r > max(r.max() * 1e-3, 1e-12)],
    }


def single_pole_from_step(times, impedance, *, tau_grid=None,
                          tau_lo=1e-5, tau_hi=10.0, n=120) -> dict:
    """单极点对照：固定 R=Z(∞)，在 tau 网格上最小化 RMSE（标量搜索）。"""
    tt = np.asarray(times, dtype=float)
    z = np.asarray(impedance, dtype=float)
    r_inf = float(z[-1])
    grid = log_tau_grid(tau_lo, tau_hi, n) if tau_grid is None else np.asarray(tau_grid)
    best = None
    for tau in grid:
        pred = r_inf * (1.0 - np.exp(-tt / tau))
        rmse = float(np.sqrt(np.mean((pred - z) ** 2)))
        if best is None or rmse < best["rmse_K_per_W"]:
            best = {"tau": float(tau), "r": r_inf, "rmse_K_per_W": rmse,
                    "rel_rmse": rmse / max(np.ptp(z), 1e-30)}
    best["network"] = FosterNetwork(np.array([best["r"]]), np.array([best["tau"]]))
    return best


def pole_stability(fits: list[dict], *, rtol: float = 0.25) -> dict:
    """比较不同时间窗拟合的极点位置稳定性。

    对每个拟合取相对权重最大的极点 tau，统计其跨窗的极值倍数；若 >1/rtol
    （默认 4×），判为不稳定（数值 artifact）。
    """
    taus = []
    for f in fits:
        r, tau = f["r"], f["tau"]
        if r.max() <= 0:
            continue
        taus.append(float(tau[np.argmax(r)]))
    if not taus:
        return {"stable": False, "n": 0, "tau_top": [], "spread_factor": float("inf")}
    spread = max(taus) / min(taus)
    return {"stable": bool(spread <= 1.0 / rtol), "n": len(taus),
            "tau_top": taus, "spread_factor": float(spread),
            "criterion": f"max/min <= {1.0 / rtol:.1f}"}
