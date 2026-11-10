"""后处理：曲线误差、热瞬态指标、扩散热阻、参数扫描汇总。"""
from __future__ import annotations

import numpy as np

from .materials import ThermalCase
from .solver import (TransientResult, peak_over_last_cycle,
                     periodic_steady_metrics, trough_over_last_cycle)


def curve_error_full(a: TransientResult, b: TransientResult, period: float,
                     t_amb: float = 0.0) -> dict:
    """完整误差指标（b 为参考解）。

    注意：峰值温度是绝对量（含环境温度），直接拿它当分母会把误差压小约
    (T_peak - T_amb)/T_peak 倍。故另给 peak_rise_rel_err_pct 以**温升**
    为分母；瞬态集总模型失真最敏感的指标是 swing_rel_err_pct（摆幅）。
    """
    if len(a.t) != len(b.t) or not np.allclose(a.t, b.t):
        raise ValueError("时间网格不一致")
    x, y = a.t_junction, b.t_junction
    amp = float(max(y.max() - y.min(), 1e-9))
    rise = float(max(y.max() - t_amb, 1e-9))
    pa = peak_over_last_cycle(a.t, x, period)
    pb = peak_over_last_cycle(b.t, y, period)
    ta = trough_over_last_cycle(a.t, x, period)
    tb = trough_over_last_cycle(b.t, y, period)
    return {
        "max_abs_K": float(np.max(np.abs(x - y))),
        "rmse_K": float(np.sqrt(np.mean((x - y) ** 2))),
        "rmse_rel_amp_pct": float(100 * np.sqrt(np.mean((x - y) ** 2)) / amp),
        "peak_a_K": pa,
        "peak_b_K": pb,
        "peak_rel_err_pct": float(100 * abs(pa - pb) / max(pb, 1e-9)),
        "peak_rise_rel_err_pct": float(100 * abs(pa - pb) / rise),
        "swing_a_K": pa - ta,
        "swing_b_K": pb - tb,
        "swing_rel_err_pct": float(100 * abs((pa - ta) - (pb - tb)) / max(pb - tb, 1e-9)),
    }


def thermal_metrics(res: TransientResult, case: ThermalCase) -> dict:
    m = periodic_steady_metrics(res, case)
    m["rise_steady_analytic"] = case.t_junction_steady
    m["tau_cauer_s"] = case.cauer_time_constant
    m["method"] = res.method
    return m


def spreading_resistance(case: ThermalCase, t_j_2d: float, t_j_1d: float) -> dict:
    """二维横向扩散热阻：R_spread = (T_j,2D - T_j,1D)/P' (K m / W)。"""
    r_spread = (t_j_2d - t_j_1d) / case.power_per_length
    return {
        "delta_T_K": t_j_2d - t_j_1d,
        "r_spread": r_spread,
        "r_total_1d": case.r_total,
        "r_spread_ratio_pct": 100 * r_spread / case.r_total,
    }


def fit_first_order_tau(t: np.ndarray, y: np.ndarray, t0: float = 0.0) -> float:
    """从阶跃上升段拟合一阶时间常数：ln(T_ss - T) vs t 线性斜率。"""
    mask = t >= t0
    tt, yy = t[mask], y[mask]
    y_ss = float(yy[-1])
    d = y_ss - yy
    good = d > 0.02 * max(d.max(), 1e-9)
    if good.sum() < 5:
        return float("nan")
    slope = np.polyfit(tt[good], np.log(d[good]), 1)[0]
    return float(-1.0 / slope) if slope < 0 else float("nan")


def sweep_table(rows: list[dict]) -> str:
    """简单 markdown 表输出（键取各行并集）。"""
    keys = list(dict.fromkeys(k for r in rows for k in r))
    out = ["| " + " | ".join(keys) + " |", "|" + "|".join(["---"] * len(keys)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(
            f"{r[k]:.4g}" if isinstance(r.get(k), (int, float)) else str(r.get(k, ""))
            for k in keys) + " |")
    return "\n".join(out)
