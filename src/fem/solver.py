"""瞬态热传导 theta 法时间积分与稳态求解。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse.linalg import splu

from .assembly import FEMSystem, solve_steady
from .materials import ThermalCase


def power_pulse(t: float, case: ThermalCase) -> float:
    """方波脉冲功率：duty 占空比，t=0 起处于导通相。"""
    phase = (t % case.pulse_period) / case.pulse_period
    return case.power_per_length if phase < case.duty else 0.0


@dataclass
class TransientResult:
    t: np.ndarray
    t_junction: np.ndarray
    t_max: np.ndarray
    t_bottom: np.ndarray
    power: np.ndarray
    snapshots: dict = field(default_factory=dict)
    method: str = "FEM"


def peak_over_last_cycle(t: np.ndarray, y: np.ndarray, period: float) -> float:
    mask = t >= t[-1] - period
    return float(np.max(y[mask]))


def trough_over_last_cycle(t: np.ndarray, y: np.ndarray, period: float) -> float:
    mask = t >= t[-1] - period
    return float(np.min(y[mask]))


def run_transient(sys: FEMSystem, case: ThermalCase, *,
                  dt: float | None = None, theta: float | None = None,
                  t_end: float | None = None,
                  snapshot_times: tuple[float, ...] = (),
                  power_fn=None) -> TransientResult:
    dt = case.dt if dt is None else dt
    theta = case.theta if theta is None else theta
    t_end = case.t_end if t_end is None else t_end
    pfn = power_pulse if power_fn is None else power_fn

    K, M = sys.K, sys.M
    A = (M / dt + theta * K).tocsc()
    B = (M / dt - (1.0 - theta) * K).tocsr()
    lu = splu(A)

    n_steps = int(round(t_end / dt))
    times = np.arange(n_steps + 1) * dt
    T = np.full(sys.nnode, case.t_amb, dtype=float)
    f_prev = sys.load(pfn(0.0, case) if power_fn is None else pfn(0.0))

    tj = np.empty(n_steps + 1)
    tmax = np.empty(n_steps + 1)
    tb = np.empty(n_steps + 1)
    pw = np.empty(n_steps + 1)
    tj[0] = T[sys.mesh.junction_node]
    tmax[0] = T.max()
    tb[0] = T[sys.mesh.bottom_edges[nx_center(sys), 0]]
    pw[0] = pfn(0.0, case) if power_fn is None else pfn(0.0)

    snaps: dict[float, np.ndarray] = {}
    snap_arr = np.asarray(sorted(float(s) for s in snapshot_times))
    if snap_arr.size and np.isclose(0.0, snap_arr, atol=dt * 1e-6).any():
        snaps[0.0] = T.copy()

    for n in range(1, n_steps + 1):
        p_now = pfn(times[n], case) if power_fn is None else pfn(times[n])
        f_now = sys.load(p_now)
        rhs = B @ T + theta * f_now + (1.0 - theta) * f_prev
        T = lu.solve(rhs)
        f_prev = f_now
        tj[n] = T[sys.mesh.junction_node]
        tmax[n] = T.max()
        tb[n] = T[sys.mesh.bottom_edges[nx_center(sys), 0]]
        pw[n] = p_now
        # 用容差匹配，避免 arange 累积浮点误差导致快照静默丢失
        if snap_arr.size and np.isclose(times[n], snap_arr, atol=dt * 1e-6).any():
            snaps[round(float(times[n]), 8)] = T.copy()

    return TransientResult(t=times, t_junction=tj, t_max=tmax,
                           t_bottom=tb, power=pw, snapshots=snaps,
                           method=f"FEM-theta{theta:g}")


def nx_center(sys: FEMSystem) -> int:
    return sys.mesh.nx // 2


def steady_field(sys: FEMSystem, case: ThermalCase, power: float | None = None):
    return solve_steady(sys, case.power_per_length if power is None else power)


def periodic_steady_metrics(res: TransientResult, case: ThermalCase) -> dict:
    return {
        "t_peak": peak_over_last_cycle(res.t, res.t_junction, case.pulse_period),
        "t_trough": trough_over_last_cycle(res.t, res.t_junction, case.pulse_period),
        "t_avg": float(res.t_junction[res.t >= res.t[-1] - case.pulse_period].mean()),
        "t_max_field": float(res.t_max.max()),
        "swing": peak_over_last_cycle(res.t, res.t_junction, case.pulse_period)
        - trough_over_last_cycle(res.t, res.t_junction, case.pulse_period),
    }
