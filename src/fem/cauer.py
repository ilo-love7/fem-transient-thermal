"""Cauer 热阻网络：精细 N 节/层 RC 梯子与单节集总模型。

离散采用「界面节点」格式（标准集中质量有限体积 / Cauer 梯子）：
  - 每层切成 N 个节，每节热阻 r_seg = L/(N k W)，热容 c_seg = rho*c*L*W/N；
  - 界面节点热容 = 相邻两半节热容之和（端节点取半节），故顶界面节点
    与 FEM 顶面测点严格对齐；
  - 相邻界面节点之间为完整节热阻，底界面节点经 R_conv=1/(h W) 接环境，
    顶界面绝热；热源按梯形权重分到芯片层各节两端界面节点。

状态方程  C dT/dt = -G T + g_amb*T_amb + q(t)
分段常数功率用增广矩阵指数精确推进：[T;1] <- expm([[A,b],[0,0]]*dt) [T;1]。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from .materials import ThermalCase
from .solver import TransientResult, power_pulse


@dataclass
class CauerNetwork:
    case: ThermalCase
    n_per_layer: tuple[int, ...]

    def __post_init__(self):
        c = self.case
        # 层序从顶（芯片）到底（散热器基座）
        top_down = list(reversed(c.layers))
        reps = list(reversed(self.n_per_layer))

        r_seg: list[float] = []
        c_seg: list[float] = []
        die_seg: list[bool] = []
        for L, n in zip(top_down, reps):
            r_seg.extend([L.thickness / (n * L.k * c.width)] * n)
            c_seg.extend([L.rc * L.thickness * c.width / n] * n)
            die_seg.extend([L.name == c.die.name] * n)
        r_seg = np.asarray(r_seg)
        c_seg = np.asarray(c_seg)
        die_seg = np.asarray(die_seg)
        n_seg = len(r_seg)

        # 界面节点 (n_seg+1 个)：c_node[i] = (c_seg[i-1]+c_seg[i])/2
        self.caps = np.zeros(n_seg + 1)
        self.caps[0] = c_seg[0] / 2.0
        self.caps[-1] = c_seg[-1] / 2.0
        self.caps[1:-1] = 0.5 * (c_seg[:-1] + c_seg[1:])
        self.n = n_seg + 1
        self.r_between = r_seg
        self.r_conv = 1.0 / (c.h_conv * c.width)

        # 单位功率的热源模式：芯片层各节 P/N_die 均分到两端节点各一半
        self.q_pattern = np.zeros(self.n)
        idx = np.nonzero(die_seg)[0]
        for s in idx:
            self.q_pattern[s] += 0.5 / idx.size
            self.q_pattern[s + 1] += 0.5 / idx.size
        self.junction_index = 0
        self.bottom_index = self.n - 1

        # 热导矩阵 G
        G = np.zeros((self.n, self.n))
        self.g_amb = np.zeros(self.n)
        for i in range(n_seg):
            g = 1.0 / self.r_between[i]
            G[i, i] += g
            G[i + 1, i + 1] += g
            G[i, i + 1] -= g
            G[i + 1, i] -= g
        gb = 1.0 / self.r_conv
        G[self.n - 1, self.n - 1] += gb
        self.g_amb[self.n - 1] = gb
        self.A = -G / self.caps[:, None]
        self.C_inv = 1.0 / self.caps

    def b_vec(self, power: float) -> np.ndarray:
        return self.C_inv * (self.g_amb * self.case.t_amb + power * self.q_pattern)

    def propagator(self, power: float, dt: float) -> np.ndarray:
        n = self.n
        M = np.zeros((n + 1, n + 1))
        M[:n, :n] = self.A
        M[:n, n] = self.b_vec(power)
        return expm(M * dt)

    def steady(self, power: float) -> np.ndarray:
        return np.linalg.solve(-self.A, self.b_vec(power))

    def run(self, *, dt: float | None = None, t_end: float | None = None,
            snapshot_times: tuple[float, ...] = (), power_fn=None) -> TransientResult:
        c = self.case
        dt = c.dt if dt is None else dt
        t_end = c.t_end if t_end is None else t_end
        pfn = power_pulse if power_fn is None else power_fn
        n_steps = int(round(t_end / dt))
        times = np.arange(n_steps + 1) * dt
        props = {p: self.propagator(p, dt) for p in (0.0, c.power_per_length)}

        T = np.full(self.n, c.t_amb)
        tj = np.empty(n_steps + 1)
        tmax = np.empty(n_steps + 1)
        tb = np.empty(n_steps + 1)
        pw = np.empty(n_steps + 1)
        tj[0], tmax[0], tb[0] = T[0], T.max(), T[self.bottom_index]
        pw[0] = pfn(0.0, c) if power_fn is None else pfn(0.0)
        snaps: dict[float, np.ndarray] = {}
        snap_set = {float(s) for s in snapshot_times}
        if 0.0 in snap_set:
            snaps[0.0] = T.copy()

        ext = np.concatenate([T, [1.0]])
        for k in range(1, n_steps + 1):
            p = pfn(times[k], c) if power_fn is None else pfn(times[k])
            P = props.get(p)
            if P is None:
                P = self.propagator(p, dt)
                props[p] = P
            ext = P @ ext
            T = ext[:self.n]
            tj[k], tmax[k], tb[k] = T[0], T.max(), T[self.bottom_index]
            pw[k] = p
            if float(times[k]) in snap_set:
                snaps[float(times[k])] = T.copy()

        return TransientResult(t=times, t_junction=tj, t_max=tmax, t_bottom=tb,
                               power=pw, snapshots=snaps,
                               method=f"Cauer-N{'-'.join(map(str, self.n_per_layer))}")


def fine_network(case: ThermalCase, n: int = 64) -> CauerNetwork:
    return CauerNetwork(case, tuple([n] * case.n_layers))


def lumped_network(case: ThermalCase) -> CauerNetwork:
    return CauerNetwork(case, tuple([1] * case.n_layers))
