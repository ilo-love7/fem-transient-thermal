"""v1.1 深化 A：轴对称 (r-z) P1 Galerkin FEM —— 真三维扩散热阻。

v1.0 只有平面二维条形局部热源；本模块把同一多层结构放到圆柱坐标下，
热源是半径 a 的圆盘、器件半径 R，于是「侧向收缩」是真正的三维效应。

弱形式（对 dV = 2*pi*r*dr*dz 积分，底面 Robin）：

    int 2*pi*r*k grad(T).grad(v) dA + int 2*pi*r*h*T*v dr
      = int 2*pi*r*q'''*v dA + int 2*pi*r*h*T_amb*dr

所有积分对线性三角形解析精确（r 在单元内线性）：
  - 刚度：grad 常向量，int r dA = Ae*r_bar（r_bar 为重心半径）
  - 一致质量：int r N_i N_j dA = Ae/60 * (4*r_i+2*r_tot) [i=j]
                                  Ae/60 * (r_tot+r_i+r_j) [i!=j]
  - 底边 Robin：int r N_i N_j dr 用精确线性加权，不用中点近似
  - 热源：int r N_i dA = Ae/12*(r_tot+r_i)

关键验证：a/R=1（全宽热源）时问题严格退化为一维，稳态结温必须回到
1D 解析 T_amb + q''*(sum L/k + L_die/(2k_die) + 1/h)。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from .materials import Layer, ThermalCase
from .assembly import _tri_grad, _coo_accum
from .solver import TransientResult, power_pulse


@dataclass
class AxisymCase:
    """圆柱叠层功率器件：半径 R，从底往上堆叠 layers，顶面圆盘热源半径 a。"""

    radius: float
    layers: tuple[Layer, ...]
    h_conv: float
    t_amb: float
    q_flux: float            # W/m^2，全宽热源口径的面热流密度
    source_ratio: float = 1.0  # a/R
    nr: int = 60
    nz_per_layer: int = 6
    theta: float = 1.0
    dt: float = 0.01
    t_end: float = 8.0
    pulse_period: float = 2.0
    duty: float = 0.5

    @property
    def source_radius(self) -> float:
        return self.radius * self.source_ratio

    @property
    def area(self) -> float:
        return np.pi * self.radius ** 2

    @property
    def total_power(self) -> float:
        """总功率 W：保持全宽口径的面热流密度不变。"""
        return self.q_flux * self.area

    @property
    def die(self) -> Layer:
        return self.layers[-1]

    @property
    def height(self) -> float:
        return sum(L.thickness for L in self.layers)

    @property
    def y_interfaces(self) -> list[float]:
        ys = [0.0]
        for L in self.layers:
            ys.append(ys[-1] + L.thickness)
        return ys

    @property
    def r_area_specific(self) -> float:
        """面积比热阻 m^2 K/W（体积热源口径，与 v1.0 的 r_effective 一致）。"""
        r = sum(L.thickness / L.k for L in self.layers[:-1])
        r += self.die.thickness / (2.0 * self.die.k)
        return r + 1.0 / self.h_conv

    @property
    def r_1d_abs(self) -> float:
        """一维绝对热阻 K/W。"""
        return self.r_area_specific / self.area

    @property
    def t_junction_1d(self) -> float:
        """一维解析稳态结温（全宽热源基准）。"""
        return self.t_amb + self.total_power * self.r_1d_abs

    @property
    def q3(self) -> float:
        """当前 a 下的体积热源密度 W/m^3。"""
        a = self.source_radius
        return self.total_power / (np.pi * a * a * self.die.thickness)

    def with_params(self, **kw) -> "AxisymCase":
        from dataclasses import replace
        return replace(self, **kw)


def axisym_from_case(case: ThermalCase, source_ratio: float = 1.0,
                     nr: int = 60, nz_per_layer: int = 6) -> AxisymCase:
    """把 v1.0 的单位厚度二维工况映射为面积等效的圆柱工况。

    映射保持面热流密度 q''=P'/W 与各层厚度不变，圆柱半径 R=W/2，
    总功率 P=q''*pi*R^2；因此 a=R 时两种几何给出完全相同的结温。
    """
    return AxisymCase(
        radius=0.5 * case.width,
        layers=case.layers,
        h_conv=case.h_conv,
        t_amb=case.t_amb,
        q_flux=case.power_per_length / case.width,
        source_ratio=source_ratio,
        nr=nr,
        nz_per_layer=nz_per_layer,
        theta=case.theta,
        dt=case.dt,
        t_end=case.t_end,
        pulse_period=case.pulse_period,
        duty=case.duty,
    )


@dataclass
class AxisymMesh:
    radius: float
    height: float
    nr: int
    nz_per_layer: int
    n_layers: int
    nodes: np.ndarray           # (nnode,2) r,z
    elems: np.ndarray           # (nelem,3)
    elem_layer: np.ndarray
    elem_heated: np.ndarray
    bottom_edges: np.ndarray    # (nr,2)
    junction_node: int          # 顶面轴心 r=0
    nx: int                     # 兼容 run_transient 的 nx_center 接口

    @property
    def nnode(self) -> int:
        return self.nodes.shape[0]

    @property
    def nelem(self) -> int:
        return self.elems.shape[0]


def build_axisym_mesh(case: AxisymCase, nr: int | None = None,
                      nz_per_layer: int | None = None) -> AxisymMesh:
    nr = case.nr if nr is None else nr
    npl = case.nz_per_layer if nz_per_layer is None else nz_per_layer
    n_layers = len(case.layers)
    ys = case.y_interfaces

    z_nodes = [0.0]
    layer_of_row = []
    for j, L in enumerate(case.layers):
        for r in range(npl):
            z_nodes.append(ys[j] + (r + 1) * L.thickness / npl)
            layer_of_row.append(j)
    z_nodes = np.asarray(z_nodes)
    r_nodes = np.linspace(0.0, case.radius, nr + 1)
    nz_total = npl * n_layers

    rr, zz = np.meshgrid(r_nodes, z_nodes)
    nodes = np.column_stack([rr.ravel(), zz.ravel()])

    elems, elem_layer, elem_heated = [], [], []
    a = case.source_radius
    die_layer = n_layers - 1
    for row in range(nz_total):
        j = layer_of_row[row]
        for col in range(nr):
            v00 = row * (nr + 1) + col
            v10 = v00 + 1
            v01 = v00 + (nr + 1)
            v11 = v01 + 1
            if (row + col) % 2 == 0:
                tris = ((v00, v10, v11), (v00, v11, v01))
            else:
                tris = ((v00, v10, v01), (v10, v11, v01))
            for tri in tris:
                elems.append(tri)
                elem_layer.append(j)
                rc = float(nodes[list(tri), 0].mean())
                elem_heated.append(j == die_layer and rc <= a)
    elems = np.asarray(elems, dtype=np.int64)
    elem_layer = np.asarray(elem_layer, dtype=np.int64)
    elem_heated = np.asarray(elem_heated, dtype=bool)

    bottom_edges = np.array([[i, i + 1] for i in range(nr)], dtype=np.int64)
    junction_node = nz_total * (nr + 1)  # 顶面 r=0
    return AxisymMesh(case.radius, case.height, nr, npl, n_layers,
                      nodes, elems, elem_layer, elem_heated,
                      bottom_edges, junction_node, nr)


@dataclass
class AxisymSystem:
    K: object
    M: object
    f_conv: np.ndarray
    f_heat_unit: np.ndarray   # 单位总功率 1 W 的载荷
    mesh: AxisymMesh
    case: AxisymCase
    heated_area: float        # 参与热源的 (r,z) 面积加权和，用于校验

    @property
    def nnode(self) -> int:
        return self.mesh.nnode

    def load(self, power: float) -> np.ndarray:
        return self.f_conv + power * self.f_heat_unit


def assemble_axisym(case: AxisymCase, mesh: AxisymMesh | None = None) -> AxisymSystem:
    mesh = mesh or build_axisym_mesh(case)
    nodes, elems = mesh.nodes, mesh.elems
    nnode = mesh.nnode
    k_of = [L.k for L in case.layers]
    rc_of = [L.rc for L in case.layers]
    q3_unit = 1.0 / (np.pi * case.source_radius ** 2 * case.die.thickness)

    rows_K, cols_K, vals_K = [], [], []
    rows_M, cols_M, vals_M = [], [], []
    f_heat = np.zeros(nnode)
    weighted_area = 0.0

    for e, tri in enumerate(elems):
        p = nodes[tri]
        Ae, grad = _tri_grad(p)
        r_tot = float(p[:, 0].sum())
        lay = mesh.elem_layer[e]
        r_bar = r_tot / 3.0
        _coo_accum(rows_K, cols_K, vals_K, tri,
                   2.0 * np.pi * r_bar * k_of[lay] * Ae * (grad @ grad.T))

        # 精确 r 加权一致质量
        M_e = np.empty((3, 3))
        for i in range(3):
            for j in range(3):
                M_e[i, j] = (4.0 * p[i, 0] + 2.0 * r_tot) if i == j \
                    else (r_tot + p[i, 0] + p[j, 0])
        _coo_accum(rows_M, cols_M, vals_M, tri,
                   2.0 * np.pi * rc_of[lay] * Ae / 60.0 * M_e)

        if mesh.elem_heated[e]:
            f_heat[tri] += 2.0 * np.pi * q3_unit * Ae / 12.0 * (r_tot + p[:, 0])
            weighted_area += Ae * r_bar

    K = coo_matrix((np.concatenate(vals_K),
                    (np.concatenate(rows_K), np.concatenate(cols_K))),
                   shape=(nnode, nnode)).tocsr()
    M = coo_matrix((np.concatenate(vals_M),
                    (np.concatenate(rows_M), np.concatenate(cols_M))),
                   shape=(nnode, nnode)).tocsr()

    # 底边 Robin：对 int_0^L r N_i N_j dr 做精确积分（r 沿线段线性）
    rows_H, cols_H, vals_H = [], [], []
    f_conv = np.zeros(nnode)
    for edge in mesh.bottom_edges:
        p = nodes[edge]
        r0 = float(p[0, 0])
        L = float(p[1, 0] - p[0, 0])
        He = 2.0 * np.pi * case.h_conv * L * np.array([
            [r0 / 3.0 + L / 12.0, r0 / 6.0 + L / 12.0],
            [r0 / 6.0 + L / 12.0, r0 / 3.0 + L / 4.0],
        ])
        _coo_accum(rows_H, cols_H, vals_H, edge, He)
        f_conv[edge] += 2.0 * np.pi * case.h_conv * case.t_amb * L * np.array(
            [r0 / 2.0 + L / 6.0, r0 / 2.0 + L / 3.0])
    H = coo_matrix((np.concatenate(vals_H),
                    (np.concatenate(rows_H), np.concatenate(cols_H))),
                   shape=(nnode, nnode)).tocsr()

    return AxisymSystem((K + H).tocsr(), M.tocsr(), f_conv, f_heat,
                        mesh, case, 2.0 * np.pi * weighted_area)


def solve_axisym_steady(sys: AxisymSystem, power: float | None = None) -> np.ndarray:
    P = sys.case.total_power if power is None else power
    return splu(sys.K.tocsc()).solve(sys.load(P))


def axisym_junction(sys: AxisymSystem, field: np.ndarray) -> float:
    return float(field[sys.mesh.junction_node])


def run_axisym_transient(sys: AxisymSystem, *, dt: float | None = None,
                         theta: float | None = None, t_end: float | None = None,
                         power_fn=None) -> TransientResult:
    c = sys.case
    dt = c.dt if dt is None else dt
    theta = c.theta if theta is None else theta
    t_end = c.t_end if t_end is None else t_end
    A = (sys.M / dt + theta * sys.K).tocsc()
    B = (sys.M / dt - (1.0 - theta) * sys.K).tocsr()
    lu = splu(A)
    n_steps = int(round(t_end / dt))
    times = np.arange(n_steps + 1) * dt
    P_on = c.total_power

    def pw(t):
        if power_fn is not None:
            return power_fn(t) * P_on
        ph = (t % c.pulse_period) / c.pulse_period
        return P_on if ph < c.duty else 0.0

    T = np.full(sys.nnode, c.t_amb)
    f_prev = sys.load(pw(0.0))
    tj = np.empty(n_steps + 1)
    tmax = np.empty(n_steps + 1)
    tb = np.empty(n_steps + 1)
    pwr = np.empty(n_steps + 1)
    tj[0], tmax[0], tb[0] = T[sys.mesh.junction_node], T.max(), T[0]
    pwr[0] = pw(0.0)
    for n in range(1, n_steps + 1):
        f_now = sys.load(pw(times[n]))
        T = lu.solve(B @ T + theta * f_now + (1.0 - theta) * f_prev)
        f_prev = f_now
        tj[n], tmax[n], tb[n] = T[sys.mesh.junction_node], T.max(), T[0]
        pwr[n] = pw(times[n])
    return TransientResult(times, tj, tmax, tb, pwr,
                           method=f"AxisymFEM-theta{theta:g}")


def spreading_resistance_abs(t_junction: float, case: AxisymCase) -> float:
    """R_sp = (T_j,3D - T_j,1D) / P，单位 K/W。"""
    return (t_junction - case.t_junction_1d) / case.total_power


def semi_infinite_spreading(k: float, a: float) -> float:
    """半无限体表面均匀热流圆形源的经典扩散热阻 8/(3*pi^2*k*a) (K/W)。"""
    return 8.0 / (3.0 * np.pi ** 2 * k * a)
