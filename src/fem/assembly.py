"""手写 Galerkin FEM 装配：瞬态热传导弱形式。

    M dT/dt + (K_cond + H_conv) T = f_conv + P(t) * f_heat_unit

单元矩阵（线性三角形、常物性，全部解析积分）：
    K_e = k * Ae * (grad N_i . grad N_j)
    M_e = rho*c * Ae/12 * [[2,1,1],[1,2,1],[1,1,2]]     （一致质量）
    H_e = h * L/6 * [[2,1],[1,2]]                        （Robin 边界）
    f_conv_e = h*T_amb * L/2 * [1,1]
    f_heat_e = q''' * Ae/3 * [1,1,1]
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from .materials import ThermalCase
from .mesh import StructuredMesh


@dataclass
class FEMSystem:
    K: object  # csr，含对流边界刚度
    M: object  # csr，一致质量
    f_conv: np.ndarray  # 常数载荷（Robin T_amb 项）
    f_heat_unit: np.ndarray  # 单位线功率 (P'=1 W/m) 对应的载荷
    mesh: StructuredMesh

    @property
    def nnode(self) -> int:
        return self.mesh.nnode

    def load(self, power: float) -> np.ndarray:
        return self.f_conv + power * self.f_heat_unit


def _tri_grad(p: np.ndarray):
    """返回三角形面积 Ae 与形函数梯度 (3,2)。"""
    b = np.array([p[1, 1] - p[2, 1], p[2, 1] - p[0, 1], p[0, 1] - p[1, 1]])
    c = np.array([p[2, 0] - p[1, 0], p[0, 0] - p[2, 0], p[1, 0] - p[0, 0]])
    det = b[1] * c[2] - b[2] * c[1]  # = 2*Ae（有符号）
    if det <= 0:
        raise ValueError("三角形节点顺序非逆时针，梯度符号会反")
    grad = np.column_stack([b / det, c / det])
    return 0.5 * abs(det), grad


def _coo_accum(rows: list, cols: list, vals: list, idx: np.ndarray, mat: np.ndarray):
    rr = np.repeat(idx, len(idx))
    cc = np.tile(idx, len(idx))
    rows.append(rr)
    cols.append(cc)
    vals.append(mat.ravel())


def assemble(case: ThermalCase, mesh: StructuredMesh | None = None) -> FEMSystem:
    if mesh is None:
        mesh = build_mesh_safe(case)
    nodes, elems = mesh.nodes, mesh.elems
    nnode = mesh.nnode

    k_of = [L.k for L in case.layers]
    rc_of = [L.rc for L in case.layers]
    # 单位线功率的体积热源密度：P'=1 W/m 分摊到热源体积 (w_h * L_die * 1)
    w_h = case.heat_source_width
    q_unit = 1.0 / (w_h * case.die.thickness)

    rows_K: list = []
    cols_K: list = []
    vals_K: list = []
    rows_M: list = []
    cols_M: list = []
    vals_M: list = []
    f_heat = np.zeros(nnode)

    M_ref = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]) / 12.0

    for e, tri in enumerate(elems):
        p = nodes[tri]
        Ae, grad = _tri_grad(p)
        lay = mesh.elem_layer[e]
        _coo_accum(rows_K, cols_K, vals_K, tri,
                   k_of[lay] * Ae * (grad @ grad.T))
        _coo_accum(rows_M, cols_M, vals_M, tri, rc_of[lay] * Ae * M_ref)
        if mesh.elem_heated[e]:
            f_heat[tri] += q_unit * Ae / 3.0

    K = coo_matrix((np.concatenate(vals_K),
                    (np.concatenate(rows_K), np.concatenate(cols_K))),
                   shape=(nnode, nnode)).tocsr()
    M = coo_matrix((np.concatenate(vals_M),
                    (np.concatenate(rows_M), np.concatenate(cols_M))),
                   shape=(nnode, nnode)).tocsr()

    # Robin 对流边界（底边线段）
    rows_H: list = []
    cols_H: list = []
    vals_H: list = []
    f_conv = np.zeros(nnode)
    for edge in mesh.bottom_edges:
        p = nodes[edge]
        L = float(np.linalg.norm(p[1] - p[0]))
        He = case.h_conv * L / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
        _coo_accum(rows_H, cols_H, vals_H, edge, He)
        f_conv[edge] += case.h_conv * case.t_amb * L / 2.0
    H = coo_matrix((np.concatenate(vals_H),
                    (np.concatenate(rows_H), np.concatenate(cols_H))),
                   shape=(nnode, nnode)).tocsr()

    return FEMSystem(K=(K + H).tocsr(), M=M.tocsr(),
                     f_conv=f_conv, f_heat_unit=f_heat, mesh=mesh)


def build_mesh_safe(case: ThermalCase) -> StructuredMesh:
    from .mesh import build_mesh

    return build_mesh(case)


def solve_steady(sys: FEMSystem, power: float) -> np.ndarray:
    return splu(sys.K.tocsc()).solve(sys.load(power))
