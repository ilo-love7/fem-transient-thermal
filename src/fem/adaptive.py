"""v1.3 深化 E18：后验误差指示子驱动的 h-自适应网格加密。

v1.0–v1.2 一律用均匀 x 网格；局部热源（a/R<<1）时热流在热源边缘高度集中，
均匀网格把自由度浪费在远离热源的绝热区。本模块实现最小可用的自适应循环：

  1) 在粗均匀网格上解稳态温度场 T；
  2) 逐单元能量范数局部贡献  eta_e = Ae * |grad T_e|^2   （P1 单元内梯度精确）；
  3) 把单元 eta 聚合到 x 区间（列），按 eta 降序选 top fraction 的区间；
  4) 在被标记区间插入中点，得到非均匀网格；复用既有 assemble/solve 路径重解。

关键可证伪命题：
  * 对局部热源，自适应网格在**同等自由度**下应比均匀网格结温更准；
  * 对全宽热源（严格 1D，x 方向无梯度），x 方向细化不应带来收益
    （负结果，必须如实写入报告）——这是自洽性边界。

非均匀网格仍返回标准 StructuredMesh，故 bottom_edges / layer_of_row /
junction_node / node_at 全部保持原有语义，装配与瞬态求解零改动。
"""
from __future__ import annotations

import numpy as np

from .assembly import _tri_grad, assemble, solve_steady
from .materials import ThermalCase
from .mesh import StructuredMesh


def build_mesh_nonuniform(case: ThermalCase, x_nodes,
                          ny_per_layer: int | None = None) -> StructuredMesh:
    """与 mesh.build_mesh 逐位同构，但 x 节点由 x_nodes 显式给定。

    要求 x_nodes 严格递增、首尾为 0 与 case.width；nx = len(x_nodes)-1。
    当 x_nodes = linspace(0, width, nx+1) 时输出与 build_mesh 完全一致。
    """
    xs = np.asarray(x_nodes, dtype=float)
    if xs.ndim != 1 or xs.size < 2:
        raise ValueError("x_nodes 须为一维且至少 2 点")
    if not np.all(np.diff(xs) > 0):
        raise ValueError("x_nodes 必须严格递增")
    if not (np.isclose(xs[0], 0.0) and np.isclose(xs[-1], case.width)):
        raise ValueError("x_nodes 首尾须为 0 与 case.width")
    nx = int(xs.size - 1)
    npl = case.ny_per_layer if ny_per_layer is None else int(ny_per_layer)
    if npl < 1:
        raise ValueError("ny_per_layer 必须 >=1")
    n_layers = case.n_layers
    ys = case.y_interfaces

    y_nodes = [0.0]
    layer_of_row: list[int] = []
    for j, L in enumerate(case.layers):
        for r in range(npl):
            y_nodes.append(ys[j] + (r + 1) * L.thickness / npl)
            layer_of_row.append(j)
    y_nodes = np.asarray(y_nodes)
    ny_total = npl * n_layers

    xx, yy = np.meshgrid(xs, y_nodes)
    nodes = np.column_stack([xx.ravel(), yy.ravel()])

    elems: list[tuple[int, int, int]] = []
    elem_layer: list[int] = []
    elem_heated: list[bool] = []
    x0, x1 = case.heat_source_bounds
    die_layer = n_layers - 1
    for row in range(ny_total):
        j = layer_of_row[row]
        for col in range(nx):
            v00 = row * (nx + 1) + col
            v10 = v00 + 1
            v01 = v00 + (nx + 1)
            v11 = v01 + 1
            if (row + col) % 2 == 0:
                tris = ((v00, v10, v11), (v00, v11, v01))
            else:
                tris = ((v00, v10, v01), (v10, v11, v01))
            for tri in tris:
                elems.append(tri)
                elem_layer.append(j)
                cx = float(nodes[list(tri), 0].mean())
                elem_heated.append(j == die_layer and x0 <= cx <= x1)

    elems = np.asarray(elems, dtype=np.int64)
    elem_layer = np.asarray(elem_layer, dtype=np.int64)
    elem_heated = np.asarray(elem_heated, dtype=bool)
    bottom_edges = np.array([[i, i + 1] for i in range(nx)], dtype=np.int64)
    center_col = int(round(nx / 2))
    junction_node = ny_total * (nx + 1) + center_col

    return StructuredMesh(
        width=case.width, height=case.height, nx=nx, ny_per_layer=npl,
        n_layers=n_layers, nodes=nodes, elems=elems, elem_layer=elem_layer,
        elem_heated=elem_heated, bottom_edges=bottom_edges,
        junction_node=junction_node,
        layer_of_row=np.asarray(layer_of_row, dtype=np.int64),
    )


def gradient_indicator(mesh: StructuredMesh, T: np.ndarray) -> np.ndarray:
    """逐单元能量范数局部贡献 eta_e = Ae * |grad T_e|^2。"""
    T = np.asarray(T, dtype=float)
    if T.size != mesh.nnode:
        raise ValueError("T 长度须等于网格节点数")
    eta = np.empty(mesh.nelem, dtype=float)
    for e, tri in enumerate(mesh.elems):
        Ae, grad = _tri_grad(mesh.nodes[tri])
        gT = grad.T @ T[tri]  # (2,) 单元内温度梯度
        eta[e] = Ae * float(gT @ gT)
    return eta


def interval_indicators(mesh: StructuredMesh, eta: np.ndarray) -> np.ndarray:
    """把单元级 eta 聚合到 x 列级；每列含 2*ny_total 个三角形。"""
    eta = np.asarray(eta, dtype=float)
    per_col = 2 * mesh.ny_per_layer * mesh.n_layers
    if eta.size != mesh.nx * per_col:
        raise ValueError("eta 长度与网格列数不匹配")
    return eta.reshape(mesh.nx, per_col).sum(axis=1)


def refine_intervals(x_nodes, interval_eta, fraction: float):
    """在 eta 最大的 fraction 比例区间插入中点，返回新节点与标记列。"""
    xs = np.asarray(x_nodes, dtype=float)
    eta = np.asarray(interval_eta, dtype=float)
    nx = int(xs.size - 1)
    if eta.size != nx:
        raise ValueError("eta 长度须等于区间数")
    if not (0 < fraction <= 1):
        raise ValueError("fraction 须在 (0,1]")
    k = min(nx, max(1, int(np.ceil(nx * fraction))))
    marked = np.sort(np.argsort(-eta)[:k])
    mids = 0.5 * (xs[marked] + xs[marked + 1])
    out = np.sort(np.concatenate([xs, mids]))
    return out, marked


def _junction_steady(case: ThermalCase, x_nodes):
    mesh = build_mesh_nonuniform(case, x_nodes)
    sys = assemble(case, mesh)
    T = solve_steady(sys, case.power_per_length)
    return float(T[mesh.junction_node]), mesh, T


def adaptivity_study(case: ThermalCase, *, coarse_nx: int = 24,
                     refine_fraction: float = 0.35,
                     reference_nx: int = 320) -> dict:
    """粗解 -> 指示子 -> 自适应 -> 同自由度均匀，四方对照。

    误差以参考解温升 (Tj_ref - t_amb) 归一，避免环境温度平移稀释相对量。
    """
    if coarse_nx < 4 or reference_nx <= coarse_nx:
        raise ValueError("要求 reference_nx > coarse_nx >= 4")
    x_coarse = np.linspace(0.0, case.width, coarse_nx + 1)
    x_ref = np.linspace(0.0, case.width, reference_nx + 1)

    tj_ref, mesh_ref, _ = _junction_steady(case, x_ref)
    tj_coarse, mesh_c, T_c = _junction_steady(case, x_coarse)

    eta_elem = gradient_indicator(mesh_c, T_c)
    eta_col = interval_indicators(mesh_c, eta_elem)
    x_ad, marked = refine_intervals(x_coarse, eta_col, refine_fraction)
    tj_ad, mesh_ad, T_ad = _junction_steady(case, x_ad)

    nx_ad = mesh_ad.nx
    x_uni = np.linspace(0.0, case.width, nx_ad + 1)
    tj_uni, mesh_uni, _ = _junction_steady(case, x_uni)

    scale = max(abs(tj_ref - case.t_amb), 1e-12)
    err = lambda tj: abs(tj - tj_ref) / scale

    # 热源边缘是否真的被标记（重心落在 [x0-r, x1+r] 的区间）
    x0, x1 = case.heat_source_bounds
    band = 0.5 * (case.width / coarse_nx)
    near_src = set(int(np.floor((b - case.width * 1e-12) / (case.width / coarse_nx)))
                   for b in (x0, x1))
    hit = any(any(abs(j - b) <= 1 for j in near_src) for b in marked)

    return {
        "coarse_nx": coarse_nx,
        "adaptive_nx": nx_ad,
        "uniform_same_dof_nx": nx_ad,
        "reference_nx": reference_nx,
        "marked_columns": marked.tolist(),
        "refined_near_heat_source": bool(hit),
        "t_junction_ref_C": tj_ref,
        "t_junction_coarse_C": tj_coarse,
        "t_junction_adaptive_C": tj_ad,
        "t_junction_uniform_C": tj_uni,
        "rel_err_coarse": err(tj_coarse),
        "rel_err_adaptive": err(tj_ad),
        "rel_err_uniform_same_dof": err(tj_uni),
        "speedup_over_uniform": err(tj_uni) / max(err(tj_ad), 1e-30),
        "element_eta_sum": float(eta_elem.sum()),
        "interval_eta_max_over_min": float(eta_col.max() / max(eta_col.min(), 1e-30)),
        "nnode_coarse": mesh_c.nnode,
        "nnode_adaptive": mesh_ad.nnode,
        "nnode_reference": mesh_ref.nnode,
    }
