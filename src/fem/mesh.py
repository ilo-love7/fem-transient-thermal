"""结构化多层矩形网格：线性三角形单元、材料分区、对流边界线段。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .materials import ThermalCase


@dataclass
class StructuredMesh:
    width: float
    height: float
    nx: int
    ny_per_layer: int
    n_layers: int
    nodes: np.ndarray  # (nnode, 2) x,y
    elems: np.ndarray  # (nelem, 3) 节点编号
    elem_layer: np.ndarray  # (nelem,) 材料层索引（从底 0 起）
    elem_heated: np.ndarray  # (nelem,) bool，是否属于热源区
    bottom_edges: np.ndarray  # (nx, 2) 底面对流线段节点编号
    junction_node: int  # 顶面中心节点（热点测点）
    layer_of_row: np.ndarray  # (ny_total,) 每行所属层

    @property
    def nnode(self) -> int:
        return self.nodes.shape[0]

    @property
    def nelem(self) -> int:
        return self.elems.shape[0]


def build_mesh(case: ThermalCase, nx: int | None = None,
               ny_per_layer: int | None = None) -> StructuredMesh:
    """按层界面剖分：每层 ny_per_layer 行，x 方向 nx 段，每矩形分两个三角形。"""
    nx = case.nx if nx is None else nx
    npl = case.ny_per_layer if ny_per_layer is None else ny_per_layer
    n_layers = case.n_layers
    ys = case.y_interfaces

    # y 节点（层界面必须落在网格线上）
    y_nodes = [0.0]
    layer_of_row = []
    for j, L in enumerate(case.layers):
        for r in range(npl):
            y_nodes.append(ys[j] + (r + 1) * L.thickness / npl)
            layer_of_row.append(j)
    y_nodes = np.array(y_nodes)
    x_nodes = np.linspace(0.0, case.width, nx + 1)
    ny_total = npl * n_layers

    # 节点按行优先：idx = row*(nx+1) + col
    xx, yy = np.meshgrid(x_nodes, y_nodes)
    nodes = np.column_stack([xx.ravel(), yy.ravel()])

    # 单元：两三角形 / 矩形
    elems = []
    elem_layer = []
    elem_heated = []
    x0, x1 = case.heat_source_bounds
    die_layer = n_layers - 1
    # 热源区判定用单元重心 x（要求热源宽度与网格对齐，见 scripts 校验）
    for row in range(ny_total):
        j = layer_of_row[row]
        for col in range(nx):
            v00 = row * (nx + 1) + col
            v10 = v00 + 1
            v01 = v00 + (nx + 1)
            v11 = v01 + 1
            # 交替对角线（checkerboard）：消除单向剖分在一维场上引入的
            # x 方向锯齿伪影（否则严格 1D 解会出现 O(h) 的各向异性误差）。
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
        width=case.width,
        height=case.height,
        nx=nx,
        ny_per_layer=npl,
        n_layers=n_layers,
        nodes=nodes,
        elems=elems,
        elem_layer=elem_layer,
        elem_heated=elem_heated,
        bottom_edges=bottom_edges,
        junction_node=junction_node,
        layer_of_row=np.asarray(layer_of_row, dtype=np.int64),
    )


def node_at(mesh: StructuredMesh, col: int, row: int) -> int:
    return row * (mesh.nx + 1) + col
