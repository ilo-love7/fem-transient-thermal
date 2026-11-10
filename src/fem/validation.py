"""验证工具：一维稳态解析解、scikit-fem 独立装配交叉验证、误差与收敛阶指标。"""
from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import spsolve

from .materials import ThermalCase
from .mesh import StructuredMesh


def analytic_1d_steady(case: ThermalCase, y: np.ndarray) -> np.ndarray:
    """严格一维（热源铺满）稳态温度 T(y) 解析解。

    - 底面：T(0) = T_amb + q''/h，q'' = P'/width；
    - 无热源层：线性，dT/dy = q''/k_i（全部热流穿过）；
    - 芯片层（均匀体积热源 q'''=q''/L_die，顶面绝热）：抛物线
      T(y) = base + q''*(y-y0)/k - q'''*(y-y0)^2/(2k)。
      顶面温升只有 q''*L/(2k)，这正是 FEM/Cauer 与「朴素串联热阻」
      公式相差 q''*L/(2k) 的根源。
    """
    y = np.asarray(y, dtype=float)
    qflux = case.power_per_length / case.width
    out = np.full_like(y, case.t_amb + qflux / case.h_conv)
    interfaces = case.y_interfaces
    die_idx = case.n_layers - 1
    for j, L in enumerate(case.layers):
        y0 = interfaces[j]
        base = case.t_amb + qflux / case.h_conv + sum(
            qflux * Lp.thickness / Lp.k for Lp in case.layers[:j]
        )
        mask = (y >= y0 - 1e-15) & (y <= interfaces[j + 1] + 1e-15)
        dy = y[mask] - y0
        out[mask] = base + qflux * dy / L.k
        if j == die_idx:
            q3 = qflux / L.thickness
            out[mask] -= q3 * dy ** 2 / (2.0 * L.k)
    return out


def analytic_1d_field(case: ThermalCase, mesh: StructuredMesh) -> np.ndarray:
    return analytic_1d_steady(case, mesh.nodes[:, 1])


def skfem_steady(case: ThermalCase, mesh: StructuredMesh,
                 power: float | None = None) -> tuple[np.ndarray, dict]:
    """用 scikit-fem 12 独立装配同一弱形式并求解，返回手写网格节点处的温度。

    与手写装配互相独立（形函数/积分/边界处理全部走 skfem 内核），
    用 probes 按坐标取值，不依赖节点编号是否一致。
    """
    import skfem as fem
    from skfem.helpers import dot, grad

    power = case.power_per_length if power is None else power
    m = fem.MeshTri(mesh.nodes.T.copy(), mesh.elems.T.copy())
    B = fem.Basis(m, fem.ElementTriP1())
    B0 = B.with_element(fem.ElementTriP0())

    k_elem = np.array([case.layers[j].k for j in mesh.elem_layer])
    q_unit = 1.0 / (case.heat_source_width * case.die.thickness)
    q_elem = np.where(mesh.elem_heated, q_unit * power, 0.0)

    @fem.BilinearForm
    def conduction(u, v, w):
        return w["k"] * dot(grad(u), grad(v))

    @fem.LinearForm
    def source(v, w):
        return w["q"] * v

    A = conduction.assemble(B, k=B0.interpolate(k_elem))
    f = source.assemble(B, q=B0.interpolate(q_elem))

    bottom = m.facets_satisfying(lambda x: np.isclose(x[1], 0.0))
    Fb = fem.FacetBasis(m, fem.ElementTriP1(), facets=bottom)

    @fem.BilinearForm
    def robin(u, v, w):
        return w["h"] * u * v

    @fem.LinearForm
    def conv_load(v, w):
        return w["ht"] * v

    H = robin.assemble(Fb, h=case.h_conv)
    g = conv_load.assemble(Fb, ht=case.h_conv * case.t_amb)
    T_sf = spsolve((A + H).tocsr(), f + g)

    probes = B.probes(np.vstack([mesh.nodes[:, 0], mesh.nodes[:, 1]]))
    T_at_nodes = np.asarray(probes @ T_sf).ravel()
    return T_at_nodes, {
        "n_skem_nodes": int(B.N),
        "n_bottom_facets": int(len(bottom)),
        "t_junction": float(T_at_nodes[mesh.junction_node]),
    }


def error_metrics(a: np.ndarray, b: np.ndarray, scale: float | None = None) -> dict:
    """相对/绝对误差指标（b 为参考解）。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    denom = float(np.max(np.abs(b - np.mean(b)))) if scale is None else float(scale)
    if denom <= 0:
        denom = 1.0
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "l2_rel": float(np.linalg.norm(diff) / np.sqrt(len(b)) / denom),
        "peak_abs": float(abs(a.max() - b.max())),
        "mean_abs": float(np.mean(np.abs(diff))),
    }


def convergence_order(h: np.ndarray, err: np.ndarray) -> tuple[float, float]:
    """log-log 最小二乘斜率 p 与截距（err ≈ C h^p）。"""
    h = np.asarray(h, dtype=float)
    err = np.asarray(err, dtype=float)
    mask = (h > 0) & (err > 0)
    p, logc = np.polyfit(np.log(h[mask]), np.log(err[mask]), 1)
    return float(p), float(np.exp(logc))


def bi_numbers(case: ThermalCase) -> dict[str, float]:
    """各层 Bi = h*L/k（外部对流尺度）；集总模型适用性判据。"""
    return {L.name: case.h_conv * L.thickness / L.k for L in case.layers}


def internal_bi(case: ThermalCase, idx: int) -> float:
    """层内 Bi（相邻层热阻决定的等效换热尺度）。"""
    L = case.layers[idx]
    r_layer = L.thickness / L.k
    r_other = sum(Lp.thickness / Lp.k for j, Lp in enumerate(case.layers) if j != idx)
    return float(r_layer / (r_other + 1.0 / case.h_conv))
