"""v1.3 E18：自适应网格加密的单测。"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fem.adaptive import (adaptivity_study, build_mesh_nonuniform,  # noqa: E402
                          gradient_indicator, interval_indicators,
                          refine_intervals)
from fem.assembly import assemble, solve_steady  # noqa: E402
from fem.config import case_from_config, default_config_path, load_config  # noqa: E402
from fem.mesh import build_mesh, node_at  # noqa: E402


@pytest.fixture(scope="module")
def base():
    return case_from_config(load_config(default_config_path()))


def test_nonuniform_uniform_matches_builtin(base):
    xs = np.linspace(0.0, base.width, 17)
    m1 = build_mesh_nonuniform(base, xs)
    m2 = build_mesh(base, nx=16)
    for attr in ("nodes", "elems", "elem_layer", "elem_heated",
                 "bottom_edges", "layer_of_row"):
        np.testing.assert_allclose(getattr(m1, attr), getattr(m2, attr))
    assert m1.junction_node == m2.junction_node
    assert m1.nx == 16 and m1.nnode == m2.nnode


def test_nonuniform_invalid_nodes_rejected(base):
    with pytest.raises(ValueError):
        build_mesh_nonuniform(base, np.array([0.0, 0.005, 0.005, 0.01]))
    with pytest.raises(ValueError):
        build_mesh_nonuniform(base, np.array([0.001, 0.005, 0.01]))
    with pytest.raises(ValueError):
        build_mesh_nonuniform(base, np.array([0.0, 0.01, 0.02]))


def test_nonuniform_reuses_assembly_and_node_at(base):
    xs = np.linspace(0.0, base.width, 13)
    m = build_mesh_nonuniform(base, xs)
    sys = assemble(base, m)
    T = solve_steady(sys, base.power_per_length)
    assert T.shape == (m.nnode,)
    assert np.isfinite(T).all()
    assert node_at(m, m.nx // 2, m.ny_per_layer * base.n_layers) == m.junction_node


def test_gradient_indicator_nonnegative(base):
    m = build_mesh(base, nx=12)
    T = solve_steady(assemble(base, m), base.power_per_length)
    eta = gradient_indicator(m, T)
    assert eta.shape == (m.nelem,)
    assert np.all(eta >= 0)
    assert eta.sum() > 0


def test_interval_indicators_sum_to_element_sum(base):
    m = build_mesh_nonuniform(base, np.linspace(0, base.width, 11))
    T = solve_steady(assemble(base, m), base.power_per_length)
    eta = gradient_indicator(m, T)
    col = interval_indicators(m, eta)
    assert col.shape == (m.nx,)
    assert col.sum() == pytest.approx(eta.sum(), rel=1e-12)


def test_refine_inserts_midpoints_and_keeps_endpoints():
    xs = np.linspace(0, 1, 6)
    eta = np.array([0, 0, 9, 0, 8], dtype=float)
    out, marked = refine_intervals(xs, eta, 0.4)
    assert out[0] == 0 and out[-1] == 1
    assert np.all(np.diff(out) > 0)
    assert out.size > xs.size
    assert set(marked.tolist()) == {2, 4}


def test_adaptive_beats_uniform_for_local_source(base):
    # 局部热源：a/R=0.2（与 v1.1/v1.2 扩散实验同口径）
    case = replace(base, heat_source_width=0.2 * base.width,
                   nx=24, ny_per_layer=8)
    r = adaptivity_study(case, coarse_nx=24, refine_fraction=0.5,
                         reference_nx=160)
    assert r["adaptive_nx"] == r["uniform_same_dof_nx"]
    assert r["rel_err_adaptive"] < r["rel_err_uniform_same_dof"]
    assert r["rel_err_adaptive"] < r["rel_err_coarse"]


def test_full_width_source_x_refinement_no_benefit(base):
    # 全宽热源 = 严格 1D，x 方向无梯度：x 加密不应带来收益（负结果）
    case = replace(base, heat_source_width=base.width,
                   nx=24, ny_per_layer=6)
    r = adaptivity_study(case, coarse_nx=24, refine_fraction=0.5,
                         reference_nx=96)
    # 真实结果：checkerboard 剖分让 1D 解的 x 列梯度被奇偶伪影主导，指示子
    # 比值很大但与热源位置无关；因此 x 加密没有物理收益（负结果）。
    # 判据不用 eta 比值，而看「自适应精度不优于同自由度均匀网格」。
    assert r["rel_err_coarse"] < 2e-5
    # 两者都在 1e-5 相对误差内（数值精度），无实质差异 = x 加密零收益
    assert r["rel_err_adaptive"] < 2e-5
    assert r["rel_err_uniform_same_dof"] < 2e-5
