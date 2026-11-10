"""00 环境校验：依赖、配置、网格恒等式、装配自检。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from fem import (load_config, case_from_config, default_config_path,
                 default_case, build_mesh, assemble, solve_steady,
                 analytic_1d_field, error_metrics, skfem_steady)


def main():
    cfg = load_config(default_config_path())
    case = case_from_config(cfg)
    print("=== 依赖与配置 ===")
    import scipy
    import skfem
    import matplotlib
    print(f"numpy {np.__version__} / scipy {scipy.__version__} / "
          f"skfem {skfem.__version__} / matplotlib {matplotlib.__version__}")
    print(f"工况: W={case.width*1e3:.1f} mm, H={case.height*1e3:.2f} mm, "
          f"P'={case.power_per_length} W/m, h={case.h_conv} W/(m2 K), "
          f"T_amb={case.t_amb} C")
    print(f"层（底→顶）: {[(L.name, L.thickness*1e3, L.k) for L in case.layers]}")
    print(f"R_cond={case.r_cond_series:.6f}, R_conv={case.r_conv:.6f}, "
          f"R_eff={case.r_effective:.6f} K*m/W")
    print(f"T_j(解析/体积热源)={case.t_junction_steady:.4f} C, "
          f"T_j(朴素串联/表面热源)={case.t_junction_surface_source:.4f} C, "
          f"口径差={case.t_junction_surface_source-case.t_junction_steady:.4f} K")

    m = build_mesh(case)
    s = assemble(case, m)
    print(f"\n=== 网格与装配 ===")
    print(f"节点 {m.nnode}, 单元 {m.nelem}, 底边线段 {len(m.bottom_edges)}")
    print(f"M 总和 {s.M.sum():.6e} J/(m K)  "
          f"(理论 {sum(L.rc*L.thickness*case.width for L in case.layers):.6e})")
    assert abs(s.M.sum() - sum(L.rc*L.thickness*case.width for L in case.layers)) < 1e-6
    assert abs(s.f_heat_unit.sum() - 1.0) < 1e-12
    print("单位热源载荷总和 = 1 W/m  [OK]")

    T = solve_steady(s, case.power_per_length)
    Ta = analytic_1d_field(case, m)
    e1 = error_metrics(T, Ta)
    Tsf, info = skfem_steady(case, m)
    e2 = error_metrics(T, Tsf)
    print(f"\n=== 稳态交叉校验 ===")
    print(f"FEM vs 解析: max_abs={e1['max_abs']:.3e} K")
    print(f"FEM vs skfem: max_abs={e2['max_abs']:.3e} K ({info['n_skem_nodes']} 节点)")
    assert e1["max_abs"] < 0.02 and e2["max_abs"] < 1e-8
    print("\n环境校验全部通过 [OK]")


if __name__ == "__main__":
    main()
