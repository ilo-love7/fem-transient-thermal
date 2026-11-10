"""04 二维局部热源在脉冲功率下的温度场演化动画。"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from fem import (load_config, case_from_config, default_config_path,
                 build_mesh, assemble, run_transient, solve_steady,
                 periodic_steady_metrics)
from fem.plots import animate_field, plot_transient, plot_field


def main():
    t0 = time.time()
    cfg = load_config(default_config_path())
    base = case_from_config(cfg)
    case = base.with_params(heat_source_width=0.002, dt=0.05, t_end=4.0)
    mesh = build_mesh(case)
    system = assemble(case, mesh)
    print(f"[动画] {mesh.nnode} 节点, 热源 w={case.heat_source_width*1e3:.1f} mm, "
          f"dt={case.dt}s, t_end={case.t_end}s", flush=True)

    snaps = tuple(round(i * case.dt, 8) for i in range(int(case.t_end / case.dt) + 1))
    r = run_transient(system, case, snapshot_times=snaps)
    print(f"[动画] 捕获 {len(r.snapshots)} 帧, 结温 {r.t_junction.min():.2f}–"
          f"{r.t_junction.max():.2f} C", flush=True)
    assert len(r.snapshots) >= len(snaps) - 1, "快照缺失过多"

    animate_field(mesh, case, r.snapshots,
                  ROOT / "figures" / "thermal_pulse_2d.gif", fps=10)
    plot_transient({"FEM-BE · 二维局部热源": r},
                   "二维局部热源脉冲结温曲线（w/W=0.2）",
                   ROOT / "figures" / "transient_2d.png")
    Tss = solve_steady(system, case.power_per_length)
    plot_field(mesh, Tss, case, "二维局部热源稳态温度场（w/W=0.2）",
               ROOT / "figures" / "steady_2d_field_anim.png")
    (ROOT / "results" / "series_2d_pulse.json").write_text(
        json.dumps(periodic_steady_metrics(r, case), ensure_ascii=False, indent=2),
        encoding="utf-8")
    np.savez(ROOT / "results" / "series_2d_pulse.npz",
             t=r.t, t_junction=r.t_junction, power=r.power)
    print(f"[04 完成] 用时 {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
