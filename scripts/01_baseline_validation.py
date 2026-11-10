"""01 基线验证：稳态解析/第三方 FEM、瞬态 Cauer 交叉验证、时间与空间收敛阶。"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from fem import (load_config, case_from_config, default_config_path,
                 build_mesh, assemble, solve_steady, analytic_1d_field,
                 skfem_steady, error_metrics, convergence_order,
                 fine_network, lumped_network, run_transient,
                 curve_error_full, periodic_steady_metrics,
                 thermal_metrics, bi_numbers)
from fem.plots import plot_field, plot_transient, plot_convergence


def dump(path, obj):
    (ROOT / "results" / path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=float), encoding="utf-8")


def main():
    t_start = time.time()
    res_dir, fig_dir = ROOT / "results", ROOT / "figures"
    cfg = load_config(default_config_path())
    case = case_from_config(cfg)
    mesh = build_mesh(case)
    system = assemble(case, mesh)
    print(f"[网格] {mesh.nnode} 节点 / {mesh.nelem} 单元 / nx={case.nx} npl={case.ny_per_layer}", flush=True)

    # ---------- A 稳态三重验证 ----------
    T = solve_steady(system, case.power_per_length)
    Ta = analytic_1d_field(case, mesh)
    Tsf, skinfo = skfem_steady(case, mesh)
    e_analytic = error_metrics(T, Ta)
    e_skfem = error_metrics(T, Tsf)
    cauer_steady = fine_network(case, 64).steady(case.power_per_length)[0]
    q_out = case.h_conv * case.width * (T[mesh.bottom_edges].mean() - case.t_amb)
    gap = case.t_junction_surface_source - case.t_junction_steady
    steady = {
        "t_junction_fem": float(T[mesh.junction_node]),
        "t_junction_analytic": float(case.t_junction_steady),
        "t_junction_skfem": float(Tsf[mesh.junction_node]),
        "t_junction_cauer": float(cauer_steady),
        "t_junction_surface_source": float(case.t_junction_surface_source),
        "surface_vs_volume_gap_K": float(gap),
        "err_fem_vs_analytic": e_analytic,
        "err_fem_vs_skfem": e_skfem,
        "energy_balance_rel_err": float(abs(q_out - case.power_per_length) / case.power_per_length),
        "r_effective": case.r_effective, "r_total_surface": case.r_total,
        "bi": bi_numbers(case),
    }
    dump("validation_steady.json", steady)
    plot_field(mesh, T, case, "稳态温度场 · 一维验证口径（热源铺满）",
               fig_dir / "steady_1d_field.png")
    print(f"[稳态] FEM {steady['t_junction_fem']:.4f} / 解析 {case.t_junction_steady:.4f} / "
          f"skfem {steady['t_junction_skfem']:.4f} / Cauer {cauer_steady:.4f} C", flush=True)
    print(f"       FEM-解析 {e_analytic['max_abs']:.3e} K, FEM-skfem {e_skfem['max_abs']:.3e} K, "
          f"口径差 {gap:.4f} K", flush=True)

    # ---------- B 瞬态：FEM(BE/CN) vs 精细 Cauer ----------
    dt, t_end = 0.005, 10.0
    curves, metrics = {}, {}
    r_be = run_transient(system, case, dt=dt, t_end=t_end, theta=1.0,
                         snapshot_times=(0.5, 1.0, 1.5, 2.0, 8.5, 9.0, 9.5, 10.0))
    r_cn = run_transient(system, case, dt=dt, t_end=t_end, theta=0.5)
    net_fine = fine_network(case, 64)
    r_cauer = net_fine.run(dt=dt, t_end=t_end)
    r_lumped = lumped_network(case).run(dt=dt, t_end=t_end)
    r_n8 = fine_network(case, 8).run(dt=dt, t_end=t_end)
    curves = {"FEM-BE": r_be, "FEM-CN": r_cn,
              "Cauer-N64": r_cauer, "Cauer-N1": r_lumped}
    for name, r in curves.items():
        metrics[name] = thermal_metrics(r, case)
    errors = {
        "FEM_BE_vs_Cauer64": curve_error_full(r_be, r_cauer, case.pulse_period, case.t_amb),
        "FEM_CN_vs_Cauer64": curve_error_full(r_cn, r_cauer, case.pulse_period, case.t_amb),
        "Cauer_N1_vs_Cauer64": curve_error_full(r_lumped, r_cauer, case.pulse_period, case.t_amb),
        "Cauer_N8_vs_Cauer64": curve_error_full(r_n8, r_cauer, case.pulse_period, case.t_amb),
    }
    dump("validation_transient.json", {"metrics": metrics, "errors": errors,
                                       "dt": dt, "t_end": t_end})
    plot_transient(curves, "脉冲功率激励下结温瞬态 · FEM 与 Cauer 交叉验证",
                   fig_dir / "transient_1d.png")
    print(f"[瞬态] dt={dt}s t_end={t_end}s", flush=True)
    for k, v in errors.items():
        print(f"       {k}: 峰值误差 {v['peak_rel_err_pct']:.3f}%, "
              f"摆幅误差 {v['swing_rel_err_pct']:.3f}%, "
              f"RMSE/幅值 {v['rmse_rel_amp_pct']:.3f}%", flush=True)

    # ---------- C 时间收敛（阶跃功率，BE 与 CN 分别拟合） ----------
    # 平滑 ramp 功率：避免 t=0 处功率突变造成初值层（会污染 CN 的二阶收敛）
    smooth = lambda t: case.power_per_length * (1.0 - np.exp(-t / 0.05))
    t_conv, dt_list = 0.5, [0.02, 0.01, 0.005, 0.0025, 0.00125]
    time_conv = {}
    for theta, label in ((1.0, "BE"), (0.5, "CN")):
        ref = run_transient(system, case, dt=0.00025, t_end=t_conv,
                            theta=theta, power_fn=smooth)
        errs = []
        for d in dt_list:
            r = run_transient(system, case, dt=d, t_end=t_conv,
                              theta=theta, power_fn=smooth)
            errs.append(abs(r.t_junction[-1] - ref.t_junction[-1]))
        p, c = convergence_order(np.array(dt_list), np.array(errs))
        time_conv[label] = {"dt": dt_list, "err": [float(x) for x in errs],
                            "order": p, "coef": c,
                            "ref_T": float(ref.t_junction[-1])}
        plot_convergence(np.array(dt_list), np.array(errs), p,
                         "dt [s]", f"时间收敛 · {label} (p={p:.2f})",
                         fig_dir / f"conv_time_{label.lower()}.png")
        print(f"[时间收敛] {label}: p={p:.2f}, errs={[round(x,5) for x in errs]}", flush=True)

    # ---------- D 空间收敛（几何相似同时加密 nx 与 npl） ----------
    pairs = [(10, 2), (20, 4), (40, 8), (80, 16), (160, 32)]
    hs, errs = [], []
    for nx, npl in pairs:
        c2 = case.with_params(nx=nx, ny_per_layer=npl)
        m2 = build_mesh(c2)
        s2 = assemble(c2, m2)
        T2 = solve_steady(s2, c2.power_per_length)
        errs.append(error_metrics(T2, analytic_1d_field(c2, m2))["max_abs"])
        hs.append(1.0 / npl)
        print(f"[空间收敛] nx={nx} npl={npl} -> err={errs[-1]:.3e} K", flush=True)
    p_space, _ = convergence_order(np.array(hs), np.array(errs))
    space_conv = {"nx": [p[0] for p in pairs], "npl": [p[1] for p in pairs],
                  "h": hs, "err": [float(x) for x in errs], "order": p_space}
    plot_convergence(np.array(hs), np.array(errs), p_space, "h = 1/npl",
                     f"空间收敛 · 稳态结温 (p={p_space:.2f})",
                     fig_dir / "conv_space.png")
    print(f"[空间收敛] p={p_space:.2f}", flush=True)

    # ---------- E Cauer 精细度扫描 ----------
    ns = [1, 2, 4, 8, 16, 32, 64]
    ref = fine_network(case, 128).run(dt=0.005, t_end=6.0)
    n_scan = []
    for n in ns:
        r = fine_network(case, n).run(dt=0.005, t_end=6.0)
        e = curve_error_full(r, ref, case.pulse_period, case.t_amb)
        n_scan.append({"n": n, "peak_rel_err_pct": e["peak_rel_err_pct"],
                       "swing_rel_err_pct": e["swing_rel_err_pct"],
                       "rmse_rel_amp_pct": e["rmse_rel_amp_pct"],
                       "peak_K": e["peak_a_K"]})
        print(f"[Cauer N] N={n:3d}: 峰值误差 {e['peak_rel_err_pct']:.4f}%", flush=True)
    plot_convergence(np.array([1.0 / n for n in ns]),
                     np.array([max(r["peak_rel_err_pct"], 1e-8) for r in n_scan]),
                     convergence_order(np.array([1.0 / n for n in ns]),
                                       np.array([max(r["peak_rel_err_pct"], 1e-8) for r in n_scan]))[0],
                     "1/N (每层节数倒数)", "Cauer 精细度收敛",
                     fig_dir / "cauer_n_scan.png")
    dump("validation_convergence.json", {"time": time_conv, "space": space_conv,
                                         "cauer_n": n_scan})
    print(f"\n[01 完成] 用时 {time.time()-t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
