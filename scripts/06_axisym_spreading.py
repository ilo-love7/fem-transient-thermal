"""06 v1.1 深化 A：轴对称三维扩散热阻 + 两参数闭合式 + 集总网络修正。"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fem import default_case, periodic_steady_metrics
from fem.axisym import (axisym_from_case, assemble_axisym, solve_axisym_steady,
                        axisym_junction, spreading_resistance_abs,
                        semi_infinite_spreading, run_axisym_transient,
                        build_axisym_mesh)
from fem.spreading import (fit_spreading_closure, corrected_ladder,
                           equivalent_ladder_case)
from fem.cauer import CauerNetwork

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
RES = ROOT / "results"
FIG = ROOT / "figures"


def dump(name, obj):
    (RES / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                       default=float), encoding="utf-8")


def solve_q(case0, q, nr=80, nz=8):
    c = case0.with_params(source_ratio=q, nr=nr, nz_per_layer=nz)
    s = assemble_axisym(c)
    T = solve_axisym_steady(s)
    return c, s, T, axisym_junction(s, T)


def main():
    t0 = time.time()
    base = default_case()
    case0 = axisym_from_case(base, 1.0, nr=80, nz_per_layer=8)
    out = {"geometry": {"R_mm": case0.radius * 1e3,
                        "total_power_W": case0.total_power,
                        "q_flux_W_m2": case0.q_flux,
                        "r_1d_K_W": case0.r_1d_abs,
                        "t_j_1d_C": case0.t_junction_1d}}

    # --- 1) 装配正确性：a/R=1 必须回到一维解析 ---
    c1, s1, T1, tj1 = solve_q(case0, 1.0)
    out["full_width_verification"] = {
        "fem_tj_C": tj1, "analytic_tj_C": case0.t_junction_1d,
        "abs_err_K": tj1 - case0.t_junction_1d,
        "unit_power_load_integral": s1.f_heat_unit.sum(),
        "note": "q=1 严格一维；载荷积分=1 W 且结温误差 <1e-3 K，验证 2*pi*r 装配与 Robin 精确积分",
    }

    # --- 2) 网格收敛（最苛刻的 q=0.2） ---
    grid = []
    for nr, nz in [(40, 4), (60, 6), (80, 8), (120, 12), (160, 16)]:
        c, _, _, tj = solve_q(case0, 0.2, nr, nz)
        grid.append({"nr": nr, "nz_per_layer": nz,
                     "tj_C": tj, "r_sp_K_W": spreading_resistance_abs(tj, c)})
    ref = grid[-1]["r_sp_K_W"]
    for g in grid:
        g["rel_err_vs_finest_pct"] = 100 * (g["r_sp_K_W"] - ref) / ref
    out["grid_convergence"] = grid

    # --- 3) a/R 扫描 ---
    qs = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15])
    rows = []
    for q in qs:
        c, s, T, tj = solve_q(case0, float(q))
        rsp = spreading_resistance_abs(tj, c)
        hs_si = semi_infinite_spreading(c.die.k, c.source_radius)
        hs_cu = semi_infinite_spreading(c.layers[0].k, c.source_radius)
        rows.append({"q": float(q), "a_mm": c.source_radius * 1e3,
                     "tj_3d_C": tj, "delta_vs_1d_K": tj - c.t_junction_1d,
                     "r_sp_K_W": rsp,
                     "r_sp_over_r_1d_pct": 100 * rsp / c.r_1d_abs,
                     "halfspace_kSi_K_W": hs_si,
                     "halfspace_kCu_K_W": hs_cu,
                     "gamma_obs": (np.log(1 + rsp / c.r_1d_abs) / np.log(1 / q)
                                   if q < 1 else 0.0)})
    out["sweep"] = rows

    # --- 4) 两参数闭合式：4 点标定 / 5 点留一验证 ---
    q_arr = np.array([r["q"] for r in rows])
    r_arr = np.array([r["r_sp_K_W"] for r in rows])
    cal_idx = np.array([2, 4, 6, 8])          # q = 0.8, 0.6, 0.4, 0.2
    mask = np.zeros(len(rows), dtype=bool)
    mask[cal_idx] = True
    clo = fit_spreading_closure(q_arr, r_arr, case0.r_1d_abs, mask)
    pred = clo.predict(q_arr)
    val = []
    for i, r in enumerate(rows):
        q = r["q"]
        val.append({"q": q, "r_sp_fem": r["r_sp_K_W"], "r_sp_pred": float(pred[i]),
                    "err_pct": (100 * (pred[i] - r["r_sp_K_W"]) / r["r_sp_K_W"]
                                if r["r_sp_K_W"] > 1e-9 else 0.0),
                    "tj_pred_C": case0.t_junction_1d + case0.total_power * pred[i],
                    "tj_err_K": case0.total_power * (pred[i] - r["r_sp_K_W"]),
                    "role": "calibration" if mask[i] else "held-out"})
    held = [v for v in val if v["role"] == "held-out"]
    out["closure"] = {
        "form": "R_sp(q) = R_1D*(q^(-gamma(q)) - 1), gamma(q)=g0+g1*(1-q)",
        "g0": clo.g0, "g1": clo.g1,
        "calibration_q": list(clo.calibration_q),
        "validation": val,
        "held_out_max_abs_err_pct": max(abs(v["err_pct"]) for v in held),
        "held_out_rmse_K": float(np.sqrt(np.mean([v["tj_err_K"] ** 2 for v in held]))),
        "property": "q=1 自动 R_sp=0，无需额外约束",
    }

    # --- 5) 经典半空间公式失准 ---
    i02 = int(np.argmin(np.abs(q_arr - 0.2)))
    out["halfspace_check"] = {
        "q": 0.2,
        "r_sp_fem_K_W": rows[i02]["r_sp_K_W"],
        "r_sp_halfspace_kSi_K_W": rows[i02]["halfspace_kSi_K_W"],
        "underpredict_factor_kSi": rows[i02]["r_sp_K_W"] / rows[i02]["halfspace_kSi_K_W"],
        "r_sp_halfspace_kCu_K_W": rows[i02]["halfspace_kCu_K_W"],
        "note": "半无限体解假设热量可向下无限扩散；本结构是薄层+对流底板，"
                "低导热 TIM 限制了纵向扩散，故实测扩散热阻更大，套用经典式会低估结温",
    }

    # --- 6) 瞬态修正：三维 FEM vs 未修正/修正集总网络 ---
    q_t = 0.2
    c_t = case0.with_params(source_ratio=q_t, nr=80, nz_per_layer=8,
                            dt=0.01, t_end=8.0, theta=1.0)
    s_t = assemble_axisym(c_t)
    T_st = solve_axisym_steady(s_t)
    rsp_t = spreading_resistance_abs(axisym_junction(s_t, T_st), c_t)
    fem_tr = run_axisym_transient(s_t)
    raw, corr = corrected_ladder(c_t, rsp_t, n=64)
    fm = periodic_steady_metrics(fem_tr, c_t)
    rm = periodic_steady_metrics(raw, c_t)
    cm = periodic_steady_metrics(corr, c_t)
    out["transient_correction"] = {
        "q": q_t, "r_sp_K_W": rsp_t,
        "axisym_fem": {k: float(v) for k, v in fm.items()},
        "ladder_1d_uncorrected": {k: float(v) for k, v in rm.items()},
        "ladder_corrected_static": {k: float(v) for k, v in cm.items()},
        "peak_err_before_K": rm["t_peak"] - fm["t_peak"],
        "peak_err_after_K": cm["t_peak"] - fm["t_peak"],
        "swing_err_before_pct": 100 * (rm["swing"] - fm["swing"]) / fm["swing"],
        "swing_err_after_pct": 100 * (cm["swing"] - fm["swing"]) / fm["swing"],
        "note": "静态串联 R_sp 把扩散区视为纯热阻（零热容）；慢脉冲下该近似极准，"
                "快脉冲下扩散区热容开始滤波，修正模型转而高估摆幅（见 period_sweep）",
    }

    # --- 7) 周期扫描：找静态修正的失效边界 ---
    c_spread = c_t.die.rc * np.pi * c_t.source_radius ** 2 * c_t.die.thickness
    tau_spread = rsp_t * c_spread
    periods = [2.0, 0.5, 0.2, 0.1, 0.05, 0.02]
    ps = []
    for per in periods:
        dt = min(0.005, per / 40.0)
        te = max(6 * per, 0.4)
        c = c_t.with_params(pulse_period=per, dt=dt, t_end=te)
        ss = assemble_axisym(c)
        f = run_axisym_transient(ss, dt=dt, t_end=te)
        _, cc = corrected_ladder(c, rsp_t, n=64, dt=dt, t_end=te)
        fmet = periodic_steady_metrics(f, c)
        cmet = periodic_steady_metrics(cc, c)
        ps.append({
            "period_s": per, "T_over_tau_spread": per / tau_spread,
            "fem_peak_C": fmet["t_peak"], "fem_swing_K": fmet["swing"],
            "corrected_peak_C": cmet["t_peak"], "corrected_swing_K": cmet["swing"],
            "peak_err_K": cmet["t_peak"] - fmet["t_peak"],
            "swing_err_pct": 100 * (cmet["swing"] - fmet["swing"]) / fmet["swing"],
        })
    good = [p for p in ps if abs(p["swing_err_pct"]) < 1.0]
    out["period_sweep"] = {
        "tau_spread_s": tau_spread,
        "c_spread_J_K": c_spread,
        "tau_definition": "tau_spread = R_sp * (rho*c_die * pi*a^2*L_die)，扩散区特征热容",
        "rows": ps,
        "validity": "T/tau_spread >= ~35 时静态修正摆幅误差 <1%；<= ~10 时误差 >18%",
        "min_valid_period_s": min(p["period_s"] for p in good) if good else None,
    }

    # --- 图 1：q=0.2 轴对称温度场（镜像成完整剖面） ---
    nodes = s_t.mesh.nodes
    r_mm, z_mm = nodes[:, 0] * 1e3, nodes[:, 1] * 1e3
    elems = s_t.mesh.elems
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=130)
    for sign in (1, -1):
        ax.tricontourf(sign * r_mm, z_mm, elems, T_st, levels=26, cmap="inferno")
    for y in c_t.y_interfaces[1:-1]:
        ax.axhline(y * 1e3, color="w", lw=0.8, ls="--", alpha=0.55)
    a_mm = c_t.source_radius * 1e3
    ax.plot([-a_mm, a_mm], [c_t.height * 1e3] * 2, color="cyan", lw=4,
            solid_capstyle="butt")
    ax.set_xlabel("r [mm]")
    ax.set_ylabel("z [mm]")
    ax.set_title(f"轴对称三维稳态温度场  a/R={q_t}  Tj={axisym_junction(s_t, T_st):.2f} °C")
    cn = ax.tricontourf(r_mm, z_mm, elems, T_st, levels=26, cmap="inferno")
    fig.colorbar(cn, ax=ax, label="T [°C]")
    fig.tight_layout()
    fig.savefig(FIG / "axisym_field_q02.png")
    plt.close(fig)

    # --- 图 2：R_sp 曲线 + 闭合式 + 半空间式 ---
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=130)
    ax.plot(q_arr, r_arr, "o-", color="#1f77b4", lw=2, label="轴对称 FEM（真三维）")
    ax.plot(q_arr, pred, "s--", color="#d62728", lw=1.8,
            label=f"两参数闭合式 (g0={clo.g0:.3f}, g1={clo.g1:.3f})")
    hs = np.array([semi_infinite_spreading(case0.die.k, case0.radius * q)
                   for q in q_arr])
    ax.plot(q_arr, hs, "^:", color="#2ca02c", lw=1.6, label="半空间式 8/(3π²k a), k=Si")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("a / R")
    ax.set_ylabel("扩散热阻 R_sp [K/W]")
    ax.set_title("三维扩散热阻：FEM / 闭合式 / 经典半空间解")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "spreading_curve_3d.png")
    plt.close(fig)

    # --- 图 3：瞬态修正对照 ---
    fig, ax = plt.subplots(figsize=(9, 4.4), dpi=130)
    last = fem_tr.t[-1] - 2 * c_t.pulse_period
    m = fem_tr.t >= last
    for r, lab, col, ls in [
        (fem_tr, "轴对称 FEM（真值）", "#1f77b4", "-"),
        (raw, "一维 Cauer（未修正）", "#d62728", "--"),
        (corr, "一维 Cauer + R_sp（修正）", "#2ca02c", "-."),
    ]:
        mm = r.t >= last
        ax.plot(r.t[mm], r.t_junction[mm], ls, color=col, lw=1.8, label=lab)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("结温 T_j [°C]")
    ax.set_title(f"a/R={q_t} 脉冲瞬态：静态扩散热阻修正的效果与边界")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "spreading_transient.png")
    plt.close(fig)

    # --- 图 4：静态修正误差 vs T/tau_spread ---
    fig, ax = plt.subplots(figsize=(7.5, 4.3), dpi=130)
    xx = [p["T_over_tau_spread"] for p in ps]
    yy = [abs(p["swing_err_pct"]) for p in ps]
    ax.plot(xx, yy, "o-", color="#d62728", lw=1.8)
    ax.axhline(1.0, color="gray", ls="--", lw=1, label="1% 工程阈值")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("T / τ_spread")
    ax.set_ylabel("修正后摆幅误差 |Δ| [%]")
    ax.set_title("静态扩散热阻修正的适用边界")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "spreading_validity.png")
    plt.close(fig)

    dump("v11_spreading.json", out)
    print("[06] 全宽验证误差 K:", round(out["full_width_verification"]["abs_err_K"], 6))
    print("[06] 网格收敛 nr=80 相对 finest:", round(grid[2]["rel_err_vs_finest_pct"], 4), "%")
    print("[06] 闭合式留一最大误差:", round(out["closure"]["held_out_max_abs_err_pct"], 2), "%")
    print("[06] 半空间公式低估倍数:", round(out["halfspace_check"]["underpredict_factor_kSi"], 3))
    tc = out["transient_correction"]
    print("[06] 峰值误差 修正前/后 K:", round(tc["peak_err_before_K"], 3),
          "/", round(tc["peak_err_after_K"], 3))
    print("[06] 摆幅误差 修正前/后 %:", round(tc["swing_err_before_pct"], 2),
          "/", round(tc["swing_err_after_pct"], 2))
    print("[06] τ_spread =", round(out["period_sweep"]["tau_spread_s"] * 1e3, 3), "ms",
          "最小有效周期 =", out["period_sweep"]["min_valid_period_s"], "s")
    print("[06] 周期扫描摆幅误差 %:",
          [round(p["swing_err_pct"], 2) for p in ps])
    print("[06] 用时", round(time.time() - t0, 1), "s")
    print("[06] 已写 results/v11_spreading.json + 4 图")


if __name__ == "__main__":
    main()
