"""07 v1.1 深化 B：k(T) 非线性 / 占空比与周期波形 / θ 法在不连续载荷下的阶数退化。"""
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

from fem import (default_case, build_mesh, assemble, solve_steady,
                 run_transient, periodic_steady_metrics, fine_network)
from fem.nonlinear import (solve_nonlinear_steady, rc_periodic_closed_form,
                           theta_artifacts)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
RES, FIG = ROOT / "results", ROOT / "figures"


def dump(name, obj):
    (RES / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                       default=float), encoding="utf-8")


def main():
    t0 = time.time()
    c = default_case()
    mesh = build_mesh(c)
    sys = assemble(c, mesh)
    T_lin = solve_steady(sys, c.power_per_length)
    tj_lin = float(T_lin[mesh.junction_node])
    R, C = c.r_effective, c.heat_capacity_per_length
    tau = R * C
    out = {"linear_reference": {"tj_C": tj_lin, "r_effective": R,
                                "heat_capacity": C, "tau_s": tau}}

    # ---------- B1 温度相关物性 k(T) ----------
    alphas = [-0.005, -0.003, -0.001, 0.0, 0.001, 0.003]
    nl = []
    for a in alphas:
        r = solve_nonlinear_steady(c, a, mesh=mesh)
        nl.append({"alpha_per_K": a, "tj_C": r.t_junction,
                   "delta_vs_linear_K": r.t_junction - tj_lin,
                   "delta_pct_of_rise": 100 * (r.t_junction - tj_lin)
                   / (tj_lin - c.t_amb),
                   "iterations": r.iterations, "converged": r.converged,
                   "k_min": r.k_min, "k_max": r.k_max,
                   "final_residual": r.residual_history[-1]})
    out["nonlinear_kT"] = {
        "model": "k(T) = k0*[1 + alpha*(T - T_amb)]，逐单元重心温度，Picard + 欠松弛 0.7",
        "rows": nl,
        "alpha_zero_regression_K": next(x["tj_C"] for x in nl
                                        if x["alpha_per_K"] == 0.0) - tj_lin,
        "si_like_alpha": -0.003,
        "si_like_delta_K": next(x["delta_vs_linear_K"] for x in nl
                                if x["alpha_per_K"] == -0.003),
        "note": "常物性假设在 Si 量级 alpha=-0.003/K 下低估结温约 3.4 K；"
                "对流热阻占 82% 使误差被稀释，但不可忽略",
    }

    # ---------- B2a 占空比扫描（闭式 / 精细 Cauer / FEM 三方对照） ----------
    duties = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    duty_rows = []
    for D in duties:
        cc = c.with_params(duty=D, dt=0.005, t_end=10.0)
        cf = rc_periodic_closed_form(c.power_per_length, R, tau,
                                     c.t_amb, c.pulse_period, D)
        net = fine_network(cc, 64).run(dt=0.005, t_end=10.0)
        fem = run_transient(sys, cc, dt=0.005, t_end=10.0)
        nm = periodic_steady_metrics(net, cc)
        fm = periodic_steady_metrics(fem, cc)
        duty_rows.append({
            "duty": D,
            "closed_peak_C": cf["t_peak"], "closed_swing_K": cf["swing"],
            "closed_avg_C": cf["t_avg_energy"],
            "cauer_peak_C": nm["t_peak"], "cauer_swing_K": nm["swing"],
            "fem_peak_C": fm["t_peak"], "fem_swing_K": fm["swing"],
            "fem_avg_C": fm["t_avg"],
            "peak_closed_err_pct": 100 * (cf["t_peak"] - fm["t_peak"])
            / (fm["t_peak"] - c.t_amb),
            "swing_closed_err_pct": 100 * (cf["swing"] - fm["swing"]) / fm["swing"],
            "fem_vs_cauer_peak_K": fm["t_peak"] - nm["t_peak"],
            "energy_invariant_err_K": fm["t_avg"] - cf["t_avg_energy"],
        })
    out["duty_sweep"] = {
        "period_s": c.pulse_period, "rows": duty_rows,
        "max_fem_vs_cauer_K": max(abs(x["fem_vs_cauer_peak_K"]) for x in duty_rows),
        "max_energy_invariant_err_K": max(abs(x["energy_invariant_err_K"])
                                          for x in duty_rows),
        "note": "能量守恒给出 T_avg = T_amb + D*P*R，与周期无关，单节闭式解精确满足；"
                "但单时间常数模型把薄芯片层的快模态抹掉，峰值/摆幅系统性偏低",
    }

    # ---------- B2b 周期扫描（充分长仿真，避免慢模态未达周期稳态） ----------
    periods = [4.0, 2.0, 0.5, 0.2, 0.05]
    per_rows = []
    for P in periods:
        te = max(12 * tau, 12 * P)
        dt = min(0.001, P / 40.0)
        cc = c.with_params(pulse_period=P, duty=0.5, dt=dt, t_end=te)
        cf = rc_periodic_closed_form(c.power_per_length, R, tau,
                                     c.t_amb, P, 0.5)
        net = fine_network(cc, 64).run(dt=dt, t_end=te)
        nm = periodic_steady_metrics(net, cc)
        per_rows.append({
            "period_s": P, "T_over_tau": P / tau,
            "closed_peak_C": cf["t_peak"], "closed_swing_K": cf["swing"],
            "cauer_peak_C": nm["t_peak"], "cauer_swing_K": nm["swing"],
            "peak_err_pct": 100 * (cf["t_peak"] - nm["t_peak"])
            / (nm["t_peak"] - c.t_amb),
            "swing_err_pct": 100 * (cf["swing"] - nm["swing"]) / nm["swing"],
        })
    out["period_sweep"] = {
        "duty": 0.5, "rows": per_rows,
        "note": "单节闭式解的摆幅误差随 T/tau 减小单调恶化（最快脉冲 -84%），"
                "因为摆动由薄芯片层快模态主导而单节模型的 tau 是整体平均",
    }

    # ---------- B3 θ 法在不连续载荷下的阶数退化 ----------
    cc = c.with_params(pulse_period=0.2, duty=0.5, t_end=2.0)
    ref = fine_network(cc, 128).run(dt=0.0002, t_end=2.0)
    ref_peak = periodic_steady_metrics(ref, cc)["t_peak"]
    dts = [0.02, 0.01, 0.005, 0.0025]
    thetas = [0.5, 0.6, 0.75, 1.0]
    grid, artifacts = [], []
    for dt in dts:
        row = {"dt": dt}
        for th in thetas:
            r = run_transient(sys, cc.with_params(dt=dt), dt=dt,
                              t_end=2.0, theta=th)
            met = periodic_steady_metrics(r, cc)
            row[f"theta_{th:g}"] = met["t_peak"] - ref_peak
            if dt == dts[0]:
                artifacts.append({"theta": th, **theta_artifacts(r, c.t_amb)})
        grid.append(row)
    slope = {}
    for th in thetas:
        e = [g[f"theta_{th:g}"] for g in grid]
        slope[f"theta_{th:g}"] = float(np.polyfit(np.log(dts), np.log(e), 1)[0])
    # CN 与 BE 谁更准的交叉步长（对数插值）
    diff = np.array([g["theta_0.5"] - g["theta_1"] for g in grid])
    logdt = np.log(dts)
    cross = None
    for i in range(len(dts) - 1):
        if diff[i] * diff[i + 1] < 0:
            w = diff[i] / (diff[i] - diff[i + 1])
            cross = float(np.exp(logdt[i] + w * (logdt[i + 1] - logdt[i])))
            break
    out["theta_study"] = {
        "reference": "fine Cauer N=128 + 精确 expm 传播子（无时间离散误差）",
        "ref_peak_C": ref_peak,
        "period_s": 0.2, "duty": 0.5,
        "peak_error_K": grid, "empirical_order": slope,
        "cn_be_crossover_dt_s": cross,
        "artifacts_at_dt0_02": artifacts,
        "monotonicity": "四种 θ 在关断相内均单调下降，未观测到 CN 振铃；"
                        "CN 的代价是粗步长下的峰值过冲而非非物理振荡",
        "note": "不连续方波载荷使所有 θ 的经验收敛阶退化到 0.65~0.90（远低于 CN 的 2 阶）；"
                "粗步长下 L-稳定的 BE 更准，细步长下 CN 的二阶主项才占优",
    }

    # ---------- 图 1：k(T) 影响 + Picard 收敛史 ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.1), dpi=130)
    ax1.plot([x["alpha_per_K"] * 1e3 for x in nl],
             [x["tj_C"] for x in nl], "o-", color="#d62728", lw=2)
    ax1.axhline(tj_lin, color="gray", ls="--", label="常物性 FEM")
    ax1.set_xlabel("alpha [1e-3 /K]")
    ax1.set_ylabel("稳态结温 [°C]")
    ax1.set_title("温度相关导热系数对结温的影响")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9)
    r = solve_nonlinear_steady(c, -0.003, mesh=mesh)
    ax2.semilogy(range(1, len(r.residual_history) + 1), r.residual_history,
                 "o-", color="#1f77b4", lw=1.6)
    ax2.set_xlabel("Picard 迭代")
    ax2.set_ylabel("max |ΔT| [K]")
    ax2.set_title(f"非线性收敛史（α=-0.003，{r.iterations} 次）")
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG / "nonlinear_kT.png")
    plt.close(fig)

    # ---------- 图 2：占空比响应 ----------
    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=130)
    D = [x["duty"] for x in duty_rows]
    ax.plot(D, [x["fem_peak_C"] for x in duty_rows], "o-", color="#1f77b4",
            lw=2, label="FEM 峰值")
    ax.plot(D, [x["cauer_peak_C"] for x in duty_rows], "s", color="#2ca02c",
            ms=5, label="精细 Cauer 峰值")
    ax.plot(D, [x["closed_peak_C"] for x in duty_rows], "^--",
            color="#d62728", lw=1.6, label="单节 RC 闭式解峰值")
    ax.plot(D, [x["closed_avg_C"] for x in duty_rows], ":", color="gray",
            lw=1.8, label="能量守恒均值 T_amb+D·P·R")
    ax.set_xlabel("占空比 D")
    ax.set_ylabel("结温 [°C]")
    ax.set_title("占空比扫描：单节闭式解系统性低估峰值")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "duty_response.png")
    plt.close(fig)

    # ---------- 图 3：周期响应误差 ----------
    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=130)
    xx = [x["T_over_tau"] for x in per_rows]
    ax.plot(xx, [x["swing_err_pct"] for x in per_rows], "o-", color="#d62728",
            lw=2, label="摆幅误差")
    ax.plot(xx, [x["peak_err_pct"] for x in per_rows], "s--", color="#1f77b4",
            lw=1.6, label="峰值温升误差")
    ax.axhline(0, color="gray", lw=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("T / τ")
    ax.set_ylabel("单节闭式解误差 [%]")
    ax.set_title("周期扫描：快脉冲下多时间常数效应主导")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "period_response.png")
    plt.close(fig)

    # ---------- 图 4：θ 法误差收敛 ----------
    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=130)
    for th, col in zip(thetas, ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]):
        ax.loglog(dts, [g[f"theta_{th:g}"] for g in grid], "o-", color=col,
                  lw=1.7, label=f"θ={th:g} (p={slope[f'theta_{th:g}']:.2f})")
    ax.set_xlabel("dt [s]")
    ax.set_ylabel("峰值误差 [K]")
    ax.set_title("不连续载荷下 θ 法的误差收敛（参考：精确传播子）")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "theta_convergence.png")
    plt.close(fig)

    dump("v11_nonlinear_waveform.json", out)
    print("[07] α=0 回归误差 K:", round(out["nonlinear_kT"]["alpha_zero_regression_K"], 9))
    print("[07] Si 量级 α=-0.003 低估结温 K:",
          round(out["nonlinear_kT"]["si_like_delta_K"], 4))
    print("[07] 占空比扫描 FEM vs Cauer 最大差 K:",
          round(out["duty_sweep"]["max_fem_vs_cauer_K"], 4),
          "能量守恒误差 K:", round(out["duty_sweep"]["max_energy_invariant_err_K"], 4))
    print("[07] 周期扫描摆幅误差 %:",
          [round(x["swing_err_pct"], 1) for x in per_rows])
    print("[07] θ 经验阶:", {k: round(v, 2) for k, v in slope.items()},
          "CN/BE 交叉 dt =", out["theta_study"]["cn_be_crossover_dt_s"])
    print("[07] dt=0.02 关断相最大升温 K:",
          [round(a["max_rise_in_off_phase_K"], 6) for a in artifacts])
    print("[07] 用时", round(time.time() - t0, 1), "s")
    print("[07] 已写 results/v11_nonlinear_waveform.json + 4 图")


if __name__ == "__main__":
    main()
