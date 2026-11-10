"""09 汇总 v1.1 三组深化实验的表与关键数字。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RES = ROOT / "results"


def load(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))


def main():
    A = load("v11_spreading.json")
    B = load("v11_nonlinear_waveform.json")
    C = load("v11_identification.json")
    lines = ["# N6 v1.1 深化汇总（A 三维扩散 / B 非线性与波形 / C 反辨识）", ""]

    lines += ["## A1 轴对称三维扩散热阻（a/R 扫描）", "",
              "| a/R | a [mm] | 三维结温 [°C] | 相对 1D 附加温升 [K] | R_sp [K/W] | R_sp/R_1D [%] |",
              "|---|---|---|---|---|---|"]
    for r in A["sweep"]:
        lines.append(f"| {r['q']:.2f} | {r['a_mm']:.3f} | {r['tj_3d_C']:.3f} | "
                     f"{r['delta_vs_1d_K']:.3f} | {r['r_sp_K_W']:.5f} | "
                     f"{r['r_sp_over_r_1d_pct']:.2f} |")
    v = A["full_width_verification"]
    gc = A["grid_convergence"]
    lines += ["", f"- 装配自验：a/R=1 时 FEM {v['fem_tj_C']:.4f} °C vs 一维解析 "
              f"{v['analytic_tj_C']:.4f} °C（误差 {v['abs_err_K']:.6f} K）；"
              f"单位功率载荷积分 {v['unit_power_load_integral']:.9f} W。",
              f"- 网格收敛：nr=40→160 的 R_sp 变化 "
              f"{gc[0]['rel_err_vs_finest_pct']:.3f}%→0%，生产网格 nr=80 仅偏 "
              f"{gc[2]['rel_err_vs_finest_pct']:.4f}%。", ""]

    clo = A["closure"]
    lines += ["## A2 两参数闭合式（4 点标定 / 5 点留一验证）", "",
              f"形式：`{clo['form']}`，g0={clo['g0']:.5f}，g1={clo['g1']:.5f}，"
              f"标定点 {clo['calibration_q']}。", "",
              "| a/R | FEM R_sp | 预测 R_sp | 误差 [%] | 结温误差 [K] | 角色 |",
              "|---|---|---|---|---|---|"]
    for x in clo["validation"]:
        if x["q"] >= 1.0:
            continue
        lines.append(f"| {x['q']:.2f} | {x['r_sp_fem']:.5f} | {x['r_sp_pred']:.5f} | "
                     f"{x['err_pct']:+.2f} | {x['tj_err_K']:+.3f} | {x['role']} |")
    hs = A["halfspace_check"]
    lines += ["", f"- 留一最大误差 {clo['held_out_max_abs_err_pct']:.2f}%，"
              f"结温 RMSE {clo['held_out_rmse_K']:.3f} K。",
              f"- 经典半空间式 8/(3π²ka)（k=Si）在 a/R=0.2 给 "
              f"{hs['r_sp_halfspace_kSi_K_W']:.4f} K/W，实测 "
              f"{hs['r_sp_fem_K_W']:.4f} K/W，**低估 {hs['underpredict_factor_kSi']:.3f} 倍**："
              "薄层 + 低导热 TIM 限制纵向扩散，半无限体假设不成立。", ""]

    tc = A["transient_correction"]
    ps = A["period_sweep"]
    lines += ["## A3 静态扩散热阻修正集总网络", "",
              "| 模型 | 峰值 [°C] | 谷值 [°C] | 摆幅 [K] |",
              "|---|---|---|---|"]
    for key, lab in [("axisym_fem", "轴对称 FEM（真值）"),
                     ("ladder_1d_uncorrected", "一维 Cauer（未修正）"),
                     ("ladder_corrected_static", "一维 Cauer + R_sp（修正）")]:
        m = tc[key]
        lines.append(f"| {lab} | {m['t_peak']:.3f} | {m['t_trough']:.3f} | {m['swing']:.3f} |")
    lines += ["", f"- a/R=0.2、T=2 s：峰值误差 {tc['peak_err_before_K']:.2f} K → "
              f"{tc['peak_err_after_K']:.3f} K；摆幅误差 "
              f"{tc['swing_err_before_pct']:.2f}% → {tc['swing_err_after_pct']:.2f}%。",
              f"- 但静态串联把扩散区视为零热容：τ_spread="
              f"{ps['tau_spread_s']*1e3:.3f} ms，周期扫描摆幅误差 "
              + "/".join(f"{p['swing_err_pct']:.2f}%" for p in ps["rows"])
              + f"（T={'/'.join(str(p['period_s']) for p in ps['rows'])} s）。",
              f"- {ps['validity']}；最小有效周期 {ps['min_valid_period_s']} s。", ""]

    nl = B["nonlinear_kT"]
    lines += ["## B1 温度相关导热系数 k(T)", "",
              "| alpha [1/K] | 稳态结温 [°C] | 相对常物性 [K] | 占温升 [%] | Picard 迭代 |",
              "|---|---|---|---|---|"]
    for x in nl["rows"]:
        lines.append(f"| {x['alpha_per_K']:+.4f} | {x['tj_C']:.4f} | "
                     f"{x['delta_vs_linear_K']:+.4f} | {x['delta_pct_of_rise']:+.2f} | "
                     f"{x['iterations']} |")
    lines += ["", f"- α=0 回归误差 {nl['alpha_zero_regression_K']:.1e} K；Si 量级 "
              f"α=-0.003/K 时常物性假设低估结温 {nl['si_like_delta_K']:.3f} K。", ""]

    ds = B["duty_sweep"]
    lines += ["## B2 占空比扫描（闭式 / 精细 Cauer / FEM）", "",
              "| D | FEM 峰值 | Cauer 峰值 | 单节闭式峰值 | 闭式峰值误差 | FEM 摆幅 | 闭式摆幅误差 |",
              "|---|---|---|---|---|---|---|"]
    for x in ds["rows"]:
        lines.append(f"| {x['duty']:.1f} | {x['fem_peak_C']:.2f} | {x['cauer_peak_C']:.2f} | "
                     f"{x['closed_peak_C']:.2f} | {x['peak_closed_err_pct']:.2f}% | "
                     f"{x['fem_swing_K']:.2f} | {x['swing_closed_err_pct']:.2f}% |")
    pr = B["period_sweep"]["rows"]
    lines += ["", f"- FEM 与精细 Cauer 峰值最大差 {ds['max_fem_vs_cauer_K']:.4f} K；"
              f"能量守恒均值 T_amb+D·P·R 最大偏差 {ds['max_energy_invariant_err_K']:.4f} K。",
              "- 周期扫描（D=0.5）单节闭式解摆幅误差 " +
              " / ".join(f"T/τ={p['T_over_tau']:.2f}: {p['swing_err_pct']:.1f}%"
                         for p in pr) + "。", ""]

    th = B["theta_study"]
    lines += ["## B3 不连续载荷下 θ 法的阶数退化", "",
              "| dt [s] | θ=0.5 | θ=0.6 | θ=0.75 | θ=1.0 |", "|---|---|---|---|---|"]
    for row in th["peak_error_K"]:
        lines.append(f"| {row['dt']} | " + " | ".join(
            f"{row[f'theta_{t:g}']:.4f}" for t in (0.5, 0.6, 0.75, 1.0)) + " |")
    lines += ["", "- 经验收敛阶：" +
              ", ".join(f"{k}={v:.2f}" for k, v in th["empirical_order"].items()) +
              f"；CN/BE 优劣交叉步长 {th['cn_be_crossover_dt_s']:.5f} s。",
              f"- {th['monotonicity']}。", ""]

    fim = C["fisher"]["rows"]
    lines += ["## C 参数反辨识可辨识性", "",
              "| 观测集 | 点数 | FIM 秩 | 条件数 | corr(ln k, ln h) |",
              "|---|---|---|---|---|"]
    for o, x in fim.items():
        lines.append(f"| {C['observables'][o]} | {x['n_obs']} | {x['effective_rank']} | "
                     f"{x['condition_number']:.3g} | {x['corr_lnK_lnH']:+.4f} |")
    deg = C["steady_degeneracy"]
    lines += ["", f"- 稳态退化：4 个初值收敛到 k_TIM {min(x['fit_k'] for x in deg['rows']):.2f}~"
              f"{max(x['fit_k'] for x in deg['rows']):.2f}（跨度 {deg['spread_k']:.3f}），"
              f"代价全为 0 → 结构性不可辨识。",
              f"- 阶跃数据无噪声可精确还原 (5.000, 5000.0)；"
              f"{C['monte_carlo']['note']}", "",
              f"- 解析阶跃响应（对称化 eigh）与 expm 逐步推进差 "
              f"{C['exact_step_verification']['max_abs_diff_K']:.2e} K。", ""]

    lines += ["## 复现", "", "```bash",
              'PY="D:/项目/venvs/flsoh-py312/Scripts/python.exe"',
              "bash run_all.sh", "```", "",
              "脚本 06/07/08 分别产出 `v11_spreading.json`、`v11_nonlinear_waveform.json`、"
              "`v11_identification.json`，本表与 `v11_summary.json` 由 09 生成。"]

    (RES / "v11_summary_tables.md").write_text("\n".join(lines) + "\n",
                                                encoding="utf-8")
    summary = {
        "version": "v1.1.0",
        "axisym": {
            "full_width_err_K": v["abs_err_K"],
            "grid_nr80_err_pct": gc[2]["rel_err_vs_finest_pct"],
            "closure_heldout_max_err_pct": clo["held_out_max_abs_err_pct"],
            "halfspace_underpredict_factor": hs["underpredict_factor_kSi"],
            "tj_3d_q02_C": A["sweep"][8]["tj_3d_C"],
            "r_sp_q02_K_W": A["sweep"][8]["r_sp_K_W"],
            "peak_err_before_K": tc["peak_err_before_K"],
            "peak_err_after_K": tc["peak_err_after_K"],
            "tau_spread_ms": ps["tau_spread_s"] * 1e3,
            "min_valid_period_s": ps["min_valid_period_s"],
        },
        "nonlinear": {
            "si_like_delta_K": nl["si_like_delta_K"],
            "alpha_zero_regression_K": nl["alpha_zero_regression_K"],
            "max_fem_vs_cauer_K": ds["max_fem_vs_cauer_K"],
            "energy_invariant_max_err_K": ds["max_energy_invariant_err_K"],
            "worst_swing_err_pct": min(p["swing_err_pct"] for p in pr),
            "theta_orders": th["empirical_order"],
            "theta_crossover_dt_s": th["cn_be_crossover_dt_s"],
        },
        "identification": {
            "steady_fim_rank": fim["steady"]["effective_rank"],
            "steady_fim_cond": fim["steady"]["condition_number"],
            "step_top_fim_cond": fim["step_top"]["condition_number"],
            "steady_solution_spread_k": deg["spread_k"],
            "exact_step_err_K": C["exact_step_verification"]["max_abs_diff_K"],
            "fim_vs_mc_gap_pct": C["monte_carlo"]["fim_vs_mc_mean_rel_gap_pct"],
            "k_sd_top_pct": C["monte_carlo"]["k_sd_top_pct"],
            "k_sd_dual_pct": C["monte_carlo"]["k_sd_dual_pct"],
        },
        "tests": 33,
    }
    (RES / "v11_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    print("[09] 已写 results/v11_summary_tables.md 与 v11_summary.json")
    print("[09] 关键：三维 a/R=0.2 结温", round(summary["axisym"]["tj_3d_q02_C"], 2),
          "°C；修正后峰值误差", round(summary["axisym"]["peak_err_after_K"], 3), "K")
    print("[09] 稳态 FIM 秩", summary["identification"]["steady_fim_rank"],
          "阶跃 FIM 条件数", round(summary["identification"]["step_top_fim_cond"], 2))


if __name__ == "__main__":
    main()
