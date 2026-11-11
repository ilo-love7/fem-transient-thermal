"""13 v1.2 汇总：把三个深化实验的 JSON 汇成一张可复核的 markdown 指标表。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RES = ROOT / "results"


def load(name):
    p = RES / name
    if not p.exists():
        raise SystemExit(f"缺少 {p}，请先运行 scripts/10-12")
    return json.loads(p.read_text(encoding="utf-8"))


def table(headers, rows, fmt=None):
    fmt = fmt or {}
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        cells = []
        for h in headers:
            v = r.get(h, "")
            if isinstance(v, float):
                cells.append(fmt.get(h, lambda x: f"{x:.4g}")(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    d13 = load("v12_nonlinear_transient.json")
    d14 = load("v12_dynamic_diffusion.json")
    d15 = load("v12_radiative_contact.json")
    lines = ["# N6 v1.2 深化实验汇总表", "",
             "> 由 `scripts/13_summary_v12.py` 从三个结果 JSON 自动生成；"
             "完整推导见 `reports/基准评测报告-N6.md` §10–§12。", ""]

    # --- E13 ---
    lines += ["## E13 k(T) 瞬态 Newton / Picard", "",
              f"网格 {d13['protocol']['nnode']} 节点，dt={d13['protocol']['dt_s']} s，"
              f"t_end={d13['protocol']['t_end_s']} s。", "",
              table(["alpha", "method", "t_peak_C", "total_iterations",
                     "max_iter_per_step", "max_energy_residual_W_per_m",
                     "peak_shift_vs_constant_K", "wall_s"],
                    d13["alpha_sweep"]), "",
              "**容差审计 / 人为截断迭代**", "",
              table(["tol", "total_iterations", "max_energy_residual_W_per_m",
                     "max_rel_energy_residual", "converged"],
                    d13["tolerance_audit"]), ""]
    sc = d13["steady_crosscheck"]
    lines += [f"常功率交叉验证：瞬态终值 {sc['transient_final_C']:.5f} °C vs "
              f"v1.1 稳态 Picard {sc['v11_steady_picard_C']:.5f} °C，"
              f"差 {sc['abs_diff_K']:.5f} K。", ""]

    # --- E14 ---
    diag = d14["single_pole_diagnosis"]
    lines += ["## E14 动态扩散 RC 极点", "",
              f"R_sp={d14['protocol']['r_sp_K_W']:.4f} K/W，"
              f"物理 tau={d14['protocol']['tau_spread_physical_s']*1e3:.4f} ms，"
              f"留一标定 tau_fixed={diag['tau_fixed_s']*1e3:.4f} ms，"
              f"逐周期标定漂移 {diag['tau_best_spread_factor']:.2f}×。", "",
              table(["period_s", "T_over_tau_spread", "fem_swing_K",
                     "raw_swing_err_pct", "static_swing_err_pct",
                     "dynamic_fixed_swing_err_pct",
                     "dynamic_perperiod_swing_err_pct",
                     "static_peak_err_K", "dynamic_fixed_peak_err_K"],
                    d14["model_comparison"]), "",
              "**逐周期标定的 tau_best**", "",
              table(["period_s", "tau_best_s", "tau_over_tau_phys",
                     "bracket_ok"], d14["per_period_fit"]), ""]
    fr = d14["validity_frontier"]
    lines += [f"1% 有效边界：静态 {fr['static_min_period_1pct_s']} s / "
              f"动态(单一 tau) {fr['dynamic_fixed_min_period_1pct_s']} s / "
              f"动态(逐周期最优) {fr['dynamic_perperiod_min_period_1pct_s']} s；"
              f"留一最大摆幅误差 {d14['leave_one_out']['max_abs_swing_err_pct']:.2f}%。", ""]

    # --- E15 ---
    reg = d15["radiation_regime"]
    lines += ["## E15 辐射与接触热阻", "",
              f"h_rad@T_amb(eps=0.9) = "
              f"{d15['protocol']['h_rad_at_ambient_eps09_W_m2K']:.3f} W/(m²K)；"
              f"强制对流 h={reg['h_conv_forced']:g} 辐射占比上限 "
              f"{100*reg['forced_share_max']:.4f}%，自然对流 h={reg['h_conv_natural']:g} "
              f"上限 {100*reg['natural_share_max']:.2f}%；"
              f"线性化最大结温误差 {reg['linearization_max_err_K']:.3f} K。", "",
              table(["h_conv_W_m2K", "epsilon", "scale", "power_W_per_m",
                     "t_ref_no_rad_C", "t_exact_newton_C",
                     "cooling_by_radiation_K", "linearization_err_K",
                     "radiation_share", "newton_iterations"],
                    [r for r in d15["radiation_sweep"]
                     if r["epsilon"] in (0.3, 0.9) and r["scale"] == 1.0]), "",
              table(["r_contact_m2K_W", "k_tim_eff_W_mK", "t_junction_fem_C",
                     "rise_vs_baseline_K"], d15["contact_sweep"]), "",
              table(["k_tim_W_mK", "t_junction_C"], d15["tim_sweep"]), ""]
    dc = d15["design_comparison"]
    # 由接触扫描的实测斜率反推：ΔT = slope * r_c''
    slope = next(r["rise_vs_baseline_K"] / r["r_contact_m2K_W"]
                 for r in d15["contact_sweep"] if r["r_contact_m2K_W"] > 0)
    critical_rc = dc["k_tim_5_to_50_cooling_K"] / slope
    # eps 最大、scale=1 的辐射占比做 log-log 插值，求 5% 显著性阈值
    import numpy as np

    eps_max = max(r["epsilon"] for r in d15["radiation_sweep"])
    pts = [(r["h_conv_W_m2K"], r["radiation_share"])
           for r in d15["radiation_sweep"]
           if r["epsilon"] == eps_max and r["scale"] == 1.0]
    lh = np.log([p[0] for p in pts])
    ls = np.log([p[1] for p in pts])
    a, b = np.polyfit(lh, ls, 1)
    h_at_5pct = float(np.exp((np.log(0.05) - b) / a))
    lines += [f"TIM 5→50 W/(mK) 降温 {dc['k_tim_5_to_50_cooling_K']:.3f} K；"
              f"最大接触热阻恶化 {dc['max_contact_penalty_K']:.3f} K "
              f"（{dc['contact_penalty_over_tim_gain']:.2f} 倍）。", "",
              f"线性关系 ΔT = {slope:.3e}·r_c'' 给出**临界接触热阻**："
              f"r_c'' = {critical_rc:.3e} m²K/W 即可把 TIM 本体从 5 做到 "
              f"50 W/(mK) 的全部收益抵消（{dc['k_tim_5_to_50_cooling_K']:.2f} K）。", "",
              f"辐射占比随 h_conv 的 log-log 斜率 {a:.3f}；5% 显著性阈值对应 "
              f"h_conv ≈ {h_at_5pct:.0f} W/(m²K)（ε={eps_max:g}，等温升 300 K 协议）。", ""]

    path = RES / "v12_summary_tables.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print("[13] 已写", path)
    print("[13] E13 稳态交叉验证差 K:", round(sc["abs_diff_K"], 5))
    print("[13] E14 tau 漂移倍数:", round(diag["tau_best_spread_factor"], 2))
    print("[13] E15 线性化最大误差 K:", round(reg["linearization_max_err_K"], 3))


if __name__ == "__main__":
    main()
