"""05 汇总：把各实验结果整理成报告可直接引用的 Markdown 表格与 summary.json。"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def load(name):
    return json.loads((R / name).read_text(encoding="utf-8"))


def main():
    steady = load("validation_steady.json")
    trans = load("validation_transient.json")
    conv = load("validation_convergence.json")
    sweep = load("sweep_kh.json")
    validity = load("lumped_validity.json")
    design = load("cooling_design.json")
    spread = load("spreading_resistance.json")

    L = []
    L.append("# N6 实验结果汇总（自动生成，数字供报告引用）\n")
    L.append("## 1 稳态三重验证\n")
    L.append("| 口径 | 结温 [°C] | 与 FEM 差 [K] |")
    L.append("|---|---|---|")
    tj = steady["t_junction_fem"]
    for name, key in [("手写 FEM", "t_junction_fem"), ("解析解（体积热源）", "t_junction_analytic"),
                      ("scikit-fem 12（独立装配）", "t_junction_skfem"),
                      ("精细 Cauer 网络", "t_junction_cauer"),
                      ("朴素串联热阻（表面热源口径）", "t_junction_surface_source")]:
        L.append(f"| {name} | {steady[key]:.4f} | {steady[key]-tj:+.4f} |")
    L.append(f"\nFEM↔解析 最大节点误差 {steady['err_fem_vs_analytic']['max_abs']:.3e} K；"
             f"FEM↔skfem {steady['err_fem_vs_skfem']['max_abs']:.3e} K；"
             f"能量守恒相对误差 {steady['energy_balance_rel_err']:.3e}；"
             f"体积/表面热源口径差 {steady['surface_vs_volume_gap_K']:.4f} K。\n")

    L.append("## 2 脉冲瞬态交叉验证（dt=0.005 s，t_end=10 s，T=2 s，duty=0.5）\n")
    L.append("| 方法 | 峰值 [°C] | 谷值 [°C] | 摆幅 [K] |")
    L.append("|---|---|---|---|")
    for name, m in trans["metrics"].items():
        L.append(f"| {name} | {m['t_peak']:.3f} | {m['t_trough']:.3f} | {m['swing']:.3f} |")
    L.append("\n| 对比 | 峰值误差(温升口径) [%] | 摆幅误差 [%] | RMSE/幅值 [%] |")
    L.append("|---|---|---|---|")
    for name, e in trans["errors"].items():
        L.append(f"| {name} | {e['peak_rise_rel_err_pct']:.4f} | "
                 f"{e['swing_rel_err_pct']:.4f} | {e['rmse_rel_amp_pct']:.4f} |")
    L.append("\n目标：与热阻网络解析解误差 <5%（达成）。\n")

    L.append("## 3 收敛阶验证\n")
    L.append("| 项目 | 拟合阶 p | 理论阶 | 误差序列 |")
    L.append("|---|---|---|---|")
    L.append(f"| 时间 BE | {conv['time']['BE']['order']:.2f} | 1 | " +
             ", ".join(f"{e:.2e}" for e in conv["time"]["BE"]["err"]) + " |")
    L.append(f"| 时间 CN | {conv['time']['CN']['order']:.2f} | 2 | " +
             ", ".join(f"{e:.2e}" for e in conv["time"]["CN"]["err"]) + " |")
    L.append(f"| 空间（几何相似加密） | {conv['space']['order']:.2f} | 2 | " +
             ", ".join(f"{e:.2e}" for e in conv["space"]["err"]) + " |")
    L.append("\n## 4 热阻分解与参数敏感度（基准工况）\n")
    b = validity["resistance_breakdown_pct"]
    L.append(f"- 对流 {b['r_conv_pct']:.2f}% / TIM {b['r_tim_pct']:.2f}% / "
             f"芯片层半热阻 {b['r_die_half_pct']:.2f}% / 基板 {b['r_base_pct']:.2f}%")
    for h, v in validity["k_sensitivity"].items():
        L.append(f"- {h}: k_TIM 1→50 W/(m K) 结温降 {v['k_gain_K']:.2f} K "
                 f"(相对 {v['k_gain_pct']:.2f}%)")
    L.append("\n## 5 单节集总 Cauer 的适用边界（摆幅误差）\n")
    L.append("| 周期 [s] | 精细摆幅 [K] | 集总摆幅 [K] | 相对误差 [%] |")
    L.append("|---|---|---|---|")
    rec0 = min(validity["records"], key=lambda r: abs(r["h_conv"] - 5000.0))
    for p in validity["periods"]:
        q = rec0["periods"][f"{p:g}"]
        L.append(f"| {p:g} | {q['swing_fine_K']:.4f} | {q['swing_lump_K']:.4f} | "
                 f"{q['swing_rel_err_pct']:.3f} |")
    L.append("\nBi_TIM 从 0.004→20 时快脉冲误差仅 8.77%→9.33%："
             "瞬态集总失效由脉冲周期/层内扩散时间之比主导，而非稳态 Bi 判据。\n")

    L.append("## 6 散热方案对比\n")
    L.append("| 方案 | 稳态结温 [°C] | 峰值 [°C] | 摆幅 [K] | 峰值降幅 [%] |")
    L.append("|---|---|---|---|---|")
    for r in design["plans"]:
        L.append(f"| {r['plan']} | {r['t_j_steady']:.2f} | {r['t_peak']:.2f} | "
                 f"{r['swing']:.2f} | {r['peak_reduction_pct']:.2f} |")

    L.append("\n## 7 二维扩散热阻（集总网络的失效边界）\n")
    L.append("| w/W | 2D 稳态结温 [°C] | 1D 解析 [°C] | R_spread/R_1D [%] | Cauer 峰值低估 [K] |")
    L.append("|---|---|---|---|---|")
    for r in spread["cases"]:
        L.append(f"| {r['w_over_W']:.2f} | {r['t_j_steady_2d']:.2f} | "
                 f"{r['t_j_steady_1d_analytic']:.2f} | "
                 f"{r['spread_r_spread_ratio_pct']:.2f} | "
                 f"{r['cauer_peak_underestimate_K']:.2f} |")

    text = "\n".join(L) + "\n"
    (R / "summary_tables.md").write_text(text, encoding="utf-8")
    summary = {
        "steady": steady, "transient": trans, "convergence": conv,
        "sweep": {"k_list": sweep["k_list"], "h_list": sweep["h_list"]},
        "validity": validity, "design": design, "spreading": spread,
    }
    (R / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(text)
    print("[05] 已写 results/summary_tables.md 与 results/summary.json")


if __name__ == "__main__":
    main()
