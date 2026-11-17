"""17 v1.3 汇总：把 E16/E17/E18 三个 JSON 汇成可复核的 Markdown 表。"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def load(name):
    p = RES / name
    if not p.exists():
        raise SystemExit(f"缺少 {p}，请先运行 scripts/14-16")
    return json.loads(p.read_text(encoding="utf-8"))


def fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        if v == 0:
            return "0"
        if abs(v) >= 1e5 or abs(v) < 1e-4:
            return f"{v:.3g}"
        return f"{v:.{nd}f}"
    if isinstance(v, bool):
        return "✅" if v else "❌"
    return str(v)


def table(headers, rows, keys=None, nd=4):
    keys = keys or headers
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(fmt(r.get(k), nd) for k in keys) + " |")
    return "\n".join(out)


def main():
    mp = load("v13_multipole.json")
    mc = load("v13_multicondition.json")
    ad = load("v13_adaptive.json")
    L = ["# N6 v1.3 深化实验汇总表", "",
         "> 由 `scripts/17_summary_v13.py` 自动生成；完整推导见 "
         "`reports/基准评测报告-N6.md` §10–§13。所有数字来自 `results/v13_*.json`。", ""]

    # ---------- E16 ----------
    L += ["## 表 A · E16 Foster 阶跃拟合", ""]
    L += [table(["极点数", "R_total [K/W]", "RMSE [K/W]", "相对RMSE",
                 "激活极点数"],
                mp["foster_fits"],
                keys=["n_poles", "r_total_K_W", "rmse_K_per_W",
                      "rel_rmse", "active_poles"]), ""]
    L += [f"阶跃真值：稳态扩散热阻 z_inf = "
          f"{mp['protocol']['z_inf_from_step_K_W']:.5f} K/W；"
          f"R_sp 参考 = {mp['protocol']['r_sp_reference_K_W']:.5f} K/W "
          f"(差 {abs(mp['protocol']['z_inf_from_step_K_W']-mp['protocol']['r_sp_reference_K_W'])/mp['protocol']['r_sp_reference_K_W']*100:.3f}%)。", ""]

    L += ["## 表 B · E16 跨周期摆幅误差（阶跃标定 → 零样本预测）", ""]
    cross = []
    for r in mp["cross_period_comparison"]:
        cross.append({
            "T [s]": r["period_s"],
            "T/tau": r["T_over_tau_spread"],
            "FEM摆幅 [K]": r["fem_swing_K"],
            "v1.2单极点 err%": r["v12_single_pole_err_pct"],
            "Foster2 err%": r["foster2_err_pct"],
            "Foster4 err%": r["foster4_err_pct"],
            "Foster8 err%": r["foster8_err_pct"],
        })
    L += [table(list(cross[0].keys()), cross), ""]
    sm = mp["cross_period_summary"]
    summary_rows = [{"模型": k, "最大绝对误差%": v["max_abs_swing_err_pct"],
                     "RMSE%": v["rmse_swing_err_pct"]}
                    for k, v in sm.items()]
    L += [table(["模型", "最大绝对误差%", "RMSE%"], summary_rows), ""]
    bm = mp["best_model"]
    ps = mp["pole_stability"]
    best_rmse = sm["foster" + str(bm["n_poles"])]["rmse_swing_err_pct"]
    L += [f"**裁决**：最佳 {bm['n_poles']} 极点，RMSE "
          f"{best_rmse:.3f}% vs "
          f"v1.2 单极点 {sm['v12_single_pole']['rmse_swing_err_pct']:.3f}% "
          f"（改善 {bm['improvement_factor_vs_v12']:.1f}×）；"
          f"主极点跨窗稳定={fmt(ps['stable'])}，spread={fmt(ps['spread_factor'],2)}。", ""]

    # ---------- E17 ----------
    L += ["## 表 C · E17 多工况联合 FIM", ""]
    fim_rows = []
    for r in mc["fim_suites"]:
        ev = r.get("eigenvalues") or [0, 0]
        fim_rows.append({"suite": r["suite"],
                         "n_conditions": r["n_conditions"],
                         "n_observations": r["n_observations"],
                         "effective_rank": r["effective_rank"],
                         "condition_number": r["condition_number"],
                         "lambda_min": ev[0],
                         "corr": r["correlation"]})
    L += [table(["工况套件", "工况数", "观测数", "有效秩", "条件数",
                 "最小特征值", "参数相关"],
                fim_rows,
                keys=["suite", "n_conditions", "n_observations",
                      "effective_rank", "condition_number", "lambda_min",
                      "corr"],
                nd=3), ""]
    verd = mc["verdict"]
    L += [f"**裁决**：稳态多功率秩1确认={fmt(verd['steady_rank_one_confirmed'])}；"
          f"最佳套件 `{verd['best_suite']}` 条件数 {fmt(verd['best_condition_number'],2)}，"
          f"秩破因子 {verd['rank_break_factor']:.3g}；"
          f"相关系数 {verd['correlation_steady']:.4f} → "
          f"{verd['correlation_best']:.4f}。", ""]
    L += ["## 表 D · E17 蒙特卡洛反辨识散布", ""]
    mcv = mc["monte_carlo"]
    L += [table(["套件", "k_TIM中位", "k_TIM IQR", "h中位", "h IQR",
                 "k相对IQR", "h相对IQR"],
                [mcv["steady"], mcv["best"]],
                keys=["suite", "k_median", "k_iqr", "h_median", "h_iqr",
                      "k_rel_iqr", "h_rel_iqr"], nd=3), ""]

    # ---------- E18 ----------
    L += ["## 表 E · E18 自适应网格扫描", ""]
    L += [table(["a/R", "加密比例", "自适应nx", "粗网格err", "自适应err",
                 "同自由度均匀err", "增益×", "指示子max/min"],
                ad["scan"],
                keys=["source_ratio", "refine_fraction", "adaptive_nx",
                      "rel_err_coarse", "rel_err_adaptive",
                      "rel_err_uniform_same_dof", "speedup_over_uniform",
                      "interval_eta_max_over_min"], nd=4), ""]
    av = ad["verdict"]
    best = av["best"]
    L += [f"**裁决**：局部热源胜 {av['local_source_adaptive_wins_count']}"
          f"/{av['local_source_total']}；最佳 a/R={best['source_ratio']:g}、"
          f"frac={best['refine_fraction']:g} 增益 "
          f"{best['speedup_over_uniform']:.1f}×（自适应 err "
          f"{best['rel_err_adaptive']:.3e} vs 均匀 "
          f"{best['rel_err_uniform']:.3e}）；"
          f"全宽热源 max err {av['full_width_max_rel_err']:.3e}，"
          f"零收益={fmt(av['full_width_zero_gain'])}。", ""]
    L += ["**边界说明**：frac=0.25 时自适应反而劣化 100×——checkerboard 剖分"
          "让能量指示子呈奇偶交替，低比例整列加密形成"
          "「细-粗-细-粗」齿状网格，破坏精度；只有 frac≥0.5 且热源够窄才稳定获胜。", ""]

    out = RES / "v13_summary_tables.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"→ {out}  ({len(L)} 行)")


if __name__ == "__main__":
    main()
