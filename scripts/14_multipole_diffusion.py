"""14 v1.3 深化 E16：多极点分布参数扩散热阻（破解单极点证伪）。

回应 §12.3 最高优先级未做项「多极点 / 分布参数扩散模型」。v1.2 E14 证伪
单极点：逐周期 tau_best 漂移 17×，留一摆幅误差 47.78%。本脚本：

  1) 跑**一次**轴对称 FEM 阶跃响应（恒定功率，源比 0.2，与 v1.1/v1.2 同口径），
     用等效一维 Cauer 网络的阶跃响应相减，得到扩散超温 ΔT_sp(t)；
  2) 在 tau 对数网格上用非负最小二乘拟合 Foster 链（1/2/3/4/6/8 极点）；
  3) 用同一组固定极点**零样本**预测 v1.1 的 6 个周期方波摆幅（无需逐周期标定），
     对照单极点（v1.2 tau_fixed）与 FEM 真值；
  4) 极点稳定性：用三个截断时间窗分别拟合，统计主极点 tau 的漂移倍数。

预设：多极点应把留一最大摆幅误差从 ~48% 显著压低；若主极点随窗剧烈漂移
则说明有限极点只是数值 artifact（分布参数需连续谱）——两种结论都如实报告。
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fem import case_from_config, default_config_path, load_config  # noqa: E402
from fem.axisym import (assemble_axisym, axisym_from_case,  # noqa: E402
                        run_axisym_transient)
from fem.cauer import CauerNetwork  # noqa: E402
from fem.multipole import (fit_foster, log_tau_grid, pole_stability,  # noqa: E402
                           single_pole_from_step)
from fem.spreading import equivalent_ladder_case  # noqa: E402
from fem.two_stage import build_two_stage  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
RES = ROOT / "results"
FIG = ROOT / "figures"


def dump(name, obj):
    (RES / name).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")


def main():
    t0 = time.time()
    cfg = load_config(default_config_path())
    v = cfg["v12"]["dynamic_diffusion"]
    base = case_from_config(cfg)
    n = int(v["n_cauer"])
    q = float(v["source_ratio"])
    ref = json.loads((ROOT / str(v["v11_reference"])).read_text(encoding="utf-8"))
    ps = ref["period_sweep"]
    r_sp = float(ref["transient_correction"]["r_sp_K_W"])

    ax = axisym_from_case(base, q, nr=int(v["nr"]),
                          nz_per_layer=int(v["nz_per_layer"]))
    P = ax.total_power
    dt = 0.0005
    t_end = 0.8  # 覆盖最慢扩散极点（tau_sp~5.6ms）的 100× 以上

    # --- 1) 轴对称 FEM 阶跃（恒定功率），及等效一维 Cauer 阶跃 ---
    step = run_axisym_transient(assemble_axisym(ax), dt=dt, t_end=t_end,
                                power_fn=lambda t: 1.0)
    ladder = equivalent_ladder_case(ax)
    net1d = CauerNetwork(ladder, tuple([n] * ladder.n_layers))
    one = net1d.run(dt=dt, t_end=t_end, power_fn=lambda t: P)
    tj_ax = step.t_junction
    tj_1d = one.t_junction
    delta_t = tj_ax - tj_1d  # 扩散超温 [K]
    z = delta_t / P          # 扩散热阻抗 [K/W]

    # 只保留上升后段（去掉最初几个 dt 的数值瞬态），并限制在 z>0
    mask = (step.t >= 2 * dt) & (z > 0)
    tt = step.t[mask]
    zz = z[mask]
    z_inf = float(np.median(zz[-50:]))  # 稳态扩散热阻（≈r_sp）

    out = {"protocol": {
        "source_ratio": q, "n_cauer_per_layer": n,
        "fem_nr": ax.nr, "fem_nz_per_layer": ax.nz_per_layer,
        "step_dt_s": dt, "step_t_end_s": t_end,
        "total_power_W": P, "r_sp_reference_K_W": r_sp,
        "z_inf_from_step_K_W": z_inf,
        "tau_spread_physical_s": float(ps["tau_spread_s"]),
        "n_periods_compared": len(ps["rows"]),
        "method": "轴对称FEM阶跃 - 等效1D Cauer阶跃 = 扩散超温；NNLS拟合Foster链",
    }}

    # --- 2) 多极点拟合（1/2/3/4/6/8） ---
    grid = log_tau_grid(1e-5, 2.0, 40)
    fits = {}
    for k in (1, 2, 3, 4, 6, 8):
        if k == 1:
            f = single_pole_from_step(tt, zz, tau_grid=grid)
        else:
            # 从 40 点网格中留 k 个最大权重的 tau（交替：贪心选择）
            f0 = fit_foster(tt, zz, grid, enforce_r_total=z_inf)
            order = np.argsort(-f0["r"])
            chosen = np.sort(grid[order[:k]])
            f = fit_foster(tt, zz, chosen, enforce_r_total=z_inf)
        fits[k] = f

    fit_rows = []
    for k, f in fits.items():
        net = f["network"]
        fit_rows.append({
            "n_poles": k, "r_total_K_W": net.r_total,
            "r_i": np.asarray(net.resistances).tolist(),
            "tau_i_s": np.asarray(net.taus).tolist(),
            "rmse_K_per_W": f["rmse_K_per_W"],
            "rel_rmse": f["rel_rmse"],
            "active_poles": f.get("active_poles", 1),
        })
    out["foster_fits"] = fit_rows
    out["step_fit_best_rel_rmse"] = min(r["rel_rmse"] for r in fit_rows)

    # --- 3) 跨周期摆幅预测（固定极点，零样本） ---
    comparison = []
    for row in ps["rows"]:
        per = float(row["period_s"])
        dt_p, te_p = min(0.005, per / 40.0), max(6 * per, 0.4)
        c = replace(ax, pulse_period=per, dt=dt_p, t_end=te_p)
        ladder_c = equivalent_ladder_case(c)
        raw = CauerNetwork(ladder_c, tuple([n] * ladder_c.n_layers)).run(
            dt=dt_p, t_end=te_p)
        # 1D Cauer 摆幅（差分：总摆幅 - 扩散摆幅 = 基础网络摆幅）
        base_swing = (raw.t_junction[raw.t >= raw.t[-1] - per].max()
                      - raw.t_junction[raw.t >= raw.t[-1] - per].min())
        fem_swing = float(row["fem_swing_K"])
        entry = {"period_s": per, "T_over_tau_spread": float(row["T_over_tau_spread"]),
                 "fem_swing_K": fem_swing, "base_1d_swing_K": float(base_swing)}
        for k, f in fits.items():
            sw = f["network"].periodic_swing(per, c.duty,
                                             dt=dt_p, t_end=te_p)["swing"]
            total = base_swing + sw * P  # 归一化到 P=1 的 Foster 摆幅 × 实际功率
            entry[f"foster{k}_swing_K"] = float(total)
            entry[f"foster{k}_err_pct"] = 100.0 * (total - fem_swing) / fem_swing
        # v1.2 单极点 tau_fixed 对照
        tau_fixed = float(ref["transient_correction"].get("tau_fixed_s",
                                                          0.001))
        c2 = replace(ax, pulse_period=per, dt=dt_p, t_end=te_p)
        dyn = build_two_stage(c2, r_sp, tau_fixed, n=n).run(c2, dt=dt_p, t_end=te_p)
        v12_sw = (dyn.t_junction[dyn.t >= dyn.t[-1] - per].max()
                  - dyn.t_junction[dyn.t >= dyn.t[-1] - per].min())
        entry["v12_single_pole_swing_K"] = float(v12_sw)
        entry["v12_single_pole_err_pct"] = 100.0 * (v12_sw - fem_swing) / fem_swing
        comparison.append(entry)
    out["cross_period_comparison"] = comparison

    summary = {}
    for k in fits:
        errs = [abs(r[f"foster{k}_err_pct"]) for r in comparison]
        summary[f"foster{k}"] = {
            "max_abs_swing_err_pct": float(max(errs)),
            "rmse_swing_err_pct": float(np.sqrt(np.mean(np.square(errs)))),
        }
    v12_errs = [abs(r["v12_single_pole_err_pct"]) for r in comparison]
    summary["v12_single_pole"] = {
        "max_abs_swing_err_pct": float(max(v12_errs)),
        "rmse_swing_err_pct": float(np.sqrt(np.mean(np.square(v12_errs)))),
    }
    out["cross_period_summary"] = summary
    best_k = min((k for k in fits),
                 key=lambda k: summary[f"foster{k}"]["rmse_swing_err_pct"])
    out["best_model"] = {
        "n_poles": best_k,
        "improvement_factor_vs_v12": (
            summary["v12_single_pole"]["rmse_swing_err_pct"]
            / max(summary[f"foster{best_k}"]["rmse_swing_err_pct"], 1e-12)),
    }

    # --- 4) 极点稳定性（三个截断窗） ---
    stability_fits = []
    for tw in (0.05, 0.2, t_end):
        m = tt <= tw
        if m.sum() < 10:
            continue
        f = fit_foster(tt[m], zz[m], grid, enforce_r_total=z_inf)
        stability_fits.append(f)
    out["pole_stability"] = pole_stability(stability_fits)

    # --- 图 ---
    xx = [r["T_over_tau_spread"] for r in comparison]
    fig, axp = plt.subplots(figsize=(9, 4.8), dpi=130)
    keys = [("v12_single_pole_err_pct", "v1.2 单极点 (tau_fixed)", "#d62728", "--")]
    for k in (2, 4, 8):
        keys.append((f"foster{k}_err_pct", f"Foster {k} 极点（阶跃标定）",
                     {2: "#1f77b4", 4: "#2ca02c", 8: "#9467bd"}[k], "-"))
    for key, lab, col, ls in keys:
        axp.plot(xx, [abs(r[key]) for r in comparison], ls, color=col,
                 lw=1.8, marker="o", ms=4, label=lab)
    axp.axhline(1.0, color="gray", ls=":", lw=1, label="1% 工程阈值")
    axp.set_xscale("log")
    axp.set_yscale("log")
    axp.set_xlabel(r"$T/\tau_{spread}$")
    axp.set_ylabel("摆幅误差 |Δ| [%]")
    axp.set_title("E16 多极点扩散网络：阶跃标定后跨周期零样本预测")
    axp.grid(alpha=0.3, which="both")
    axp.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "v13_foster_cross_period.png")
    plt.close(fig)

    fig, axp = plt.subplots(figsize=(9, 4.8), dpi=130)
    axp.plot(tt, zz, color="#111111", lw=1.8, label="FEM 扩散热阻抗 Z(t)")
    for k in (2, 4, 8):
        axp.plot(tt, fits[k]["network"].step(tt), lw=1.4,
                 label=f"Foster {k} 极点")
    axp.axhline(r_sp, color="gray", ls="--", lw=1, label=f"R_sp={r_sp:.3f} K/W")
    axp.set_xscale("log")
    axp.set_xlabel("t [s]")
    axp.set_ylabel("Z_sp(t) [K/W]")
    axp.set_title("阶跃热阻抗与 Foster 极点拟合")
    axp.grid(alpha=0.3, which="both")
    axp.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "v13_foster_step_fit.png")
    plt.close(fig)

    out["wall_s"] = time.time() - t0
    dump("v13_multipole.json", out)
    print("[14] z_inf =", round(z_inf, 5), "K/W; R_sp ref =", round(r_sp, 5))
    print("[14] best Foster poles =", best_k,
          "rmse% =", round(summary[f"foster{best_k}"]["rmse_swing_err_pct"], 3),
          "vs v1.2", round(summary["v12_single_pole"]["rmse_swing_err_pct"], 3))
    print("[14] improvement factor =",
          round(out["best_model"]["improvement_factor_vs_v12"], 2))
    print("[14] pole stability =", out["pole_stability"]["stable"],
          "spread =", round(out["pole_stability"]["spread_factor"], 2))
    print("[14] 用时", round(out["wall_s"], 1), "s；已写 JSON + 2 图")


if __name__ == "__main__":
    main()
