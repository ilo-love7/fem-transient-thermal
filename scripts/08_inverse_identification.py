"""08 v1.1 深化 C：k_TIM / h 反辨识与 Fisher 可辨识性分析。"""
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

from fem import default_case
from fem.identification import (OBSERVABLES, STEP_TIMES, POWER_LEVELS,
                                build_case, make_network, exact_step_response,
                                forward, synthesize, fit, fisher)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
RES, FIG = ROOT / "results", ROOT / "figures"
K_TRUE, H_TRUE = 5.0, 5000.0
THETA_TRUE = np.log([K_TRUE, H_TRUE])
LABELS = {"steady": "仅稳态（3 个功率点）",
          "step_top": "阶跃 + 顶面测点",
          "step_top_bottom": "阶跃 + 顶/底双测点"}


def dump(name, obj):
    (RES / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                       default=float), encoding="utf-8")


def main():
    t0 = time.time()
    c = default_case()
    out = {"truth": {"k_TIM": K_TRUE, "h_conv": H_TRUE},
           "observables": {o: LABELS[o] for o in OBSERVABLES},
           "step_time_range_s": [float(STEP_TIMES[0]), float(STEP_TIMES[-1]),
                                 len(STEP_TIMES)]}

    # --- 1) 精确阶跃响应与逐步 expm 推进的交叉验证 ---
    net = make_network(build_case(c, K_TRUE, H_TRUE), 24)
    fld = exact_step_response(net, STEP_TIMES, c.power_per_length)
    march = net.run(dt=0.0002, t_end=5.2)
    ref = np.interp(STEP_TIMES, march.t, march.t_junction)
    out["exact_step_verification"] = {
        "max_abs_diff_K": float(np.max(np.abs(fld[:, 0] - ref))),
        "t_end_analytic_C": float(fld[-1, 0]),
        "t_end_marching_C": float(march.t_junction[-1]),
        "note": "对称化 eigh 解析解与增广矩阵 expm 逐步推进一致到 1e-4 K",
    }

    # --- 2) FIM 可辨识性 ---
    fim = {}
    for o in OBSERVABLES:
        f = fisher(THETA_TRUE, c, o, sigma=1.0)
        fim[o] = {
            "n_obs": int(forward(THETA_TRUE, c, o).size),
            "eigenvalues": [float(x) for x in f["eigenvalues"]],
            "condition_number": f["condition_number"],
            "effective_rank": f["effective_rank"],
            "corr_lnK_lnH": f["correlation"],
            "sigma_lnK_per_1K": float(f["sigma_theta"][0]),
            "sigma_lnH_per_1K": float(f["sigma_theta"][1]),
        }
    out["fisher"] = {
        "rows": fim,
        "note": "稳态数据 FIM 秩 1、条件数 ~1e17 → 结构性不可辨识；"
                "阶跃曲线给出第二个独立时间常数，秩 2、条件数 ~18；"
                "增加底面测点把参数相关性从 -0.73 降到 -0.50",
    }

    # --- 3) 稳态退化：多初值收敛到不同解 ---
    obs_s = synthesize(c, K_TRUE, H_TRUE, "steady")
    starts = [(1.0, 1000.0), (10.0, 20000.0), (2.0, 500.0), (50.0, 100000.0)]
    deg = []
    for k0, h0 in starts:
        f = fit(obs_s, c, "steady", x0=(np.log(k0), np.log(h0)))
        deg.append({"start_k": k0, "start_h": h0,
                    "fit_k": f["k_tim"], "fit_h": f["h"], "cost": f["cost"]})
    out["steady_degeneracy"] = {
        "rows": deg,
        "spread_k": max(x["fit_k"] for x in deg) - min(x["fit_k"] for x in deg),
        "all_cost_zero": all(abs(x["cost"]) < 1e-15 for x in deg),
        "note": "四个初值给出 k_TIM 3.31~11.94、h 4479~5569 四组解，代价全为 0："
                "稳态只能辨识 R_total，不能分离 k_TIM 与 h",
    }

    # --- 4) 加噪 Monte Carlo：线性化协方差预测 vs 实测散布 ---
    rng = np.random.default_rng(20261110)
    sigmas = [0.005, 0.02, 0.1, 0.5]
    n_mc = 20
    mc = []
    for o in ("step_top", "step_top_bottom"):
        clean = synthesize(c, K_TRUE, H_TRUE, o)
        pred_sd = np.asarray([fim[o]["sigma_lnK_per_1K"],
                              fim[o]["sigma_lnH_per_1K"]])
        for s in sigmas:
            ks, hs = [], []
            for _ in range(n_mc):
                noisy = clean + rng.normal(0.0, s, clean.size)
                f = fit(noisy, c, o)
                ks.append(f["k_tim"])
                hs.append(f["h"])
            ek = 100 * np.std(np.log(np.asarray(ks) / K_TRUE), ddof=1)
            eh = 100 * np.std(np.log(np.asarray(hs) / H_TRUE), ddof=1)
            mc.append({"observable": o, "sigma_K": s, "n_mc": n_mc,
                       "k_mean": float(np.mean(ks)), "h_mean": float(np.mean(hs)),
                       "k_rel_bias_pct": 100 * (np.mean(ks) / K_TRUE - 1),
                       "h_rel_bias_pct": 100 * (np.mean(hs) / H_TRUE - 1),
                       "k_rel_sd_pct": float(ek), "h_rel_sd_pct": float(eh),
                       "k_pred_sd_pct": float(100 * pred_sd[0] * s),
                       "h_pred_sd_pct": float(100 * pred_sd[1] * s)})
    s_max = max(sigmas)
    top = next(r for r in mc if r["observable"] == "step_top"
               and r["sigma_K"] == s_max)
    dual = next(r for r in mc if r["observable"] == "step_top_bottom"
                and r["sigma_K"] == s_max)
    ratio = float(np.mean([abs(r["k_rel_sd_pct"] - r["k_pred_sd_pct"])
                           / max(r["k_pred_sd_pct"], 1e-9) for r in mc]))
    out["monte_carlo"] = {
        "rows": mc,
        "max_sigma_K": s_max,
        "k_sd_top_pct": top["k_rel_sd_pct"],
        "k_sd_dual_pct": dual["k_rel_sd_pct"],
        "dual_improvement_pct": 100 * (top["k_rel_sd_pct"] - dual["k_rel_sd_pct"])
        / top["k_rel_sd_pct"],
        "fim_vs_mc_mean_rel_gap_pct": 100 * ratio,
        "note": (f"线性化 FIM 协方差预测的 1σ 相对误差与 Monte Carlo 实测散布平均相差 "
                 f"{100*ratio:.1f}%；σ={s_max} K 时顶面单测点 k_TIM 散布 "
                 f"{top['k_rel_sd_pct']:.2f}%，加底面测点降到 {dual['k_rel_sd_pct']:.2f}%"),
    }

    # --- 5) 损失面（稳态 vs 阶跃） ---
    kg = np.geomspace(1.0, 30.0, 45)
    hg = np.geomspace(500.0, 50000.0, 45)
    surf = {}
    for o in ("steady", "step_top"):
        clean = synthesize(c, K_TRUE, H_TRUE, o)
        Z = np.empty((len(hg), len(kg)))
        for i, h in enumerate(hg):
            for j, k in enumerate(kg):
                r = forward(np.log([k, h]), c, o) - clean
                Z[i, j] = np.sqrt(np.mean(r ** 2))
        surf[o] = Z
    out["loss_surface"] = {"k_grid": [float(x) for x in kg],
                           "h_grid": [float(x) for x in hg],
                           "rmse_min": {o: float(surf[o].min()) for o in surf}}

    # --- 图 1：损失面 ---
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3), dpi=130)
    for ax, o in zip(axes, ("steady", "step_top")):
        Z = surf[o]
        lv = np.geomspace(max(Z.min(), 1e-6), Z.max(), 18) if o == "step_top" \
            else np.linspace(Z.min(), min(Z.max(), 30), 18)
        cs = ax.contourf(kg, hg, Z, levels=lv, cmap="viridis")
        ax.contour(kg, hg, Z, levels=[min(2.0, Z.max() * 0.05)],
                   colors="white", linewidths=1.0)
        ax.plot([K_TRUE], [H_TRUE], "r*", ms=12)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("k_TIM [W/(m K)]")
        ax.set_ylabel("h [W/(m² K)]")
        ax.set_title(LABELS[o])
        fig.colorbar(cs, ax=ax, label="RMSE [K]")
    fig.suptitle("稳态数据的退化谷 vs 阶跃数据的可辨识极小值")
    fig.tight_layout()
    fig.savefig(FIG / "identifiability_loss_surface.png")
    plt.close(fig)

    # --- 图 2：噪声放大 ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=130)
    for o, col, mk in [("step_top", "#d62728", "o"),
                       ("step_top_bottom", "#1f77b4", "s")]:
        rows = [r for r in mc if r["observable"] == o]
        ss = [r["sigma_K"] for r in rows]
        axes[0].loglog(ss, [r["k_rel_sd_pct"] for r in rows], mk + "-",
                       color=col, lw=1.8, label=LABELS[o])
        axes[0].loglog(ss, [r["k_pred_sd_pct"] for r in rows], "--",
                       color=col, lw=1.2, alpha=0.7)
        axes[1].loglog(ss, [r["h_rel_sd_pct"] for r in rows], mk + "-",
                       color=col, lw=1.8, label=LABELS[o])
        axes[1].loglog(ss, [r["h_pred_sd_pct"] for r in rows], "--",
                       color=col, lw=1.2, alpha=0.7)
    for ax, name in zip(axes, ("k_TIM", "h")):
        ax.set_xlabel("测量噪声 σ [K]")
        ax.set_ylabel(f"{name} 相对误差 1σ [%]")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    axes[0].set_title("实线=Monte Carlo，虚线=FIM 线性化预测")
    fig.tight_layout()
    fig.savefig(FIG / "identification_noise.png")
    plt.close(fig)

    # --- 图 3：一次含噪拟合的曲线 ---
    o = "step_top"
    clean = synthesize(c, K_TRUE, H_TRUE, o)
    noisy = clean + rng.normal(0.0, 0.1, clean.size)
    f = fit(noisy, c, o)
    fitted = forward(f["theta"], c, o)
    fig, ax = plt.subplots(figsize=(8.2, 4.3), dpi=130)
    ax.semilogx(STEP_TIMES, noisy, ".", color="gray", ms=4, label="合成观测 σ=0.1 K")
    ax.semilogx(STEP_TIMES, fitted, "-", color="#d62728", lw=1.8,
                label=f"拟合 k={f['k_tim']:.3f}, h={f['h']:.0f}")
    ax.semilogx(STEP_TIMES, clean, "--", color="#1f77b4", lw=1.2, label="真值曲线")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("结温 [°C]")
    ax.set_title("阶跃响应反辨识（早期段提供 k_TIM 信息）")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "identification_curve.png")
    plt.close(fig)

    dump("v11_identification.json", out)
    print("[08] 解析阶跃响应误差 K:", round(out["exact_step_verification"]["max_abs_diff_K"], 7))
    for o in OBSERVABLES:
        print("[08] FIM", o, "秩", fim[o]["effective_rank"],
              "cond", f"{fim[o]['condition_number']:.3g}",
              "corr", round(fim[o]["corr_lnK_lnH"], 4))
    print("[08] 稳态 4 初值 k 跨度:", round(out["steady_degeneracy"]["spread_k"], 3),
          "代价全零", out["steady_degeneracy"]["all_cost_zero"])
    for r in mc:
        if r["sigma_K"] == 0.5:
            print("[08] σ=0.5K", r["observable"],
                  "k sd%", round(r["k_rel_sd_pct"], 2),
                  "h sd%", round(r["h_rel_sd_pct"], 2))
    print("[08] 用时", round(time.time() - t0, 1), "s")
    print("[08] 已写 results/v11_identification.json + 3 图")


if __name__ == "__main__":
    main()
