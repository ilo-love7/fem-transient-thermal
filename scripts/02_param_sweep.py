"""02 参数扫描：导热系数 k_TIM / 对流系数 h 的热响应面，以及集总 Cauer 适用边界。"""
from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from fem import (load_config, case_from_config, default_config_path,
                 build_mesh, assemble, solve_steady, run_transient,
                 periodic_steady_metrics, fine_network, lumped_network,
                 curve_error_full, bi_numbers)
from fem.plots import plot_heatmap, plot_error_vs_h


def dump(path, obj):
    (ROOT / "results" / path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=float), encoding="utf-8")


def with_tim_k(case, k):
    layers = list(case.layers)
    layers[1] = replace(layers[1], k=float(k))
    return case.with_params(layers=tuple(layers))


def main():
    t0 = time.time()
    cfg = load_config(default_config_path())
    base = case_from_config(cfg)
    k_list = [float(x) for x in cfg["sweep"]["k_tim_list"]]
    h_list = [float(x) for x in cfg["sweep"]["h_list"]]
    fig = ROOT / "figures"

    # ---------- A. k_TIM x h 响应面（FEM 稳态 + FEM 脉冲瞬态） ----------
    Tj = np.zeros((len(k_list), len(h_list)))
    Swing = np.zeros_like(Tj)
    Peak = np.zeros_like(Tj)
    rows = []
    for i, k in enumerate(k_list):
        for j, h in enumerate(h_list):
            c = with_tim_k(base, k).with_params(h_conv=h)
            m = build_mesh(c)
            s = assemble(c, m)
            Ts = solve_steady(s, c.power_per_length)
            r = run_transient(s, c, dt=0.01, t_end=8.0)
            met = periodic_steady_metrics(r, c)
            Tj[i, j] = Ts[m.junction_node]
            Swing[i, j] = met["swing"]
            Peak[i, j] = met["t_peak"]
            rows.append(dict(k_tim=k, h_conv=h, t_j_steady=float(Tj[i, j]),
                             t_peak=float(Peak[i, j]), swing=float(Swing[i, j]),
                             r_cond=float(c.r_cond_series), r_conv=float(c.r_conv),
                             r_eff=float(c.r_effective),
                             analytic_tj=float(c.t_junction_steady)))
        print(f"[扫描] k_TIM={k:g} 完成 ({i+1}/{len(k_list)})", flush=True)

    with (ROOT / "results" / "sweep_kh.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    dump("sweep_kh.json", {"k_list": k_list, "h_list": h_list,
                           "tj_steady": Tj.tolist(), "swing": Swing.tolist(),
                           "t_peak": Peak.tolist()})
    plot_heatmap(np.array(h_list), np.array(k_list), Tj,
                 "h [W/(m² K)]", "k_TIM [W/(m K)]",
                 "稳态结温响应面（FEM）", fig / "heatmap_tj_steady.png")
    plot_heatmap(np.array(h_list), np.array(k_list), Swing,
                 "h [W/(m² K)]", "k_TIM [W/(m K)]",
                 "周期稳态结温摆幅响应面 [K]", fig / "heatmap_swing.png",
                 zlabel="ΔT_swing [K]", cmap="viridis")

    # 物理归因：基准工况下各热阻占比
    rb = base
    breakdown = {
        "r_conv_pct": 100 * rb.r_conv / rb.r_effective,
        "r_tim_pct": 100 * (base.layers[1].thickness / (base.layers[1].k * rb.width)) / rb.r_effective,
        "r_die_half_pct": 100 * (rb.die.thickness / (2 * rb.die.k * rb.width)) / rb.r_effective,
        "r_base_pct": 100 * (base.layers[0].thickness / (base.layers[0].k * rb.width)) / rb.r_effective,
    }
    # 极端工况：h 最大 + k_TIM 最小/最大，展示参数敏感度反转
    sens = {}
    for h in (h_list[0], h_list[-1]):
        vals = [next(r for r in rows if r["k_tim"] == k and r["h_conv"] == h)["t_j_steady"]
                for k in (k_list[0], k_list[-1])]
        sens[f"h={h:g}"] = {"tj_kmin": vals[0], "tj_kmax": vals[1],
                            "k_gain_K": vals[0] - vals[1],
                            "k_gain_pct": 100 * (vals[0] - vals[1]) / vals[0]}
    print("[热阻分解]", {k: round(v, 2) for k, v in breakdown.items()}, flush=True)
    print("[k 敏感度反转]", sens, flush=True)

    # ---------- B. 集总 Cauer 的适用边界（Bi / 频率扫描） ----------
    h_bi_list = [float(x) for x in cfg["sweep"]["h_bi_list"]]
    periods = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    validity = []
    for h in h_bi_list:
        c = base.with_params(h_conv=h)
        rec = {"h_conv": h, "bi_tim": bi_numbers(c)["TIM"], "periods": {}}
        for per in periods:
            # 矩阵指数精确传播：dt 只需整除半周期（保证步内功率恒定），
            # 每周期 40 步、跑 4 个周期，避免长周期下百万级无效小步。
            dt_exact = per / 40.0
            t_end = 4.0 * per
            cc = c.with_params(pulse_period=per, t_end=t_end)
            f2 = fine_network(cc, 64)
            l2 = lumped_network(cc)
            rf = f2.run(dt=dt_exact, t_end=t_end)
            rl = l2.run(dt=dt_exact, t_end=t_end)
            e = curve_error_full(rl, rf, cc.pulse_period, cc.t_amb)
            rec["periods"][f"{per:g}"] = {
                "peak_rel_err_pct": e["peak_rel_err_pct"],
                "swing_rel_err_pct": e["swing_rel_err_pct"],
                "rmse_rel_amp_pct": e["rmse_rel_amp_pct"],
                "swing_fine_K": e["swing_b_K"], "swing_lump_K": e["swing_a_K"],
            }
        validity.append(rec)
        print(f"[Bi 扫描] h={h:g} Bi_TIM={rec['bi_tim']:.3f} "
              f"摆幅误差@0.02s={rec['periods']['0.02']['swing_rel_err_pct']:.2f}% "
              f"@2s={rec['periods']['2']['swing_rel_err_pct']:.3f}%", flush=True)
    dump("lumped_validity.json", {"h_list": h_bi_list, "periods": periods,
                                  "records": validity,
                                  "resistance_breakdown_pct": breakdown,
                                  "k_sensitivity": sens})

    # 误差 vs h：用摆幅误差（对热容分布失真最敏感；峰值温度含环境温度会压小误差）
    err_hi = np.array([r["periods"]["0.02"]["swing_rel_err_pct"] for r in validity])
    err_lo = np.array([r["periods"]["2"]["swing_rel_err_pct"] for r in validity])
    bi_arr = np.array([r["bi_tim"] for r in validity])
    plot_error_vs_h(np.array(h_bi_list), err_hi, bi_arr,
                    "单节集总 Cauer 摆幅误差（快脉冲 0.02 s）与 5% 目标线",
                    fig / "bi_validity_highfreq.png")
    plot_error_vs_h(np.array(h_bi_list), err_lo, bi_arr,
                    "单节集总 Cauer 摆幅误差（慢脉冲 2 s）与 5% 目标线",
                    fig / "bi_validity_lowfreq.png")

    # 频率响应：基准 h 下摆幅 vs 周期
    rec0 = min(validity, key=lambda r: abs(r["h_conv"] - base.h_conv))
    swing_fine = np.array([rec0["periods"][f"{p:g}"]["swing_fine_K"] for p in periods])
    swing_lump = np.array([rec0["periods"][f"{p:g}"]["swing_lump_K"] for p in periods])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figx, ax = plt.subplots(figsize=(6.8, 4.2), dpi=120)
    ax.semilogx(periods, swing_fine, "o-", label="精细 Cauer (N=64)")
    ax.semilogx(periods, swing_lump, "s--", label="单节集总 Cauer (N=1)")
    ax.set_xlabel("脉冲周期 [s]")
    ax.set_ylabel("周期稳态结温摆幅 [K]")
    ax.set_title("热容网络的频率响应：集总模型在快脉冲下失效")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    figx.tight_layout()
    figx.savefig(fig / "freq_response.png")
    plt.close(figx)

    print(f"\n[02 完成] 用时 {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
