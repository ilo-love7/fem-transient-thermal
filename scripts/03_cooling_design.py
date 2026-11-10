"""03 散热方案评估与二维扩散热阻（集总热阻网络的适用边界）。"""
from __future__ import annotations

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
                 periodic_steady_metrics, fine_network, spreading_resistance)
from fem.plots import plot_field, plot_bars


def dump(path, obj):
    (ROOT / "results" / path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=float), encoding="utf-8")


def variant(base, name, *, h=None, k_tim=None, l_tim=None):
    layers = list(base.layers)
    if k_tim is not None or l_tim is not None:
        L = layers[1]
        layers[1] = replace(L, k=float(k_tim) if k_tim is not None else L.k,
                            thickness=float(l_tim) if l_tim is not None else L.thickness)
    kw = {"layers": tuple(layers)}
    if h is not None:
        kw["h_conv"] = float(h)
    return name, base.with_params(**kw)


def evaluate(case, *, t_end=8.0, dt=0.01):
    m = build_mesh(case)
    s = assemble(case, m)
    Ts = solve_steady(s, case.power_per_length)
    r = run_transient(s, case, dt=dt, t_end=t_end)
    met = periodic_steady_metrics(r, case)
    return {
        "t_j_steady": float(Ts[m.junction_node]),
        "t_peak": met["t_peak"], "t_trough": met["t_trough"],
        "swing": met["swing"], "t_avg": met["t_avg"],
        "r_effective": float(case.r_effective),
        "r_conv": float(case.r_conv),
        "analytic_tj_1d": float(case.t_junction_steady),
    }, Ts, m


def main():
    t0 = time.time()
    cfg = load_config(default_config_path())
    base = case_from_config(cfg)
    fig = ROOT / "figures"

    # ---------- A. 散热方案对比（1D 口径，保证可比） ----------
    plans = [
        variant(base, "基准"),
        variant(base, "强化对流 h×4", h=20000.0),
        variant(base, "高导热 TIM k×10", k_tim=50.0),
        variant(base, "减薄 TIM L/4", l_tim=0.05e-3),
        variant(base, "组合优化", h=20000.0, k_tim=50.0, l_tim=0.05e-3),
    ]
    rows = []
    for name, c in plans:
        met, _, _ = evaluate(c)
        rows.append({"plan": name, **met})
        print(f"[方案] {name:16s}: 稳态 {met['t_j_steady']:.2f} C / "
              f"峰值 {met['t_peak']:.2f} C / 摆幅 {met['swing']:.2f} K", flush=True)
    ref = rows[0]
    for r in rows:
        r["peak_reduction_pct"] = 100 * (ref["t_peak"] - r["t_peak"]) / ref["t_peak"]
        r["steady_reduction_pct"] = 100 * (ref["t_j_steady"] - r["t_j_steady"]) / ref["t_j_steady"]
    dump("cooling_design.json", {"plans": rows})
    plot_bars([r["plan"] for r in rows],
              np.array([r["t_peak"] for r in rows]),
              "周期稳态结温峰值 [°C]", "散热方案对比 · 脉冲工况结温峰值",
              fig / "cooling_design_peak.png", color="#d62728")
    plot_bars([r["plan"] for r in rows],
              np.array([r["peak_reduction_pct"] for r in rows]),
              "相对基准降幅 [%]", "散热方案改善幅度",
              fig / "cooling_design_gain.png", color="#2ca02c")

    # ---------- B. 二维扩散热阻：局部热源 + 集总网络低估量 ----------
    widths = [float(x) for x in cfg["design"]["source_widths"]]
    spread = []
    for w in widths:
        # 保证热源边界落在网格线上：nx=80, dx=0.125 mm
        c = base.with_params(heat_source_width=w)
        x0, x1 = c.heat_source_bounds
        dx = c.width / c.nx
        assert abs(x0 / dx - round(x0 / dx)) < 1e-9, f"热源边界未对齐网格: w={w}"
        met, Ts, m = evaluate(c)
        sp = spreading_resistance(c, met["t_j_steady"], c.t_junction_steady)
        fem_2d_peak = met["t_peak"]
        # 1D Cauer 对同一总功率的周期稳态峰值（假定热源铺满）
        c1 = base.with_params(heat_source_width=base.width)
        met1, _, _ = evaluate(c1)
        rec = {"w_mm": w * 1e3, "w_over_W": w / c.width,
               "t_j_steady_2d": met["t_j_steady"],
               "t_j_steady_1d_analytic": c.t_junction_steady,
               "t_peak_2d": fem_2d_peak, "t_peak_1d": met1["t_peak"],
               "cauer_steady_underestimate_K": met["t_j_steady"] - c.t_junction_steady,
               "cauer_peak_underestimate_K": fem_2d_peak - met1["t_peak"],
               "cauer_peak_underestimate_pct": 100 * (fem_2d_peak - met1["t_peak"])
               / max(met1["t_peak"] - c.t_amb, 1e-9),
               **{("spread_" + k): v for k, v in sp.items()}}
        spread.append(rec)
        print(f"[扩散热阻] w/W={w/c.width:.2f}: T_j,2D={met['t_j_steady']:.2f} C "
              f"(1D {c.t_junction_steady:.2f}), R_spread={sp['r_spread']:.6f} "
              f"({sp['r_spread_ratio_pct']:.1f}% of R_1D), "
              f"Cauer 峰值低估 {rec['cauer_peak_underestimate_K']:.2f} K", flush=True)
        if abs(w - min(widths)) < 1e-12:
            plot_field(m, Ts, c, "二维局部热源稳态温度场（w/W=0.2）",
                       fig / "steady_2d_field.png")
    dump("spreading_resistance.json", {"cases": spread})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    f, ax = plt.subplots(figsize=(6.8, 4.2), dpi=120)
    ax.plot([r["w_over_W"] for r in spread],
            [r["spread_r_spread_ratio_pct"] for r in spread], "o-", color="#d62728")
    ax.set_xlabel("热源宽度 / 芯片宽度  w/W")
    ax.set_ylabel("扩散热阻占一维总热阻 [%]")
    ax.set_title("二维扩散热阻随热源收窄而增大")
    ax.grid(alpha=0.3)
    ax.invert_xaxis()
    f.tight_layout()
    f.savefig(fig / "spreading_vs_width.png")
    plt.close(f)

    print(f"\n[03 完成] 用时 {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
