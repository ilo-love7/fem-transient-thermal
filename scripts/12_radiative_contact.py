"""12 v1.2 深化 F：辐射非线性边界、等效 h 线性化误差、接触热阻敏感度。

回应 v1.1 §9 未做项「辐射与接触热阻」。本轮用「等温升功率」协议
P = target_rise / R_conv，使不同 h_conv 下无辐射基准温升相同，从而干净地
分离出 h_rad/h_conv 的比值效应（否则低 h 组合会出现数万 K 的非物理温升）。

三个可证伪预设：
  H1 h_rad = 4*eps*sigma*T^3 只有 O(5-10) W/(m2K) 量级 ⇒ 强制对流
     (h=5000) 下辐射是伪需求，只有自然对流/近真空封装才需计入；
  H2 「在环境温度处线性化」会系统性低估辐射散热、高估结温；
  H3 接触热阻与 TIM 本体导热是两条独立优化路径，可定量比较边际收益。
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

from fem import (assemble, build_mesh, case_from_config,  # noqa: E402
                 default_config_path, load_config, solve_steady)
from fem.boundary_nonlinear import (
    STEFAN_BOLTZMANN, case_with_contact, contact_rise_analytic,
    effective_layer_k, radiative_conductance_local,
    solve_linearized_radiative_steady, solve_radiative_steady,
)

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
    v = cfg["v12"]["radiative_contact"]
    base = case_from_config(cfg).with_params(
        nx=int(v["nx"]), ny_per_layer=int(v["ny_per_layer"]))
    target_rise = float(v.get("target_rise_K", 300.0))
    eps_list = [float(x) for x in v["epsilon_list"]]
    h_list = [float(x) for x in v["h_list"]]
    scales = [float(x) for x in v["power_scale_list"]]
    out = {"protocol": {
        "nnode": build_mesh(base).nnode,
        "power_rule": "P = target_rise_K / R_conv * scale，使无辐射基准温升可比",
        "target_rise_K": target_rise,
        "sigma_W_m2K4": STEFAN_BOLTZMANN,
        "h_rad_at_ambient_eps09_W_m2K": float(
            radiative_conductance_local(base.t_amb, 0.9)),
        "note": "精确 Newton（T^4 边界）vs 环境温度处线性化 h0=4eps sigma T_amb^3",
    }}

    # --- 1) 辐射扫描：eps × h × 功率 ---
    rows = []
    for h in h_list:
        r_conv = 1.0 / (h * base.width)
        for scale in scales:
            p = target_rise / r_conv * scale
            c = base.with_params(h_conv=h, power_per_length=p)
            s = assemble(c, build_mesh(c))
            t_ref = float(solve_steady(s, p)[s.mesh.junction_node])
            for eps in eps_list:
                exact = solve_radiative_steady(s, c, eps)
                lin = solve_linearized_radiative_steady(s, c, eps)
                rows.append({
                    "h_conv_W_m2K": h, "power_W_per_m": p, "scale": scale,
                    "epsilon": eps,
                    "t_ref_no_rad_C": t_ref,
                    "t_exact_newton_C": exact.t_junction,
                    "t_linearized_C": lin,
                    "cooling_by_radiation_K": t_ref - exact.t_junction,
                    "linearization_err_K": lin - exact.t_junction,
                    "radiation_share": exact.radiation_share,
                    "q_rad_W_per_m": exact.q_rad_W_per_m,
                    "q_conv_W_per_m": exact.q_conv_W_per_m,
                    "energy_balance_err_W_per_m": (
                        exact.q_conv_W_per_m + exact.q_rad_W_per_m - p),
                    "newton_iterations": exact.iterations,
                    "converged": exact.converged,
                    "h_rad_at_bottom_W_m2K": float(
                        radiative_conductance_local(
                            exact.field[s.mesh.bottom_edges[0, 0]], eps)),
                })
    out["radiation_sweep"] = rows

    # 关键标度结论
    hi_h = max(h_list)
    forced = [r for r in rows
              if r["h_conv_W_m2K"] == hi_h and r["epsilon"] == max(eps_list)]
    natural = [r for r in rows
               if r["h_conv_W_m2K"] == min(h_list)
               and r["epsilon"] == max(eps_list)]
    out["radiation_regime"] = {
        "h_conv_forced": hi_h,
        "forced_share_max": max(r["radiation_share"] for r in forced),
        "forced_cooling_max_K": max(r["cooling_by_radiation_K"] for r in forced),
        "h_conv_natural": min(h_list),
        "natural_share_max": max(r["radiation_share"] for r in natural),
        "natural_cooling_max_K": max(r["cooling_by_radiation_K"] for r in natural),
        "linearization_max_err_K": max(r["linearization_err_K"] for r in rows),
        "linearization_bias": "正 = 线性化高估结温（低估散热）",
        "all_converged": all(r["converged"] for r in rows),
        "max_energy_balance_err_W_per_m": max(
            abs(r["energy_balance_err_W_per_m"]) for r in rows),
    }

    # --- 2) 接触热阻 vs TIM 本体导热 ---
    rc_list = [float(x) for x in v["rc_list"]]
    contact_rows = []
    for rc in rc_list:
        c2 = case_with_contact(base, rc)
        s2 = assemble(c2, build_mesh(c2))
        tj = float(solve_steady(s2, c2.power_per_length)[s2.mesh.junction_node])
        ana = contact_rise_analytic(base, rc)
        contact_rows.append({
            "r_contact_m2K_W": rc,
            "k_tim_eff_W_mK": ana["k_eff_W_mK"],
            "t_junction_fem_C": tj,
            "t_junction_analytic_C": ana["t_junction_C"],
            "rise_vs_baseline_K": ana["rise_vs_no_contact_K"],
        })
    out["contact_sweep"] = contact_rows

    tim_rows = []
    for k in [float(x) for x in v["k_tim_list"]]:
        layers = tuple(replace(L, k=k) if L.name == "TIM" else L
                       for L in base.layers)
        c2 = replace(base, layers=layers)
        s2 = assemble(c2, build_mesh(c2))
        tj = float(solve_steady(s2, c2.power_per_length)[s2.mesh.junction_node])
        tim_rows.append({"k_tim_W_mK": k, "t_junction_C": tj})
    out["tim_sweep"] = tim_rows
    base_tj = contact_rows[0]["t_junction_fem_C"]
    tim_gain = base_tj - min(r["t_junction_C"] for r in tim_rows)
    worst_rc = max(r["rise_vs_baseline_K"] for r in contact_rows)
    out["design_comparison"] = {
        "k_tim_5_to_50_cooling_K": tim_gain,
        "max_contact_penalty_K": worst_rc,
        "contact_penalty_over_tim_gain": worst_rc / max(tim_gain, 1e-9),
        "note": "接触热阻的恶化幅度 vs 把 TIM 本体做到 50 W/(mK) 的全部收益",
    }

    # ---------------- 图 ----------------
    fig, ax = plt.subplots(figsize=(8.4, 4.5), dpi=130)
    for eps in eps_list:
        if eps == 0.0:
            continue
        yy = [100 * r["radiation_share"] for r in rows
              if r["epsilon"] == eps and r["scale"] == 1.0]
        ax.plot(h_list, yy, "o-", lw=2, label=rf"$\varepsilon$={eps:g}")
    ax.axhline(5.0, color="gray", ls="--", lw=1, label="5% 显著性阈值")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$h_{conv}$ [W/(m$^2$K)]")
    ax.set_ylabel("辐射占总散热比例 [%]")
    ax.set_title(r"E15 $h_{rad}\sim 5$–10 W/(m$^2$K)：强制对流下辐射可忽略")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "v12_radiation_share.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.5), dpi=130)
    xs = [100 * r["radiation_share"] for r in rows if r["epsilon"] > 0]
    ys = [r["linearization_err_K"] for r in rows if r["epsilon"] > 0]
    ax.plot(xs, ys, "o", color="#d62728", ms=5, alpha=0.8)
    ax.axhline(0.0, color="gray", ls="--", lw=1)
    ax.set_xlabel("辐射占总散热比例 [%]")
    ax.set_ylabel("线性化近似的结温误差 [K]")
    ax.set_title("环境温度处线性化系统性高估结温")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "v12_radiation_linearization.png")
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=130)
    ax1.semilogx([max(r["r_contact_m2K_W"], 1e-9) for r in contact_rows],
                 [r["t_junction_fem_C"] for r in contact_rows], "o-",
                 color="#d62728", lw=2)
    ax1.set_xlabel(r"接触热阻 $r_c''$ [m$^2$K/W]")
    ax1.set_ylabel("稳态结温 [°C]")
    ax1.set_title("界面接触热阻的代价")
    ax1.grid(alpha=0.3, which="both")
    ax2.plot([r["k_tim_W_mK"] for r in tim_rows],
             [r["t_junction_C"] for r in tim_rows], "s-",
             color="#1f77b4", lw=2)
    ax2.set_xlabel("TIM 本体导热系数 [W/(m K)]")
    ax2.set_ylabel("稳态结温 [°C]")
    ax2.set_title("TIM 本体优化的收益")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "v12_contact_vs_tim.png")
    plt.close(fig)

    out["wall_s"] = time.time() - t0
    dump("v12_radiative_contact.json", out)
    reg = out["radiation_regime"]
    print("[12] h_rad@T_amb(eps=0.9) =",
          round(out["protocol"]["h_rad_at_ambient_eps09_W_m2K"], 3), "W/(m2K)")
    print("[12] 强制对流 h=5000 辐射占比上限 %:",
          round(100 * reg["forced_share_max"], 4), "/ 自然对流 h=10 上限 %:",
          round(100 * reg["natural_share_max"], 2))
    print("[12] 线性化最大结温误差 K:", round(reg["linearization_max_err_K"], 3),
          "；Newton 全部收敛:", reg["all_converged"])
    print("[12] TIM 5→50 降温 K:", round(tim_gain, 3),
          "；最大接触热阻恶化 K:", round(worst_rc, 3),
          "；比值", round(out["design_comparison"]["contact_penalty_over_tim_gain"], 2))
    print("[12] 用时", round(out["wall_s"], 1), "s；已写 JSON + 3 图")


if __name__ == "__main__":
    main()
