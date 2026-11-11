"""11 v1.2 深化 E：给扩散热阻补热容的两阶段动态网络（快脉冲修正）。

回应 v1.1 §9 未做项「静态扩散修正的快脉冲版本（给扩散区补 RC 节）」。
v1.1 静态修正 T_j = T_1D + P(t)*R_sp 在快脉冲下把摆幅高估到 +73.6%
(T=0.02 s)，有效边界 T/tau_spread ≳ 35。本实验把扩散区改成一阶极点：

    C_sp d(DeltaT)/dt = P(t) - DeltaT/R_sp ,  T_j = T_1D,top + DeltaT

核心可证伪命题：**单极点能否跨周期泛化**。做法：
  (1) 复用 v1.1 同协议的轴对称 FEM 真值（6 个周期，避免重跑昂贵 CFD/FEM）；
  (2) 每个周期独立二分标定 tau_best；若 tau_best 随周期漂移，说明
      constriction 区是分布参数系统而非单极点；
  (3) 用单一周期标定的 tau_fixed 做**留一预测**，量化泛化误差；
  (4) 与「未修正 / v1.1 静态修正」三方对照，给出新的有效边界。
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
from fem import periodic_steady_metrics  # noqa: E402
from fem.axisym import (assemble_axisym, axisym_from_case,  # noqa: E402
                        run_axisym_transient)
from fem.cauer import CauerNetwork  # noqa: E402
from fem.spreading import (add_static_spreading,  # noqa: E402
                           equivalent_ladder_case)
from fem.two_stage import (build_two_stage, fit_tau_from_swing,  # noqa: E402
                           two_stage_swing)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
RES = ROOT / "results"
FIG = ROOT / "figures"


def dump(name, obj):
    (RES / name).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")


def v11_protocol(period: float):
    """严格复刻 scripts/06 的 period_sweep 时间窗与步长。"""
    return min(0.005, period / 40.0), max(6 * period, 0.4)


def main():
    t0 = time.time()
    cfg = load_config(default_config_path())
    v = cfg["v12"]["dynamic_diffusion"]
    n = int(v["n_cauer"])
    base = case_from_config(cfg)
    ref = json.loads((ROOT / str(v["v11_reference"])).read_text(encoding="utf-8"))
    ps = ref["period_sweep"]
    r_sp = float(ref["transient_correction"]["r_sp_K_W"])
    tau_phys = float(ps["tau_spread_s"])
    c_sp_phys = float(ps["c_spread_J_K"])
    q = float(v["source_ratio"])
    ax0 = axisym_from_case(base, q, nr=int(v["nr"]),
                           nz_per_layer=int(v["nz_per_layer"]))
    cal_period = float(v["calibration_period_s"])

    out = {"protocol": {
        "source_ratio": q, "r_sp_K_W": r_sp,
        "tau_spread_physical_s": tau_phys, "c_spread_J_per_K": c_sp_phys,
        "n_cauer_per_layer": n, "calibration_period_s": cal_period,
        "reference": str(v["v11_reference"]),
        "time_protocol": "dt=min(0.005,T/40), t_end=max(6T,0.4)，与 v1.1 完全一致",
        "power_W": ax0.total_power,
    }}

    # --- 1) 每个周期独立标定 tau_best ---
    fits = []
    for row in ps["rows"]:
        per = float(row["period_s"])
        dt, te = v11_protocol(per)
        fit = fit_tau_from_swing(ax0, r_sp, float(row["fem_swing_K"]),
                                 per, ax0.duty, n=n, dt=dt, t_end=te)
        fits.append({
            "period_s": per, "T_over_tau_spread": row["T_over_tau_spread"],
            "fem_swing_K": float(row["fem_swing_K"]),
            "fem_peak_C": float(row["fem_peak_C"]),
            "tau_best_s": fit["tau"], "tau_over_tau_phys": fit["tau"] / tau_phys,
            "bracket_ok": fit["bracket_ok"],
            "swing_tau_lo_K": fit["swing_lo"], "swing_tau_hi_K": fit["swing_hi"],
            "dt_s": dt, "t_end_s": te,
        })
    out["per_period_fit"] = fits
    valid = [f for f in fits if f["bracket_ok"] and np.isfinite(f["tau_best_s"])]
    ratios = [f["tau_over_tau_phys"] for f in valid]
    tau_fixed = float(next(f["tau_best_s"] for f in valid
                           if abs(f["period_s"] - cal_period) < 1e-9))
    out["single_pole_diagnosis"] = {
        "tau_fixed_s": tau_fixed,
        "tau_fixed_over_tau_phys": tau_fixed / tau_phys,
        "tau_best_min_s": min(f["tau_best_s"] for f in valid),
        "tau_best_max_s": max(f["tau_best_s"] for f in valid),
        "tau_best_spread_factor": (max(f["tau_best_s"] for f in valid)
                                   / min(f["tau_best_s"] for f in valid)),
        "ratio_min": min(ratios), "ratio_max": max(ratios),
        "note": "若 tau_best 随周期显著漂移，则扩散区不是单极点（分布参数）",
    }

    # --- 2) 四模型对照（同一时间协议） ---
    rows = []
    for f in fits:
        per, dt, te = f["period_s"], f["dt_s"], f["t_end_s"]
        c = replace(ax0, pulse_period=per, dt=dt, t_end=te)
        ladder = equivalent_ladder_case(c)
        raw = CauerNetwork(ladder, tuple([n] * ladder.n_layers)).run(
            dt=dt, t_end=te)
        static = add_static_spreading(raw, r_sp)
        dyn_fix = build_two_stage(c, r_sp, tau_fixed, n=n).run(
            c, dt=dt, t_end=te)
        dyn_best = build_two_stage(c, r_sp, f["tau_best_s"], n=n).run(
            c, dt=dt, t_end=te)
        m_raw = periodic_steady_metrics(raw, c)
        m_st = periodic_steady_metrics(static, c)
        m_fx = periodic_steady_metrics(dyn_fix, c)
        m_be = periodic_steady_metrics(dyn_best, c)
        fem_sw, fem_pk = f["fem_swing_K"], f["fem_peak_C"]

        def err(m, key="swing"):
            return (m[key] - (fem_sw if key == "swing" else fem_pk))

        rows.append({
            "period_s": per, "T_over_tau_spread": f["T_over_tau_spread"],
            "fem_peak_C": fem_pk, "fem_swing_K": fem_sw,
            "raw_swing_K": m_raw["swing"],
            "static_swing_K": m_st["swing"],
            "dynamic_fixed_swing_K": m_fx["swing"],
            "dynamic_perperiod_swing_K": m_be["swing"],
            "raw_swing_err_pct": 100 * err(m_raw) / fem_sw,
            "static_swing_err_pct": 100 * err(m_st) / fem_sw,
            "dynamic_fixed_swing_err_pct": 100 * err(m_fx) / fem_sw,
            "dynamic_perperiod_swing_err_pct": 100 * err(m_be) / fem_sw,
            "dynamic_fixed_peak_err_K": err(m_fx, "t_peak"),
            "static_peak_err_K": err(m_st, "t_peak"),
        })
    out["model_comparison"] = rows

    def min_valid(key):
        good = [r["period_s"] for r in rows if abs(r[key]) < 1.0]
        return min(good) if good else None

    out["validity_frontier"] = {
        "static_min_period_1pct_s": min_valid("static_swing_err_pct"),
        "dynamic_fixed_min_period_1pct_s": min_valid("dynamic_fixed_swing_err_pct"),
        "dynamic_perperiod_min_period_1pct_s": min_valid(
            "dynamic_perperiod_swing_err_pct"),
        "criterion": "|摆幅误差| < 1%",
    }

    # --- 3) 留一泛化误差（用单一 tau_fixed 预测所有周期） ---
    held = [r for r in rows if abs(r["period_s"] - cal_period) > 1e-9]
    out["leave_one_out"] = {
        "calibration_period_s": cal_period,
        "held_out_periods_s": [r["period_s"] for r in held],
        "max_abs_swing_err_pct": max(abs(r["dynamic_fixed_swing_err_pct"])
                                     for r in held),
        "rmse_swing_err_pct": float(np.sqrt(np.mean(
            [r["dynamic_fixed_swing_err_pct"] ** 2 for r in held]))),
        "max_abs_peak_err_K": max(abs(r["dynamic_fixed_peak_err_K"])
                                  for r in held),
        "note": "若留一误差远大于标定周期自身误差，说明单极点不可跨尺度泛化",
    }

    # ---------------- 图 ----------------
    xx = [r["T_over_tau_spread"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=130)
    for key, lab, col, ls in (
        ("raw_swing_err_pct", "未修正一维 Cauer", "#d62728", "--"),
        ("static_swing_err_pct", "v1.1 静态 R_sp 修正", "#ff7f0e", "-."),
        ("dynamic_fixed_swing_err_pct", "v1.2 动态 RC（单一 tau）", "#1f77b4", "-"),
        ("dynamic_perperiod_swing_err_pct", "v1.2 动态 RC（逐周期最优 tau）",
         "#2ca02c", ":"),
    ):
        ax.plot(xx, [abs(r[key]) for r in rows], ls, color=col, lw=2,
                marker="o", ms=4, label=lab)
    ax.axhline(1.0, color="gray", ls="--", lw=1, label="1% 工程阈值")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$T/\tau_{spread}$")
    ax.set_ylabel("摆幅误差 |Δ| [%]")
    ax.set_title("E14 快脉冲扩散修正：静态 vs 动态 RC 极点")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "v12_dynamic_spreading_error.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.3), dpi=130)
    ax.plot([f["period_s"] for f in valid],
            [f["tau_best_s"] * 1e3 for f in valid], "o-",
            color="#1f77b4", lw=2, label="逐周期二分标定 tau_best")
    ax.axhline(tau_phys * 1e3, color="#d62728", ls="--", lw=1.8,
               label=r"v1.1 物理定义 $\tau=R_{sp}C_{sp}$")
    ax.axhline(tau_fixed * 1e3, color="#2ca02c", ls=":", lw=1.8,
               label="留一验证所用 tau_fixed")
    ax.set_xscale("log")
    ax.set_xlabel("脉冲周期 T [s]")
    ax.set_ylabel(r"$\tau$ [ms]")
    ax.set_title("单极点假设检验：标定时间常数是否随周期漂移")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "v12_fitted_tau.png")
    plt.close(fig)

    # 快脉冲曲线（只重跑一次轴对称 FEM，用于直观对照）
    per = min(r["period_s"] for r in rows)
    dt, te = v11_protocol(per)
    c_fast = replace(ax0, pulse_period=per, dt=dt, t_end=te)
    fem = run_axisym_transient(assemble_axisym(c_fast), dt=dt, t_end=te)
    ladder = equivalent_ladder_case(c_fast)
    raw = CauerNetwork(ladder, tuple([n] * ladder.n_layers)).run(dt=dt, t_end=te)
    st = add_static_spreading(raw, r_sp)
    dy = build_two_stage(c_fast, r_sp, tau_fixed, n=n).run(
        c_fast, dt=dt, t_end=te)
    fig, ax = plt.subplots(figsize=(9, 4.3), dpi=130)
    for r, lab, col, ls in (
        (fem, "轴对称 FEM（真值）", "#111111", "-"),
        (st, "v1.1 静态修正", "#ff7f0e", "-."),
        (dy, "v1.2 动态 RC 修正", "#1f77b4", "--"),
    ):
        m = r.t >= r.t[-1] - 2 * per
        ax.plot(r.t[m], r.t_junction[m], ls, color=col, lw=1.9, label=lab)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("结温 [°C]")
    ax.set_title(f"T={per:g} s 快脉冲：动态极点补回了被静态模型高估的摆幅")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "v12_fast_pulse_curves.png")
    plt.close(fig)

    out["wall_s"] = time.time() - t0
    dump("v12_dynamic_diffusion.json", out)
    print("[11] tau_fixed =", round(tau_fixed * 1e3, 4), "ms;",
          "tau_phys =", round(tau_phys * 1e3, 4), "ms")
    print("[11] tau_best 跨周期漂移倍数:",
          round(out["single_pole_diagnosis"]["tau_best_spread_factor"], 3))
    print("[11] 摆幅误差 静态 %:",
          [round(r["static_swing_err_pct"], 2) for r in rows])
    print("[11] 摆幅误差 动态(单一 tau) %:",
          [round(r["dynamic_fixed_swing_err_pct"], 2) for r in rows])
    print("[11] 1% 有效边界 静态/动态 s:",
          out["validity_frontier"]["static_min_period_1pct_s"], "/",
          out["validity_frontier"]["dynamic_fixed_min_period_1pct_s"])
    print("[11] 留一最大摆幅误差 %:",
          round(out["leave_one_out"]["max_abs_swing_err_pct"], 2))
    print("[11] 用时", round(out["wall_s"], 1), "s；已写 JSON + 3 图")


if __name__ == "__main__":
    main()
