"""10 v1.2 深化 D：k(T) 瞬态 Newton/Picard 耦合与离散能量守恒审计。

回应 v1.1 §9 未做项「k(T) 与瞬态的 Newton 耦合」（v1.1 只有稳态 Picard）。
三个可证伪预设：
  H1 Newton 比 Picard 少迭代（强非线性时）；
  H2 能量守恒残差 1^T R 是免费的收敛诊断；
  H3 常功率下瞬态终值与 v1.1 独立实现的稳态 Picard 一致。
若 H1 在本工况不成立，如实记为负结果（弱非线性下 Newton 无收益）。
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

from fem import (build_mesh, case_from_config, default_config_path,  # noqa: E402
                 load_config, periodic_steady_metrics)
from fem.nonlinear import solve_nonlinear_steady  # noqa: E402
from fem.nonlinear_transient import solve_nonlinear_transient  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
RES = ROOT / "results"
FIG = ROOT / "figures"


def dump(name, obj):
    (RES / name).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")


def record(case, alpha, method, mesh, **kw):
    t0 = time.time()
    r = solve_nonlinear_transient(case, alpha, method=method, mesh=mesh, **kw)
    met = periodic_steady_metrics(r.result, case)
    return {
        "alpha": alpha, "method": method,
        "t_peak_C": met["t_peak"], "t_trough_C": met["t_trough"],
        "swing_K": met["swing"], "t_avg_C": met["t_avg"],
        "total_iterations": r.total_iterations,
        "max_iter_per_step": r.max_iterations_per_step,
        "mean_iter_per_step": r.mean_iterations_per_step,
        "max_energy_residual_W_per_m": r.max_energy_residual_W_per_m,
        "max_rel_energy_residual": r.max_relative_energy_residual,
        "converged": r.converged, "k_min": r.k_min, "k_max": r.k_max,
        "wall_s": time.time() - t0,
    }, r


def main():
    t0 = time.time()
    cfg = load_config(default_config_path())
    v = cfg["v12"]["nonlinear_transient"]
    base = case_from_config(cfg).with_params(
        nx=int(v["nx"]), ny_per_layer=int(v["ny_per_layer"]),
        dt=float(v["dt"]), t_end=float(v["t_end"]),
        pulse_period=2.0, duty=0.5, theta=1.0)
    mesh = build_mesh(base)
    out = {"protocol": {
        "nnode": mesh.nnode, "dt_s": base.dt, "t_end_s": base.t_end,
        "pulse_period_s": base.pulse_period, "theta": base.theta,
        "k_T_model": "k_e = k0_e*[1+alpha*(mean(T_e)-T_amb)]",
        "note": "alpha=0 严格退化为 v1.0 线性 theta 法；能量残差 = 1^T R",
    }}

    # --- A：alpha × method 全扫描 ---
    rows = []
    curves = {}
    for a in v["alpha_list"]:
        for method in ("newton", "picard"):
            rec, r = record(base, float(a), method, mesh)
            rows.append(rec)
            if method == "newton":
                curves[f"{a:+.3f}"] = r.result
    out["alpha_sweep"] = rows
    lin_peak = next(r["t_peak_C"] for r in rows
                    if r["alpha"] == 0.0 and r["method"] == "newton")
    for r in rows:
        r["peak_shift_vs_constant_K"] = r["t_peak_C"] - lin_peak

    # --- B：强非线性压力测试（Picard 是否失控） ---
    stress = []
    for a in v["alpha_stress_list"]:
        for method in ("newton", "picard"):
            try:
                rec, _ = record(base, float(a), method, mesh)
            except ValueError as exc:
                rec = {"alpha": float(a), "method": method,
                       "converged": False, "error": str(exc),
                       "note": "k(T)<=0，物性外推失效"}
            stress.append(rec)
    out["stress_test"] = stress

    # --- C：容差 → 迭代数/能量残差 ---
    tols = []
    for tol in v["tol_list"]:
        rec, _ = record(base, -0.005, "newton", mesh, tol=float(tol))
        rec["tol"] = float(tol)
        tols.append(rec)
    # 人为截断迭代（诊断对照）
    trunc, _ = record(base, -0.005, "picard", mesh, omega=0.4, max_iter=1)
    trunc["tol"] = "max_iter=1 截断"
    tols.append(trunc)
    out["tolerance_audit"] = tols

    # --- D：dt 扫描（大步长增强非线性） ---
    dts = []
    for dt in v["dt_list"]:
        c = replace(base, dt=float(dt))
        rec, _ = record(c, -0.005, "newton", mesh)
        rec["dt_s"] = float(dt)
        rec["n_steps"] = int(round(c.t_end / c.dt))
        dts.append(rec)
    out["dt_sweep"] = dts

    # --- E：常功率 → v1.1 稳态 Picard 独立交叉验证 ---
    c_const = replace(base, dt=0.02, t_end=12.0)
    const_power = lambda t: base.power_per_length  # noqa: E731
    nr = solve_nonlinear_transient(c_const, -0.003, method="newton",
                                   mesh=mesh, power_fn=const_power)
    steady = solve_nonlinear_steady(c_const, -0.003, mesh=mesh)
    out["steady_crosscheck"] = {
        "alpha": -0.003, "t_end_s": c_const.t_end,
        "transient_final_C": float(nr.result.t_junction[-1]),
        "v11_steady_picard_C": float(steady.t_junction),
        "abs_diff_K": float(nr.result.t_junction[-1] - steady.t_junction),
        "steady_iterations": steady.iterations,
        "transient_total_iterations": nr.total_iterations,
        "note": "两条独立装配/求解链（瞬态 Newton vs 稳态 Picard）互证",
    }

    # ---------------- 图 ----------------
    alphas = [float(a) for a in v["alpha_list"]]
    nw = [r for r in rows if r["method"] == "newton"]
    pc = [r for r in rows if r["method"] == "picard"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=130)
    ax1.plot(alphas, [r["total_iterations"] for r in nw], "o-",
             color="#1f77b4", lw=2, label="Newton（完整切线）")
    ax1.plot(alphas, [r["total_iterations"] for r in pc], "s--",
             color="#d62728", lw=1.8, label="Picard（欠松弛 0.8）")
    ax1.axhline(len(np.arange(0, base.t_end, base.dt)), color="gray",
                ls=":", lw=1, label="理论步数（每步 1 次）")
    ax1.set_xlabel(r"$\alpha$ [1/K]")
    ax1.set_ylabel("非线性迭代总次数")
    ax1.set_title("E13 迭代代价：Newton vs Picard")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)
    ax2.plot(alphas, [r["t_peak_C"] for r in nw], "o-",
             color="#1f77b4", lw=2, label="k(T) 瞬态 Newton")
    ax2.axhline(lin_peak, color="gray", ls=":", lw=1.4, label="常物性基准")
    ax2.set_xlabel(r"$\alpha$ [1/K]")
    ax2.set_ylabel("周期稳态峰值结温 [°C]")
    ax2.set_title("温度相关导热系数对峰值的影响")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "v12_newton_iterations.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.3), dpi=130)
    good = [r for r in tols if isinstance(r["tol"], float)]
    ax.plot([r["tol"] for r in good],
            [max(r["max_energy_residual_W_per_m"], 1e-14) for r in good],
            "o-", color="#2ca02c", lw=2, label="Newton（各容差）")
    ax.axhline(trunc["max_energy_residual_W_per_m"], color="#d62728",
               ls="--", lw=1.8, label="人为截断迭代")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("非线性收敛容差 tol")
    ax.set_ylabel("最大能量失衡 |1^T R| [W/m]")
    ax.set_title("E13 能量守恒残差 = 免费的收敛诊断")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "v12_energy_audit.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=130)
    mask_t = curves["+0.000"].t >= curves["+0.000"].t[-1] - base.pulse_period
    for key, col, ls in (("-0.005", "#1f77b4", "-"),
                         ("+0.000", "#555555", ":"),
                         ("+0.005", "#d62728", "--")):
        r = curves[key]
        m = r.t >= r.t[-1] - base.pulse_period
        ax.plot(r.t[m], r.t_junction[m], ls, color=col, lw=2,
                label=rf"$\alpha$={key} /K")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("结温 [°C]")
    ax.set_title("最后一周期结温：非线性物性 vs 常物性")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "v12_kT_transient.png")
    plt.close(fig)

    out["wall_s"] = time.time() - t0
    dump("v12_nonlinear_transient.json", out)
    print("[10] alpha 扫描迭代数 Newton/Picard:",
          [r["total_iterations"] for r in nw], "/",
          [r["total_iterations"] for r in pc])
    print("[10] 峰值偏移 K:", [round(r["peak_shift_vs_constant_K"], 3) for r in nw])
    print("[10] 稳态交叉验证差 K:", round(out["steady_crosscheck"]["abs_diff_K"], 5))
    print("[10] 截断迭代能量失衡 W/m:",
          round(trunc["max_energy_residual_W_per_m"], 3),
          "vs 收敛解", round(good[-1]["max_energy_residual_W_per_m"], 8))
    print("[10] 用时", round(out["wall_s"], 1), "s；已写 JSON + 3 图")


if __name__ == "__main__":
    main()
